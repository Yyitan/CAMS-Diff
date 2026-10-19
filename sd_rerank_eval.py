import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import test
from dataset import CompositionDataset
from flags import parser as base_parser
from model.configure_model import configure_model
from utils import load_args, set_seed


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation-only Stable Diffusion top-K reranking for CAMS."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--load_model", required=True)
    parser.add_argument("--sd_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--phase", choices=["val", "test", "both"], default="both")
    parser.add_argument("--topk_pairs", type=int, default=20)
    parser.add_argument("--num_timesteps", type=int, default=1)
    parser.add_argument("--alpha_grid", default="0,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--prompt_template", default="a photo of a {attr} {obj}")
    parser.add_argument(
        "--prompt_templates",
        default=None,
        help="Optional '||'-separated prompt templates. Overrides --prompt_template.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--sd_batch_size", type=int, default=8)
    parser.add_argument("--sd_image_size", type=int, default=512)
    parser.add_argument("--sd_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--eval_open_world", action="store_true")
    return parser.parse_args()


def build_cams_args(cli_args):
    args = base_parser.parse_args([])
    load_args(cli_args.config, args)
    args.dataset_path = cli_args.dataset_path
    args.load_model = cli_args.load_model
    args.use_diffusion = False
    if cli_args.eval_batch_size is not None:
        args.eval_batch_size = cli_args.eval_batch_size
    return args


def format_stats(stats):
    keys = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    return {key: round(float(stats[key]), 4) for key in keys}


def evaluate_logits(dataset, logits, attr_gt, obj_gt, pair_gt):
    evaluator = test.Evaluator(dataset)
    stats = test.test(dataset, evaluator, logits, attr_gt, obj_gt, pair_gt)
    return format_stats(stats)


def make_dataset(args, phase, max_samples=None, open_world=False):
    dataset = CompositionDataset(
        args.dataset_path,
        phase=phase,
        split=args.splitname,
        open_world=open_world,
    )
    if max_samples is None:
        return dataset
    keep = min(max_samples, len(dataset))
    dataset.data = dataset.data[:keep]
    return dataset


def get_prompt_templates(cli_args):
    if cli_args.prompt_templates:
        templates = [item.strip() for item in cli_args.prompt_templates.split("||")]
        return [item for item in templates if item]
    return [cli_args.prompt_template]


def load_cams_model(args, dataset, device):
    model, _ = configure_model(args, dataset)
    model.to(device)
    state = torch.load(args.load_model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class StableDiffusionEnergy:
    def __init__(self, model_path, image_size=512, batch_size=8, dtype="fp16", device="cuda"):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.model_path = model_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(device)
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=True,
        )
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.clip_mean = CLIP_MEAN.to(device)
        self.clip_std = CLIP_STD.to(device)

    def encode_prompts(self, prompts, batch_size=256):
        chunks = []
        with torch.no_grad():
            for start in tqdm(range(0, len(prompts), batch_size), desc="SD text"):
                batch = prompts[start : start + batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(self.device)
                chunks.append(self.text_encoder(tokens)[0].detach().cpu())
        return torch.cat(chunks, dim=0)

    def clip_to_sd_pixels(self, images):
        rgb = images.float() * self.clip_std + self.clip_mean
        rgb = rgb.clamp(0.0, 1.0)
        sd_pixels = rgb * 2.0 - 1.0
        if sd_pixels.shape[-1] != self.image_size or sd_pixels.shape[-2] != self.image_size:
            sd_pixels = F.interpolate(
                sd_pixels,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return sd_pixels.to(dtype=self.dtype)

    def encode_latents(self, images):
        pixels = self.clip_to_sd_pixels(images)
        posterior = self.vae.encode(pixels).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    def target_from_noise(self, latents, noise, timesteps):
        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        return noise

    def topk_energy(self, images, top_indices, pair_text_embeddings, num_timesteps=1):
        images = images.to(self.device)
        top_indices = top_indices.to(self.device)
        text_embeddings = pair_text_embeddings.to(self.device, dtype=self.dtype)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.unsqueeze(0)
        batch_size, topk = top_indices.shape
        num_prompt_templates = text_embeddings.shape[0]
        energy_sum = torch.zeros(batch_size, topk, device=self.device)

        with torch.no_grad():
            latents = self.encode_latents(images)
            for _ in range(num_timesteps):
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=self.device,
                    dtype=torch.long,
                )
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                target = self.target_from_noise(latents, noise, timesteps)

                flat_pair_idx = top_indices.reshape(-1)
                flat_latents = noisy_latents[:, None].expand(-1, topk, -1, -1, -1).reshape(
                    batch_size * topk, *noisy_latents.shape[1:]
                )
                flat_target = target[:, None].expand(-1, topk, -1, -1, -1).reshape(
                    batch_size * topk, *target.shape[1:]
                )
                flat_timesteps = timesteps[:, None].expand(-1, topk).reshape(-1)

                prompt_energy_sum = torch.zeros(batch_size, topk, device=self.device)
                for prompt_idx in range(num_prompt_templates):
                    flat_conditions = text_embeddings[prompt_idx, flat_pair_idx]
                    energies = []
                    for start in range(0, flat_latents.shape[0], self.batch_size):
                        end = start + self.batch_size
                        pred = self.unet(
                            flat_latents[start:end],
                            flat_timesteps[start:end],
                            encoder_hidden_states=flat_conditions[start:end],
                        ).sample
                        energy = (pred.float() - flat_target[start:end].float()).pow(2)
                        energies.append(energy.flatten(1).mean(1))
                    prompt_energy_sum += torch.cat(energies, dim=0).view(batch_size, topk)
                energy_sum += prompt_energy_sum / float(num_prompt_templates)

        return (energy_sum / float(num_timesteps)).detach().cpu()


def compute_or_load_cache(
    phase,
    cli_args,
    cams_args,
    model,
    sd_energy,
    pair_text_embeddings,
    device,
):
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{phase}_k{cli_args.topk_pairs}_t{cli_args.num_timesteps}.pt"
    if cache_path.exists() and not cli_args.overwrite_cache:
        print(f"Loading cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    dataset = make_dataset(
        cams_args,
        phase=phase,
        max_samples=cli_args.max_samples,
        open_world=cli_args.eval_open_world,
    )
    base_logits, attr_gt, obj_gt, pair_gt = test.predict_logits(model, dataset, cams_args)
    evaluator = test.Evaluator(dataset)
    closed_mask = evaluator.closed_mask
    masked_logits = base_logits.clone()
    masked_logits[:, ~closed_mask] = -1e10
    top_indices = masked_logits.topk(cli_args.topk_pairs, dim=1).indices

    loader = DataLoader(dataset, batch_size=cams_args.eval_batch_size, shuffle=False)
    all_energy = []
    offset = 0
    for batch in tqdm(loader, desc=f"SD energy {phase}"):
        images = batch[0]
        bsz = images.shape[0]
        batch_top = top_indices[offset : offset + bsz]
        all_energy.append(
            sd_energy.topk_energy(
                images,
                batch_top,
                pair_text_embeddings,
                num_timesteps=cli_args.num_timesteps,
            )
        )
        offset += bsz

    cache = {
        "phase": phase,
        "pairs": dataset.pairs,
        "base_logits": base_logits.cpu(),
        "top_indices": top_indices.cpu(),
        "sd_energy": torch.cat(all_energy, dim=0).cpu(),
        "attr_gt": attr_gt.cpu(),
        "obj_gt": obj_gt.cpu(),
        "pair_gt": pair_gt.cpu(),
        "closed_mask": closed_mask.cpu(),
    }
    torch.save(cache, cache_path)
    print(f"Saved cache: {cache_path}")
    return cache


def apply_rerank(base_logits, top_indices, sd_energy, alpha):
    reranked = base_logits.clone()
    sd_score = -sd_energy
    sd_score = (sd_score - sd_score.mean(dim=1, keepdim=True)) / (
        sd_score.std(dim=1, keepdim=True).clamp_min(1e-6)
    )
    rows = torch.arange(base_logits.shape[0]).unsqueeze(1)
    reranked[rows, top_indices] = base_logits[rows, top_indices] + alpha * sd_score
    return reranked


def evaluate_cache(cache, cams_args, phase, alpha):
    dataset = make_dataset(
        cams_args,
        phase=phase,
        max_samples=cache["base_logits"].shape[0],
        open_world=False,
    )
    logits = apply_rerank(
        cache["base_logits"],
        cache["top_indices"],
        cache["sd_energy"],
        alpha,
    )
    return evaluate_logits(
        dataset,
        logits,
        cache["attr_gt"],
        cache["obj_gt"],
        cache["pair_gt"],
    )


def main():
    cli_args = parse_args()
    set_seed(cli_args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cams_args = build_cams_args(cli_args)

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CompositionDataset(
        cams_args.dataset_path,
        phase="train",
        split=cams_args.splitname,
        open_world=False,
    )
    model = load_cams_model(cams_args, train_dataset, device)

    pair_dataset = make_dataset(cams_args, phase="val", open_world=cli_args.eval_open_world)
    prompt_templates = get_prompt_templates(cli_args)
    prompt_groups = [
        [
            template.format(attr=attr.replace("_", " "), obj=obj.replace("_", " "))
            for attr, obj in pair_dataset.pairs
        ]
        for template in prompt_templates
    ]
    sd_energy = StableDiffusionEnergy(
        cli_args.sd_model_path,
        image_size=cli_args.sd_image_size,
        batch_size=cli_args.sd_batch_size,
        dtype=cli_args.sd_dtype,
        device=device,
    )
    pair_text_embeddings = torch.stack(
        [sd_energy.encode_prompts(prompts) for prompts in prompt_groups],
        dim=0,
    )
    cli_args.prompt_templates_used = prompt_templates

    phases = ["val", "test"] if cli_args.phase == "both" else [cli_args.phase]
    caches = {
        phase: compute_or_load_cache(
            phase,
            cli_args,
            cams_args,
            model,
            sd_energy,
            pair_text_embeddings,
            device,
        )
        for phase in phases
    }

    alphas = [float(x) for x in cli_args.alpha_grid.split(",") if x.strip()]
    results = {"settings": vars(cli_args), "results": {}}
    best_alpha = alphas[0]
    if "val" in caches:
        best_auc = None
        results["results"]["val"] = {}
        for alpha in alphas:
            stats = evaluate_cache(caches["val"], cams_args, "val", alpha)
            results["results"]["val"][str(alpha)] = stats
            print(f"VAL alpha={alpha}: {stats}")
            if best_auc is None or stats["AUC"] > best_auc:
                best_auc = stats["AUC"]
                best_alpha = alpha

    if "test" in caches:
        results["results"]["test"] = {}
        for alpha in alphas:
            stats = evaluate_cache(caches["test"], cams_args, "test", alpha)
            results["results"]["test"][str(alpha)] = stats
            print(f"TEST alpha={alpha}: {stats}")
        print(f"BEST_VAL_ALPHA {best_alpha}")
        if str(best_alpha) in results["results"]["test"]:
            print(f"TEST_AT_BEST_VAL_ALPHA {results['results']['test'][str(best_alpha)]}")

    result_path = output_dir / "rerank_results.json"
    with result_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {result_path}")


if __name__ == "__main__":
    main()
