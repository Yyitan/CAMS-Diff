import torch
from torch import nn
import torch.nn.functional as F


class AttrObjectConditionAdapter(nn.Module):
    def __init__(self, feature_dim, cross_attention_dim, num_tokens=8, hidden_dim=2048, dropout=0.0):
        super().__init__()
        attr_tokens = max(1, num_tokens // 2)
        obj_tokens = max(1, num_tokens - attr_tokens)
        self.attr_tokens = attr_tokens
        self.obj_tokens = obj_tokens
        self.cross_attention_dim = cross_attention_dim

        self.attr_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attr_tokens * cross_attention_dim),
        )
        self.obj_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, obj_tokens * cross_attention_dim),
        )
        self.token_type = nn.Parameter(torch.zeros(1, attr_tokens + obj_tokens, cross_attention_dim))
        self.out_norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, attr_features, obj_features):
        bsz = attr_features.shape[0]
        attr_tokens = self.attr_proj(attr_features).view(
            bsz, self.attr_tokens, self.cross_attention_dim
        )
        obj_tokens = self.obj_proj(obj_features).view(
            bsz, self.obj_tokens, self.cross_attention_dim
        )
        tokens = torch.cat([attr_tokens, obj_tokens], dim=1)
        tokens = self.out_norm(tokens + self.token_type.to(tokens.dtype))
        return tokens


class DiffusionFeedback(nn.Module):
    """Frozen Stable Diffusion denoising feedback for CAMS attribute/object features."""

    def __init__(self, args, feature_dim):
        super().__init__()
        try:
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        except ImportError as exc:
            raise ImportError("diffusers is required when use_diffusion=True") from exc

        self.args = args
        self.sd_model_path = getattr(
            args,
            "sd_model_path",
            "./pretrained/stable-diffusion-2-1",
        )
        self.lambda_diff = float(getattr(args, "lambda_diff", 0.1))
        self.use_rank_loss = bool(getattr(args, "use_diffusion_rank_loss", False))
        self.rank_weight = float(getattr(args, "diffusion_rank_weight", 0.1))
        self.rank_margin = float(getattr(args, "diffusion_rank_margin", 0.05))
        self.max_batch_size = int(getattr(args, "diffusion_batch_size", 4))
        self.condition_mode = getattr(args, "diffusion_condition_mode", "attr_obj")
        self.image_size = int(getattr(args, "diffusion_image_size", 512))
        self.objective = getattr(args, "diffusion_objective", "positive")
        self.energy_temperature = float(getattr(args, "diffusion_energy_temperature", 0.05))
        self.negative_mode = getattr(args, "diffusion_negative_mode", "both")
        self.detach_features = bool(getattr(args, "diffusion_detach_features", False))
        self.last_stats = {}

        dtype_name = getattr(args, "diffusion_dtype", "fp16")
        self.model_dtype = torch.float16 if dtype_name == "fp16" else torch.float32

        self.vae = AutoencoderKL.from_pretrained(
            self.sd_model_path,
            subfolder="vae",
            torch_dtype=self.model_dtype,
            local_files_only=True,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.sd_model_path,
            subfolder="unet",
            torch_dtype=self.model_dtype,
            local_files_only=True,
        )
        self.scheduler = DDPMScheduler.from_pretrained(
            self.sd_model_path,
            subfolder="scheduler",
            local_files_only=True,
        )

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()

        cross_attention_dim = int(self.unet.config.cross_attention_dim)
        self.condition_adapter = AttrObjectConditionAdapter(
            feature_dim=feature_dim,
            cross_attention_dim=cross_attention_dim,
            num_tokens=int(getattr(args, "diffusion_num_condition_tokens", 8)),
            hidden_dim=int(getattr(args, "diffusion_condition_hidden_dim", 2048)),
            dropout=float(getattr(args, "diffusion_condition_dropout", 0.0)),
        )

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.unet.eval()
        return self

    def _select_subset(self, images, attr_features, obj_features):
        if self.max_batch_size <= 0 or images.shape[0] <= self.max_batch_size:
            return images, attr_features, obj_features
        idx = torch.randperm(images.shape[0], device=images.device)[: self.max_batch_size]
        return images[idx], attr_features[idx], obj_features[idx]

    def _clip_to_sd_pixels(self, images):
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
        return sd_pixels.to(dtype=self.model_dtype)

    def _encode_latents(self, images):
        sd_pixels = self._clip_to_sd_pixels(images)
        with torch.no_grad():
            posterior = self.vae.encode(sd_pixels).latent_dist
            latents = posterior.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def _sample_noisy_latents(self, latents):
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents, noise, timesteps

    def _target_from_noise(self, latents, noise, timesteps):
        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        return noise

    def _diffusion_energy(self, noisy_latents, target, timesteps, condition_tokens):
        condition_tokens = condition_tokens.to(dtype=self.model_dtype)
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=condition_tokens,
        ).sample
        return (noise_pred.float() - target.float()).pow(2).flatten(1).mean(1)

    def _negative_permutation(self, batch_size, device):
        if batch_size < 2:
            return None
        perm = torch.randperm(batch_size, device=device)
        if torch.any(perm == torch.arange(batch_size, device=device)):
            perm = torch.roll(torch.arange(batch_size, device=device), shifts=1)
        return perm

    def forward(self, images, attr_features, obj_features):
        images, attr_features, obj_features = self._select_subset(
            images, attr_features, obj_features
        )
        if self.detach_features:
            attr_features = attr_features.detach()
            obj_features = obj_features.detach()
        latents = self._encode_latents(images)
        noisy_latents, noise, timesteps = self._sample_noisy_latents(latents)
        target = self._target_from_noise(latents, noise, timesteps)

        pos_condition = self.condition_adapter(attr_features, obj_features)
        pos_energy = self._diffusion_energy(noisy_latents, target, timesteps, pos_condition)

        if self.objective == "positive" or not self.use_rank_loss or latents.shape[0] < 2:
            loss = pos_energy.mean()
            self.last_stats = {
                "objective": "positive",
                "pos_energy": float(pos_energy.detach().mean().cpu()),
                "neg_energy": 0.0,
                "rank_acc": 0.0,
            }
            return loss

        perm = self._negative_permutation(latents.shape[0], latents.device)
        neg_energies = []
        if self.negative_mode in ("attr", "both"):
            neg_attr_condition = self.condition_adapter(attr_features[perm], obj_features)
            neg_energies.append(
                self._diffusion_energy(noisy_latents, target, timesteps, neg_attr_condition)
            )
        if self.negative_mode in ("obj", "both"):
            neg_obj_condition = self.condition_adapter(attr_features, obj_features[perm])
            neg_energies.append(
                self._diffusion_energy(noisy_latents, target, timesteps, neg_obj_condition)
            )

        neg_energy = torch.stack(neg_energies, dim=1)
        if self.objective == "energy_ce":
            logits = torch.cat([pos_energy[:, None], neg_energy], dim=1)
            logits = -logits / self.energy_temperature
            targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = F.relu(self.rank_margin + pos_energy[:, None] - neg_energy).mean()

        rank_acc = (pos_energy[:, None] < neg_energy).float().mean()
        self.last_stats = {
            "objective": self.objective,
            "pos_energy": float(pos_energy.detach().mean().cpu()),
            "neg_energy": float(neg_energy.detach().mean().cpu()),
            "rank_acc": float(rank_acc.detach().cpu()),
        }
        return loss
