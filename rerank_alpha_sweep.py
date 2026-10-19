import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import test
from dataset import CompositionDataset
from flags import parser as base_parser
from utils import load_args, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep alpha values from cached SD reranking energies."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--topk_pairs", type=int, default=20)
    parser.add_argument("--num_timesteps", type=int, default=1)
    parser.add_argument("--alpha_grid", default=None)
    parser.add_argument("--alpha_start", type=float, default=0.0)
    parser.add_argument("--alpha_end", type=float, default=0.3)
    parser.add_argument("--alpha_step", type=float, default=0.02)
    parser.add_argument("--select_metric", default="AUC")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_args(cli_args):
    args = base_parser.parse_args([])
    load_args(cli_args.config, args)
    args.dataset_path = cli_args.dataset_path
    return args


def make_dataset(args, phase, size):
    dataset = CompositionDataset(
        args.dataset_path,
        phase=phase,
        split=args.splitname,
        open_world=False,
    )
    dataset.data = dataset.data[:size]
    return dataset


def alpha_values(cli_args):
    if cli_args.alpha_grid:
        return [float(x) for x in cli_args.alpha_grid.split(",") if x.strip()]
    count = int(round((cli_args.alpha_end - cli_args.alpha_start) / cli_args.alpha_step))
    return [
        round(cli_args.alpha_start + i * cli_args.alpha_step, 10)
        for i in range(count + 1)
    ]


def apply_rerank(base_logits, top_indices, sd_energy, alpha):
    reranked = base_logits.clone()
    sd_score = -sd_energy
    sd_score = (sd_score - sd_score.mean(dim=1, keepdim=True)) / (
        sd_score.std(dim=1, keepdim=True).clamp_min(1e-6)
    )
    rows = torch.arange(base_logits.shape[0]).unsqueeze(1)
    reranked[rows, top_indices] = base_logits[rows, top_indices] + alpha * sd_score
    return reranked


def format_stats(stats):
    keys = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    return {key: round(float(stats[key]), 6) for key in keys}


def evaluate_cache(cache, args, phase, alpha):
    dataset = make_dataset(args, phase, cache["base_logits"].shape[0])
    logits = apply_rerank(
        cache["base_logits"],
        cache["top_indices"],
        cache["sd_energy"],
        alpha,
    )
    evaluator = test.Evaluator(dataset)
    stats = test.test(
        dataset,
        evaluator,
        logits,
        cache["attr_gt"],
        cache["obj_gt"],
        cache["pair_gt"],
    )
    return format_stats(stats)


def rank_diagnostics(cache, alpha):
    base = cache["base_logits"]
    top = cache["top_indices"]
    energy = cache["sd_energy"]
    gt = cache["pair_gt"]
    rows = torch.arange(base.shape[0])
    cover = (top == gt[:, None]).any(1)
    base_rank = (base > base[rows, gt][:, None]).sum(1) + 1
    reranked = apply_rerank(base, top, energy, alpha)
    rank = (reranked > reranked[rows, gt][:, None]).sum(1) + 1
    return {
        "topk_gt_coverage": round(float(cover.float().mean()), 6),
        "top1": round(float(rank.eq(1).float().mean()), 6),
        "mean_gt_rank": round(float(rank.float().mean()), 6),
        "median_gt_rank": round(float(rank.float().median()), 6),
        "improved_all": round(float(((rank < base_rank) & cover).float().mean()), 6),
        "worsened_all": round(float(((rank > base_rank) & cover).float().mean()), 6),
    }


def main():
    cli_args = parse_args()
    set_seed(cli_args.seed)
    args = build_args(cli_args)

    cache_dir = Path(cli_args.cache_dir)
    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    caches = {}
    for phase in ["val", "test"]:
        cache_path = cache_dir / f"{phase}_k{cli_args.topk_pairs}_t{cli_args.num_timesteps}.pt"
        caches[phase] = torch.load(cache_path, map_location="cpu", weights_only=False)

    alphas = alpha_values(cli_args)
    results = {
        "settings": vars(cli_args),
        "alphas": alphas,
        "results": {"val": {}, "test": {}},
        "diagnostics": {"val": {}, "test": {}},
    }

    rows = []
    for alpha in alphas:
        for phase in ["val", "test"]:
            stats = evaluate_cache(caches[phase], args, phase, alpha)
            diag = rank_diagnostics(caches[phase], alpha)
            results["results"][phase][str(alpha)] = stats
            results["diagnostics"][phase][str(alpha)] = diag
            row = {"phase": phase, "alpha": alpha}
            row.update(stats)
            row.update(diag)
            rows.append(row)
            print(f"{phase.upper()} alpha={alpha}: {stats} | {diag}")

    select_metric = cli_args.select_metric
    best_alpha = max(
        alphas,
        key=lambda a: results["results"]["val"][str(a)][select_metric],
    )
    results["best_val"] = {
        "metric": select_metric,
        "alpha": best_alpha,
        "val": results["results"]["val"][str(best_alpha)],
        "test": results["results"]["test"][str(best_alpha)],
    }
    best_test_alpha = max(
        alphas,
        key=lambda a: results["results"]["test"][str(a)][select_metric],
    )
    results["best_test_oracle"] = {
        "metric": select_metric,
        "alpha": best_test_alpha,
        "test": results["results"]["test"][str(best_test_alpha)],
    }

    json_path = output_dir / f"alpha_sweep_k{cli_args.topk_pairs}_t{cli_args.num_timesteps}.json"
    csv_path = output_dir / f"alpha_sweep_k{cli_args.topk_pairs}_t{cli_args.num_timesteps}.csv"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"BEST_VAL {results['best_val']}")
    print(f"BEST_TEST_ORACLE {results['best_test_oracle']}")
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
