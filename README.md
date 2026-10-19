English | [简体中文](README_zh.md)

# CAMS-Diff: Exploring Compositional Feedback from Frozen Stable Diffusion 2.1 (Negative Results and Systematic Diagnostics)

> Improvement research built on [CAMS](https://github.com/ybyangjing/CAMS) ([arXiv:2511.16378](https://arxiv.org/abs/2511.16378)) · Personal research project
>
> **Important: this is an unsuccessful research direction.** The experiments show clearly that the denoising/ranking feedback objective of frozen SD 2.1 **can be optimized but does not translate into CZSL metric gains**; inference-time SD reranking provides only a weak, dataset-dependent signal. This README and [`docs/CAMS_Diff_experiment_report.md`](docs/CAMS_Diff_experiment_report.md) faithfully record all positive and negative results, so that future researchers can avoid the same pitfalls.

## Motivation

The initial idea came from I2CD's Disentangle-Compose-Disentangle idea, together with the notion of "a frozen diffusion model as a visual-relation judge": Stable Diffusion has seen vast numbers of `attribute + object` combinations during pretraining — could frozen SD 2.1 provide extra supervision or scoring for CAMS's attribute/object/combination features and thereby improve compositional zero-shot recognition? Two routes were tried in succession — **training-time feedback** (v1–v3) and **inference-time reranking** — with systematic ablations answering "why doesn't it work".

## Method

### Route 1: training-time diffusion feedback (v1 → v3)

CAMS's main body stays unchanged; the VAE + UNet + DDPMScheduler of frozen SD 2.1 are added, and only a small conditioning adapter is trained:

1. `AttrObjectConditionAdapter` maps CAMS's `f_attr`, `f_obj` into cross-attention conditioning tokens for the SD UNet;
2. v1: positive-sample denoising MSE (`L_diff = MSE(UNet(z_t, t, cond(f_attr,f_obj)), noise)`);
3. v2: stabilization (`λ_diff` 0.1→0.01, 3-epoch warmup, 512 input, checkpoints exclude the frozen SD weights);
4. v3: energy ranking — for the same image, the same noise latent, and the same timestep, compare the denoising energy of the positive condition against a negative one (obtained by shuffling attributes/objects), trained with contrastive CE.

### Route 2: inference-time SD native-prompt top-K reranking (evaluation-only)

Nothing is trained: SD's native text encoder encodes `a photo of a {attr} {obj}`; for each test image, take CAMS's top-K candidate pairs, compute the frozen SD denoising energy `E_SD`, and rerank by `score_final = score_CAMS + α·zscore(−E_SD)`, with `α` searched on validation.

| Component | Role | Code location |
|---|---|---|
| **AttrObjectConditionAdapter** | Projects CAMS's attr/obj features separately and concatenates them into SD cross-attention conditioning tokens | `model/diffusion_feedback.py` |
| **DiffusionFeedback** | Wraps frozen SD 2.1 (VAE/UNet/scheduler), providing the positive and energy-ranking training-time feedback losses plus internal diagnostic metrics (`r_acc`, `e_neg−e_pos`) | `model/diffusion_feedback.py` + `patches_from_upstream/train.py.diff` |
| **SD top-K reranker** | Uses the denoising energy under frozen SD's native text conditioning as a pair-compatibility prior to rerank CAMS's top-K candidates; supports multi-timestep averaging and prompt ensembles | `sd_rerank_eval.py`, `rerank_alpha_sweep.py` |

## Repository contents

**This repository only contains new modules, configurations, scripts, and documentation.** The `patches_from_upstream/` directory records modifications made to upstream files as unified diffs; the upstream CAMS code itself is not included.

```text
CAMS-Diff/
├── docs/
│   └── CAMS_Diff_experiment_report.md   # Full experiment report: motivation / setup / all numbers / conclusions at every step
├── model/
│   └── diffusion_feedback.py            # AttrObjectConditionAdapter + DiffusionFeedback (frozen SD 2.1)
├── configs/
│   ├── mit_diff_v3.yml                  # v3 energy ranking (MIT-States)
│   ├── mit_diff_v3_lam001.yml           # λ=0.01 ablation
│   ├── mit_diff_v3_lam0005.yml          # λ=0.005 ablation
│   └── mit_diff_v3_detach.yml           # detach_features ablation (diffusion loss does not backprop into CAMS features)
├── sd_rerank_eval.py                    # Evaluation-only SD top-K reranking
├── rerank_alpha_sweep.py                # Fine α sweep reusing the cache (no SD-energy recomputation)
├── scripts/                             # Run scripts for the training ablations and each rerank configuration
└── patches_from_upstream/               # Records of modifications to upstream CAMS files
    ├── model__cams.py.diff              # forward_features / diffusion feedback interface
    ├── model__configure_model.diff      # Explicit freezing of SD parameters
    ├── train.py.diff                    # Diffusion-feedback training loop
    └── configs__utzappos.yml.diff
```

## Experimental results (real data; all negative results or weak signals)

CAMS reproduction baselines (mean over 5 training seeds, closed-world): UT-Zappos HM 56.21 / AUC 44.39; MIT-States HM 39.89 / AUC 23.03.

### Training-time diffusion feedback: the objective is optimizable, the metrics don't move

| Version | UT-Zappos test AUC (baseline 43.76) | MIT-States test AUC (baseline 23.11) | Internal ranking metric `r_acc` |
|---|---:|---:|---|
| v1 positive denoising | 41.93 (**−1.83**) | – | – |
| v2 stabilization (λ=0.01 + warmup) | 42.86 (−0.90) | – | best val occurred in a warmup epoch where diffusion was **not yet enabled** |
| v3 energy ranking | 42.86 (best checkpoint still a warmup epoch) | 22.30 (**−0.81**) | UT 0.730→0.966; MIT 0.700→0.951 |
| MIT ablations: λ=0.01 / 0.005 / detach | – | 22.37 / 22.30 / 22.38 | 0.927–0.951 |

Diagnosis: shrinking `λ_diff` and detaching the diffusion gradients from CAMS features both fail to approach the baseline — ruling out "λ too large" and "gradients corrupt the main features" as explanations. The model does learn to make positive-pair energies lower than negative ones (`r_acc` → 0.95+), **but this objective is not aligned with CZSL's pair-ranking objective**: the adapter can learn to "please" the frozen judge, and that capability does not enter the final inference scores.

### Inference-time SD reranking: weak positive signal on MIT, no effect on UT

MIT-States closed-world test (baseline CAMS seed0: AUC 23.11 / HM 39.90):

| Configuration | val-selected α | test AUC (val α) | test oracle AUC (diagnostic only, not a method result) |
|---|---:|---:|---:|
| K20 / t=1 / default prompt | 0.02 | 23.18 (+0.07) | 23.48 (α=0.20) |
| K50 / t=1 | 0.04 | 23.20 | 23.50 |
| K20 / t=4 (multi-timestep averaging) | 0.30 | 23.31 | 23.41 |
| K20 / t=1 / 4-prompt average | 0.04 | 23.16 | 23.47 |
| obj_is_attr prompt / t=1 | 0.06 | 23.23 | 23.52 |

UT-Zappos closed-world (baseline AUC 43.76): the val-optimal α=0.26 **drops** test AUC to 43.22; in AUC terms, SD reranking has no value on UT — fine-grained footwear attributes exceed the discriminative power of SD 2.1's general image-text prior.

### Overall conclusions

1. **Negative result**: plugging frozen-SD denoising/ranking losses into CAMS training is not a reliable CZSL supervision signal;
2. **Weak signal**: SD native-text energy gives a marginal positive contribution of +0.07 AUC (val-selected) on MIT, with a test-oracle ceiling of about +0.42 AUC but unstable selection weights, and a negative contribution on UT;
3. The expected returns from further tuning of `λ_diff`, warmup, temperature, K, or the number of prompts are low; if continuing at all, one should consider distilling the SD prior into a lightweight learnable pair-compatibility prior rather than reranking directly with SD energy.

For all commands, hyperparameters, step-by-step reasoning, and finer number tables, see [`docs/CAMS_Diff_experiment_report.md`](docs/CAMS_Diff_experiment_report.md).

## Reproduction

1. **Clone upstream CAMS and apply the patches**:

   ```bash
   git clone https://github.com/ybyangjing/CAMS.git
   cd CAMS
   git apply /path/to/CAMS-Diff/patches_from_upstream/*.diff
   ```

2. **Drop in the new files from this repository** (`model/`, `configs/`, `scripts/`, `sd_rerank_eval.py`, `rerank_alpha_sweep.py`).

3. **Weights and data**: download Stable Diffusion 2.1 weights in diffusers format (configured via the config's `sd_model_path` or `--sd_model_path`); the CLIP ViT-L/14 checkpoint path goes in the `clip_model` field; place the datasets under `data/`. Training-time reproduction needs only a CAMS config; reranking reproduction additionally needs a checkpoint trained with **unmodified CAMS** together with its config (the scripts read `output/cams_mit_seed0/final_model.pt` by default; override with the `LOAD_MODEL` environment variable).

4. **Training-time feedback reproduction** (expected outcome: matching the v3 reported values or better counts as aligned):

   ```bash
   python train.py --config configs/mit_diff_v3.yml --dataset_path data/mit-states
   bash scripts/run_mit_ablation.sh   # the λ / detach ablations, three groups
   ```

   Watch `r_acc` and `e_neg - e_pos` in the training logs: they keep improving, but the val/test CZSL metrics do not — which is precisely the phenomenon this repository documents.

5. **Inference-time reranking reproduction** (evaluation-only, no training):

   ```bash
   LOAD_MODEL=output/cams_mit_seed0/final_model.pt \
   SD_MODEL=./pretrained/stable-diffusion-2-1 \
   bash scripts/run_mit_sd_rerank.sh
   python rerank_alpha_sweep.py   # fine α sweep reusing the cache
   ```

## Citation

This work is built on CAMS; if you use this repository, please also cite:

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE) (Copyright (c) 2025 Yyitan), which covers only the new code and documentation added in this repository. The upstream [CAMS](https://github.com/ybyangjing/CAMS) repository currently ships no LICENSE file; follow the upstream repository's terms when using its code. The Stable Diffusion 2.1 weights remain subject to Stability AI's respective license.
