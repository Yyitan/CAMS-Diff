[English](README.md) | 简体中文

# CAMS-Diff：冻结 Stable Diffusion 2.1 的组合反馈探索（负结果与系统诊断）

> 基于 [CAMS](https://github.com/ybyangjing/CAMS)（[arXiv:2511.16378](https://arxiv.org/abs/2511.16378)）的改进研究 · 个人研究项目
>
> **重要说明：这是一条未成功的探索路线。** 实验明确表明：冻结 SD 2.1 的 denoising/ranking 反馈目标**可以被优化，但没有转化为 CZSL 指标提升**；推理期 SD reranking 只有微弱且数据集相关的信号。本 README 与 [`docs/CAMS_Diff_experiment_report.md`](docs/CAMS_Diff_experiment_report.md) 如实记录全部正负结果，供后续研究者避坑参考。

## 研究动机

最初的想法来自 I2CD 的 Disentangle-Compose-Disentangle 思路，以及"冻结扩散模型作为视觉关系 judge"的思想：Stable Diffusion 在预训练中见过海量 `attribute + object` 组合，能否让冻结 SD 2.1 对 CAMS 的 attribute/object/combination 特征提供额外监督或打分，从而提升组合零样本识别？为此先后尝试了两条路线：**训练期反馈**（v1–v3）与**推理期 reranking**，并用系统的消融回答了"为什么不行"。

## 方法

### 路线一：训练期扩散反馈（v1 → v3）

保持 CAMS 主体不变，加入冻结 SD 2.1 的 VAE + UNet + DDPMScheduler，只训练一个小型条件适配器：

1. `AttrObjectConditionAdapter` 把 CAMS 输出的 `f_attr`、`f_obj` 映射为 SD UNet 的 cross-attention 条件 tokens；
2. v1：正样本 denoising MSE（`L_diff = MSE(UNet(z_t, t, cond(f_attr,f_obj)), noise)`）；
3. v2：稳定化（`λ_diff` 0.1→0.01、3 epoch warmup、512 输入、checkpoint 排除冻结 SD 权重）；
4. v3：energy ranking——对同一图像、同一噪声 latent、同一 timestep 比较正条件与（属性/对象 shuffle 得到的）负条件的去噪能量，用 contrastive CE 训练。

### 路线二：推理期 SD 原生 prompt top-K reranking（evaluation-only）

不再训练任何东西：用 SD 原生文本编码器编码 `a photo of a {attr} {obj}`，对每张测试图像取 CAMS top-K 候选 pair，计算冻结 SD 的去噪能量 `E_SD`，按 `score_final = score_CAMS + α·zscore(−E_SD)` 重排，`α` 在 validation 上搜索。

| 组件 | 作用 | 代码位置 |
|---|---|---|
| **AttrObjectConditionAdapter** | 将 CAMS attr/obj 特征分别投影并拼成 SD cross-attention 条件 tokens | `model/diffusion_feedback.py` |
| **DiffusionFeedback** | 封装冻结 SD 2.1（VAE/UNet/调度器），提供 positive / energy-ranking 两种训练期反馈损失与内部诊断指标（`r_acc`、`e_neg−e_pos`） | `model/diffusion_feedback.py` + `patches_from_upstream/train.py.diff` |
| **SD top-K reranker** | 冻结 SD 原生文本条件的去噪能量作为 pair compatibility prior，重排 CAMS top-K 候选；支持多 timestep 平均与 prompt ensemble | `sd_rerank_eval.py`、`rerank_alpha_sweep.py` |

## 仓库内容

**本仓库只包含新增模块、配置、脚本与文档，不包含上游 CAMS 原始代码。** 对上游文件的修改以 unified diff 形式记录在 `patches_from_upstream/`。

```text
CAMS-Diff/
├── docs/
│   └── CAMS_Diff_experiment_report.md   # 完整实验报告：每一步的动机/设置/全部数字/结论
├── model/
│   └── diffusion_feedback.py            # AttrObjectConditionAdapter + DiffusionFeedback（冻结 SD2.1）
├── configs/
│   ├── mit_diff_v3.yml                  # v3 energy ranking（MIT-States）
│   ├── mit_diff_v3_lam001.yml           # λ=0.01 消融
│   ├── mit_diff_v3_lam0005.yml          # λ=0.005 消融
│   └── mit_diff_v3_detach.yml           # detach_features 消融（diffusion loss 不回传 CAMS 特征）
├── sd_rerank_eval.py                    # evaluation-only SD top-K reranking
├── rerank_alpha_sweep.py                # 复用缓存的 α 细扫（不重算 SD 能量）
├── scripts/                             # 训练消融与各 rerank 配置的运行脚本
└── patches_from_upstream/               # 对上游 CAMS 文件的修改记录
    ├── model__cams.py.diff              # forward_features / diffusion 反馈接口
    ├── model__configure_model.diff      # 显式冻结 SD 参数
    ├── train.py.diff                    # 扩散反馈训练循环
    └── configs__utzappos.yml.diff
```

## 实验结果（真实数据，均为负结果或弱信号）

CAMS 复现基线（5 训练种子均值，closed-world）：UT-Zappos HM 56.21 / AUC 44.39；MIT-States HM 39.89 / AUC 23.03。

### 训练期扩散反馈：目标可优化，指标不涨

| 版本 | UT-Zappos test AUC（基线 43.76） | MIT-States test AUC（基线 23.11） | 内部 ranking 指标 `r_acc` |
|---|---:|---:|---|
| v1 positive denoising | 41.93（**−1.83**） | – | – |
| v2 稳定化（λ=0.01 + warmup） | 42.86（−0.90） | – | best val 出现在 diffusion **未启用**的 warmup epoch |
| v3 energy ranking | 42.86（best checkpoint 仍是 warmup epoch） | 22.30（**−0.81**） | UT 0.730→0.966；MIT 0.700→0.951 |
| MIT 消融：λ=0.01 / 0.005 / detach | – | 22.37 / 22.30 / 22.38 | 0.927–0.951 |

诊断结论：把 `λ_diff` 调小、把 diffusion 梯度与 CAMS 特征 detach，都无法接近基线——排除了"λ 太大"与"梯度破坏主特征"两个解释。模型确实学会了让正组合能量低于负组合（`r_acc` → 0.95+），**但这个目标与 CZSL 的 pair ranking 目标不对齐**：adapter 可以学会"迎合"冻结 judge，而这种能力并不进入最终推理分数。

### 推理期 SD reranking：MIT 弱正信号，UT 无效

MIT-States closed-world test（基线 CAMS seed0：AUC 23.11 / HM 39.90）：

| 配置 | val-selected α | test AUC（val α） | test oracle AUC（仅诊断，不可作为方法结果） |
|---|---:|---:|---:|
| K20 / t=1 / 默认 prompt | 0.02 | 23.18（+0.07） | 23.48（α=0.20） |
| K50 / t=1 | 0.04 | 23.20 | 23.50 |
| K20 / t=4（多 timestep 平均） | 0.30 | 23.31 | 23.41 |
| K20 / t=1 / 4-prompt 平均 | 0.04 | 23.16 | 23.47 |
| obj_is_attr prompt / t=1 | 0.06 | 23.23 | 23.52 |

UT-Zappos closed-world（基线 AUC 43.76）：val 最优 α=0.26 在 test 上 AUC **降至 43.22**；按 AUC 看 SD reranking 在 UT 上没有价值——细粒度鞋类属性超出 SD 2.1 通用图文先验的分辨能力。

### 总结论

1. **负结果**：训练期把冻结 SD 的 denoising/ranking loss 接入 CAMS 不是可靠的 CZSL 监督；
2. **弱信号**：SD 原生文本能量在 MIT 上有 +0.07 AUC（val-selected）的微弱正贡献，test oracle 上限约 +0.42 AUC 但选择权重不稳定，UT 上为负；
3. 继续调 `λ_diff`、warmup、temperature、K 或 prompt 数量的预期收益很低；若要继续，应考虑把 SD prior 蒸馏为轻量可学习的 pair compatibility prior，而不是直接用 SD energy rerank。

全部命令、超参、逐步推理与更细的数字表见 [`docs/CAMS_Diff_experiment_report.md`](docs/CAMS_Diff_experiment_report.md)。

## 复现步骤

1. **克隆上游 CAMS 并应用补丁**：

   ```bash
   git clone https://github.com/ybyangjing/CAMS.git
   cd CAMS
   git apply /path/to/CAMS-Diff/patches_from_upstream/*.diff
   ```

2. **放入本仓库新增文件**（`model/`、`configs/`、`scripts/`、`sd_rerank_eval.py`、`rerank_alpha_sweep.py`）。

3. **权重与数据**：下载 diffusers 格式 Stable Diffusion 2.1 权重（config 的 `sd_model_path` 或 `--sd_model_path` 配置）；CLIP ViT-L/14 权重路径在 `clip_model` 字段；数据集放到 `data/` 下。训练期复现只需 CAMS config；reranking 复现还需要一个**未修改 CAMS** 训练出的 checkpoint 与对应 config（脚本默认读取 `output/cams_mit_seed0/final_model.pt`，用环境变量 `LOAD_MODEL` 覆盖）。

4. **训练期反馈复现**（预期结果：不低于 v3 报告值即可认为对齐）：

   ```bash
   python train.py --config configs/mit_diff_v3.yml --dataset_path data/mit-states
   bash scripts/run_mit_ablation.sh   # λ / detach 三组消融
   ```

   关注训练日志中的 `r_acc` 与 `e_neg - e_pos`：它们会持续变好，但 val/test CZSL 指标不会——这正是本仓库要记录的现象。

5. **推理期 reranking 复现**（evaluation-only，不训练）：

   ```bash
   LOAD_MODEL=output/cams_mit_seed0/final_model.pt \
   SD_MODEL=./pretrained/stable-diffusion-2-1 \
   bash scripts/run_mit_sd_rerank.sh
   python rerank_alpha_sweep.py   # 复用缓存的 α 细扫
   ```

## 引用

本工作基于 CAMS，如使用本仓库请同时引用：

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

本项目以 [MIT License](LICENSE) 发布（Copyright (c) 2025 Yyitan），仅覆盖本仓库新增的代码与文档。上游 [CAMS](https://github.com/ybyangjing/CAMS) 仓库当前未附带 LICENSE 文件，使用其代码请遵循上游仓库的声明；Stable Diffusion 2.1 权重遵循 Stability AI 的相应许可。
