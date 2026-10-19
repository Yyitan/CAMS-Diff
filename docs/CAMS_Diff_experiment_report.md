# CAMS-Diff 阶段实验整理与下一步方案

整理日期：2026-07-09

项目目录：本仓库（CAMS-Diff）

原始 CAMS 复现目录：官方 CAMS 仓库的本地克隆

## 1. 实验背景

我们的目标是基于 CAMS 做一个生成模型增强的 CZSL 方法。最初的想法来自 I2CD 的 Disentangle-Compose-Disentangle 思路，以及 VRDiff 中“冻结扩散模型作为视觉关系反馈/分类器”的思想。

核心问题是：能不能把冻结 Stable Diffusion 引入 CAMS，让它对属性特征、物体特征和组合特征提供额外监督，从而提升组合零样本识别。

目前已经完成的路线是：

1. 保持 CAMS 主体结构不变。
2. 加入冻结 SD2.1 的 VAE 和 UNet。
3. 用 CAMS 的 attribute/object visual features 经过一个 condition adapter 映射成 SD cross-attention condition tokens。
4. 用真实图像 latent 的 denoising error 作为训练反馈。
5. 逐步验证 positive denoising loss、energy ranking loss、小 lambda、detach features 等变体。

目前结论比较清楚：SD denoising/ranking 目标本身可以被优化，但它没有转化成 CZSL 指标提升。

## 2. Baseline：原始 CAMS 复现结果

原始 CAMS 的复现结果已经比较稳定，和论文报告略有差距，但可以作为可靠 baseline。

### UT-Zappos Closed-World

| seed | S | U | HM | AUC |
|---:|---:|---:|---:|---:|
| 0 | 68.62 | 76.10 | 56.56 | 43.76 |
| 1 | 69.40 | 75.67 | 55.16 | 44.10 |
| 2 | 68.62 | 74.83 | 55.48 | 43.40 |
| 3 | 70.58 | 75.52 | 57.08 | 45.43 |
| 4 | 68.62 | 76.57 | 56.76 | 45.25 |
| mean | 69.17 | 75.74 | 56.21 | 44.39 |

### MIT-States Closed-World

| seed | S | U | HM | AUC |
|---:|---:|---:|---:|---:|
| 0 | 50.25 | 54.12 | 39.90 | 23.11 |
| 1 | 51.51 | 53.10 | 40.33 | 23.30 |
| 2 | 50.00 | 53.79 | 39.94 | 22.84 |
| 3 | 50.55 | 53.41 | 39.48 | 22.93 |
| 4 | 50.38 | 53.56 | 39.81 | 22.96 |
| mean | 50.54 | 53.60 | 39.89 | 23.03 |

## 3. 第一版：Positive Diffusion Feedback

Git commit：`26a13bc baseline diffusion feedback for CAMS`

主要配置/输出目录：`output/utzappos_diff`

### 相比 CAMS 做了什么改变

第一版是在 CAMS 上直接加入冻结 Stable Diffusion denoising feedback。

主要改动：

1. 新增 `model/diffusion_feedback.py`。
2. 加载本地 SD2.1：
   - VAE：把真实图像编码成 latent。
   - UNet：预测噪声。
   - DDPMScheduler：采样 timestep 和加噪。
3. 冻结 SD 的 VAE 和 UNet，只训练一个 `AttrObjectConditionAdapter`。
4. 用 CAMS image encoder 输出的 attribute feature 和 object feature 作为条件输入。
5. adapter 将 `f_attr, f_obj` 映射成 SD UNet 的 cross-attention tokens。
6. diffusion loss 是正样本 denoising MSE：

   ```text
   L_diff = MSE(UNet(z_t, t, condition(f_attr, f_obj)), noise)
   ```

7. 总 loss：

   ```text
   L_total = L_CAMS + lambda_diff * L_diff
   ```

第一版的直觉是：如果属性和物体特征组合得合理，那么 SD 在真实图像 latent 上的 denoising error 应该更低；这样可以迫使 CAMS 学到更好的 attribute-object compositional representation。

### 实验验证结果

数据集：UT-Zappos

设置：Closed-World

| 方法 | S | U | HM | AUC | attr_acc | obj_acc |
|---|---:|---:|---:|---:|---:|---:|
| CAMS seed0 | 68.62 | 76.10 | 56.56 | 43.76 | - | - |
| Diff v1 | 64.91 | 75.25 | 54.55 | 41.93 | 50.27 | 73.13 |

val best：

| best val epoch | val S | val U | val HM | val AUC |
|---:|---:|---:|---:|---:|
| 2 | 71.84 | 75.05 | 64.55 | 50.63 |

### 结果分析

第一版效果变差比较明显：

1. Test AUC 从 `43.76` 降到 `41.93`。
2. Seen accuracy 从 `68.62` 降到 `64.91`。
3. Unseen accuracy 也没有提升。

这说明直接加入 positive denoising MSE 不是一个可靠的 CZSL 监督。可能原因：

1. SD denoising error 关注的是图像 latent 的噪声预测，不一定对应 CZSL 的属性/物体分类边界。
2. condition adapter 可能只是在学习“怎么让 UNet denoise error 下降”，而不是学习更可组合的 attr/object 表示。
3. `lambda_diff=0.1` 对 CAMS 主任务可能过强。
4. UT-Zappos 是细粒度鞋类数据集，SD 的通用视觉先验对细粒度属性可能不够敏感。

第一版结论：positive denoising feedback 可以跑通，但直接加到 CAMS 训练里会伤害主任务。

## 4. 第二版：稳定化 Positive Feedback

Git commit：`e0f9766 stabilize diffusion feedback training`

主要配置/输出目录：`output/utzappos_diff_v2`

### 相比第一版改了什么

第二版主要目标不是换思想，而是降低第一版对 CAMS 的破坏。

主要改动：

1. 降低 diffusion loss 权重：

   ```text
   lambda_diff: 0.1 -> 0.01
   ```

2. 加入 warmup：

   ```text
   diffusion_warmup_epochs: 3
   ```

   即显示的 epoch 1、2、3 不加 diffusion loss，从 epoch 4 开始 diffusion active。

3. 将 diffusion 输入图像尺寸稳定到 SD 原始需求：

   ```text
   diffusion_image_size: 512
   ```

4. 修复保存目录不存在的问题：

   ```text
   os.makedirs(args.save_path, exist_ok=True)
   ```

5. checkpoint 中排除冻结 SD VAE/UNet 权重，避免保存超大模型和加载问题。

6. 显式冻结 SD 的 VAE/UNet，避免原 CAMS 参数筛选逻辑误解冻部分 SD norm 参数。

### 实验验证结果

数据集：UT-Zappos

设置：Closed-World

| 方法 | S | U | HM | AUC | attr_acc | obj_acc |
|---|---:|---:|---:|---:|---:|---:|
| CAMS seed0 | 68.62 | 76.10 | 56.56 | 43.76 | - | - |
| Diff v1 | 64.91 | 75.25 | 54.55 | 41.93 | 50.27 | 73.13 |
| Diff v2 | 70.09 | 74.19 | 54.72 | 42.86 | 51.20 | 74.06 |

val best：

| best val epoch | diffusion active? | val S | val U | val HM | val AUC |
|---:|---|---:|---:|---:|---:|
| 3 | No | 71.61 | 74.54 | 65.09 | 50.35 |

### 结果分析

第二版比第一版稳定，seen accuracy 提升明显，但 AUC 和 HM 仍然没有超过 CAMS。

关键现象是：best val 出现在 epoch 3，而 epoch 3 还在 warmup，没有启用 diffusion。

这说明：

1. v2 的最终测试结果主要来自“没有 diffusion 的 CAMS warmup checkpoint”。
2. diffusion active 之后，val AUC 没有超过 epoch 3。
3. 降低 lambda 和 warmup 可以避免第一版那种明显破坏，但不能证明 diffusion 有正贡献。

第二版结论：稳定化策略有效降低了伤害，但 diffusion 本身没有带来提升。

## 5. 第三版：Diffusion Energy Ranking Feedback

Git commit：`5cc1095 add diffusion energy ranking feedback`

主要配置/输出目录：

```text
output/utzappos_diff_v3
output/mit_diff_v3
```

### 相比第二版改了什么

第三版从 positive denoising MSE 改成 energy ranking / contrastive feedback。

第二版只要求正组合 denoise error 低，但没有告诉模型“错误组合应该更差”。第三版加入正负组合比较：

1. 正条件：

   ```text
   condition(f_attr, f_obj)
   ```

2. 负条件：

   ```text
   condition(f_attr_shuffled, f_obj)
   condition(f_attr, f_obj_shuffled)
   ```

3. 对同一张图像、同一个 noisy latent、同一个 timestep，计算：

   ```text
   E_pos = denoising_error(z_t, condition(f_attr, f_obj))
   E_neg_attr = denoising_error(z_t, condition(f_attr_shuffled, f_obj))
   E_neg_obj = denoising_error(z_t, condition(f_attr, f_obj_shuffled))
   ```

4. 使用 energy CE：

   ```text
   logits = [-E_pos, -E_neg_attr, -E_neg_obj] / temperature
   target = positive condition
   ```

5. 记录训练内部指标：

   ```text
   e_pos, e_neg, r_acc
   ```

其中 `r_acc` 表示正条件 energy 是否低于负条件 energy。

第三版的直觉是：如果 SD 能作为组合合理性的判别器，那么正确 attr-object 条件应该比错误组合更适合解释真实图像 latent。

### UT-Zappos 实验结果

| 方法 | S | U | HM | AUC | attr_acc | obj_acc |
|---|---:|---:|---:|---:|---:|---:|
| CAMS seed0 | 68.62 | 76.10 | 56.56 | 43.76 | - | - |
| Diff v2 | 70.09 | 74.19 | 54.72 | 42.86 | 51.20 | 74.06 |
| Diff v3 | 70.09 | 74.19 | 54.72 | 42.86 | 51.20 | 74.06 |

val best：

| best val epoch | diffusion active? | val S | val U | val HM | val AUC |
|---:|---|---:|---:|---:|---:|
| 3 | No | 71.61 | 74.54 | 65.09 | 50.35 |

训练内部 diffusion ranking 指标：

| epoch | r_acc | e_neg - e_pos |
|---:|---:|---:|
| 4 | 0.730 | 0.0178 |
| 15 | 0.966 | 0.1211 |

### UT-Zappos 结果分析

v3 的 UT 结果和 v2 完全一样，不是因为 v3 没有生效，而是因为最终加载的 best checkpoint 仍然来自 epoch 3。

epoch 3 仍然是 warmup 阶段，所以：

1. final test 实际用的是 diffusion active 之前的 checkpoint。
2. epoch 4 之后 diffusion ranking 确实开始工作，`r_acc` 从 `0.73` 涨到 `0.966`。
3. 但 active 后的 val AUC 没有超过 epoch 3。

这说明第三版的 ranking objective 能被优化，但在 UT 上没有提升验证指标。

### MIT-States 实验结果

数据集：MIT-States

设置：Closed-World test

| 方法 | S | U | HM | AUC | attr_acc | obj_acc |
|---|---:|---:|---:|---:|---:|---:|
| CAMS seed0 | 50.25 | 54.12 | 39.90 | 23.11 | - | - |
| CAMS mean | 50.54 | 53.60 | 39.89 | 23.03 | - | - |
| Diff v3 | 49.33 | 53.22 | 39.42 | 22.30 | 37.95 | 55.91 |

val best：

| best val epoch | diffusion active? | val S | val U | val HM | val AUC |
|---:|---|---:|---:|---:|---:|
| 5 | Yes | 51.90 | 58.38 | 42.92 | 26.29 |

训练内部 diffusion ranking 指标：

| epoch | r_acc | e_neg - e_pos |
|---:|---:|---:|
| 4 | 0.700 | 0.0185 |
| 15 | 0.951 | 0.1331 |

### MIT-States 结果分析

MIT 和 UT 有一个区别：MIT 的 best val 出现在 epoch 5，已经是 diffusion active 阶段。

但是最终 test 仍然低于 CAMS：

1. Test AUC：`23.11 -> 22.30`，下降 `0.81`。
2. Test S：`50.25 -> 49.33`，下降 `0.92`。
3. Test U：`54.12 -> 53.22`，下降 `0.90`。

同时，内部 ranking 指标明显变好：

1. `r_acc` 最后一轮达到 `0.951`。
2. `e_neg - e_pos` 不断变大。

这说明 SD energy ranking 不是没有被模型学到；它确实学会了让正组合 energy 低于负组合。问题在于这个目标和最终 CZSL 分类指标不对齐。

第三版结论：energy ranking 比 positive loss 更合理，也确实能被优化，但仍然没有带来 test 提升。

## 6. 第四阶段：MIT Ablation 诊断实验

相关 commit：

```text
28510c8 add MIT diffusion ablation configs
30ae31c add MIT ablation runner
```

主要配置/输出目录：

```text
configs/mit_diff_v3_lam001.yml      -> output/mit_diff_v3_lam001
configs/mit_diff_v3_lam0005.yml     -> output/mit_diff_v3_lam0005
configs/mit_diff_v3_detach.yml      -> output/mit_diff_v3_detach
```

### 相比第三版改了什么

这一步不是提出新方法，而是为了诊断第三版失败原因。

我们主要验证两个假设：

1. 是不是 `lambda_diff=0.05` 太大，diffusion loss 干扰主任务？
2. 是不是 diffusion loss 直接回传到 CAMS 主特征，导致主分类空间被破坏？

因此做了三组实验：

1. `lambda_diff=0.01`
2. `lambda_diff=0.005`
3. `detach_features=True`

其中 `detach_features=True` 表示：

```python
attr_features = attr_features.detach()
obj_features = obj_features.detach()
```

这样 diffusion loss 只训练 condition adapter，不直接回传到 CAMS image features。

### 实验结果

MIT-States Closed-World test：

| 方法 | Test S | Test U | Test HM | Test AUC | attr_acc | obj_acc |
|---|---:|---:|---:|---:|---:|---:|
| CAMS seed0 | 50.25 | 54.12 | 39.90 | 23.11 | - | - |
| CAMS mean | 50.54 | 53.60 | 39.89 | 23.03 | - | - |
| v3 `lambda=0.05` | 49.33 | 53.22 | 39.42 | 22.30 | 37.95 | 55.91 |
| `lambda=0.01` | 49.54 | 53.24 | 39.46 | 22.37 | 37.90 | 55.81 |
| `lambda=0.005` | 49.41 | 53.23 | 39.42 | 22.30 | 37.91 | 55.81 |
| `detach_features` | 49.58 | 53.24 | 39.50 | 22.38 | 37.94 | 55.81 |

Best val：

| 方法 | Best Val Epoch | diffusion active? | Best Val AUC |
|---|---:|---|---:|
| v3 `lambda=0.05` | 5 | Yes | 26.29 |
| `lambda=0.01` | 5 | Yes | 26.20 |
| `lambda=0.005` | 5 | Yes | 26.25 |
| `detach_features` | 5 | Yes | 26.23 |

训练内部 ranking 指标：

| 方法 | final r_acc | final e_neg - e_pos |
|---|---:|---:|
| v3 `lambda=0.05` | 0.951 | 0.1331 |
| `lambda=0.01` | 0.927 | 0.1108 |
| `lambda=0.005` | 0.944 | 0.1051 |
| `detach_features` | 0.942 | 0.1089 |

### 结果分析

这组 ablation 给出了比较明确的诊断结论。

第一，不是 lambda 太大导致的主要问题。

`lambda=0.05` 降到 `0.01/0.005` 之后，test AUC 只从 `22.30` 到 `22.37/22.30`，变化极小，仍然低于 CAMS seed0 和 CAMS mean。

第二，不是 diffusion 梯度直接破坏 CAMS 主特征导致的主要问题。

`detach_features=True` 之后，test AUC 是 `22.38`，只比 v3 高 `0.08`，仍然没有接近 CAMS baseline。

第三，diffusion objective 本身确实被优化。

所有 diffusion 版本最后的 `r_acc` 都在 `0.927 ~ 0.951`。这说明模型学会了让正组合 denoising energy 小于负组合 denoising energy。

综合判断：

当前问题不是训练不稳定，也不是 loss 权重没有调好，而是 SD denoising/ranking objective 与 CZSL 的最终 pair classification/ranking objective 不对齐。

## 7. 当前阶段总判断

目前这条路线可以概括为：

```text
CAMS feature -> adapter -> SD condition tokens -> denoising/ranking loss
```

这条路线的主要问题是：adapter 可以学习一个让 SD energy ranking 变好的条件空间，但这个条件空间不一定改善 CAMS 的 attr/object/composition 判别空间。

更直白地说：

1. SD 被用作一个 frozen judge。
2. 但是我们训练了一个 adapter 去“迎合”这个 judge。
3. 这个 adapter 的成功不等于 CAMS 的组合分类能力提升。
4. 所以会出现 `r_acc` 很高、但 CZSL test AUC 不涨的情况。

这也解释了为什么 detach features 后变化仍然很小：如果 diffusion loss 主要训练 adapter，而 adapter 的判别能力不参与最终推理，那么它对 final pair score 的帮助天然有限。

因此，继续在当前形式上调 `lambda_diff`、warmup 或 temperature，收益可能很低。

## 8. 下一步建议：把 SD 从训练正则改成组合可行性 prior / reranker

### 8.1 为什么要换方向

CZSL 最终做的是 pair ranking：

```text
score(image, attr, obj)
```

CAMS 的最终预测依赖 attribute/object/composition logits，以及 open-world/closed-world 下的 bias、feasibility、calibration。

而我们当前的 diffusion loss 优化的是：

```text
denoising_error(image_latent, condition(attr_feature, obj_feature))
```

这两个目标之间隔了一层 adapter，而且 diffusion loss 并不直接出现在最终推理分数里。

所以更合理的方式是：不要让 SD 间接改变 CAMS feature，而是让 SD 直接参与最终 pair score 或 pair feasibility。

新的目标应该变成：

```text
score_final(image, attr, obj)
= score_CAMS(image, attr, obj) + alpha * score_SD(image, attr, obj)
```

其中：

```text
score_SD = - denoising_energy_SD(image, text_prompt(attr, obj))
```

也就是把 SD 当作一个冻结的、无需训练的 image-pair compatibility scorer。

这样 SD 的作用和 CZSL 的目标更一致：它直接帮助判断“这张图像是否符合这个属性-物体组合”。

### 8.2 核心方案：SD Native Text Prompt Reranking

当前 v1-v3 用的是：

```text
CAMS visual features -> learned adapter -> SD condition tokens
```

下一步建议改成：

```text
attr/object pair text -> SD native text encoder -> SD condition tokens
```

例如 pair prompt：

```text
a photo of a {attr} {object}
```

对于一张真实图像和一个候选 pair，计算 SD denoising energy：

```text
E_SD(image, attr, obj)
= MSE(UNet(z_t, t, text_condition("a photo of a {attr} {object}")), noise)
```

然后：

```text
score_SD(image, attr, obj) = -E_SD(image, attr, obj)
```

最后和 CAMS 分数融合：

```text
score_final = score_CAMS + alpha * normalize(score_SD)
```

这里的关键变化是：

1. 不训练 adapter，避免 adapter 学 shortcut。
2. 不直接改 CAMS feature，避免主任务被无关 denoising loss 干扰。
3. SD score 直接作用在 candidate pair ranking 上，和 CZSL evaluation 更对齐。

### 8.3 计算成本问题与 Top-K 策略

直接对每张图像的所有 pair 都算 SD energy 会非常慢。

所以建议先做 top-K reranking：

1. 先用 CAMS 得到所有 candidate pair 的分数。
2. 对每张图像取 CAMS top-K pairs，例如：

   ```text
   K = 20 / 50 / 100
   ```

3. 只对这些 top-K pairs 计算 SD denoising energy。
4. 对 top-K 内部重新排序。
5. 非 top-K pair 保持原 CAMS 分数或不给 SD 加成。

这样计算量可以下降很多，而且更符合实际需求：如果 CAMS 本来把正确 pair 排得很靠后，SD reranker 很难救回来；更现实的目标是纠正 CAMS top candidates 里的组合混淆。

### 8.4 具体实现流程

建议先做一个 evaluation-only 脚本，不训练模型。

#### Step 1：加载 CAMS baseline

先用原始 CAMS seed0 或 CAMS-Diff 里不带 diffusion 的 checkpoint 作为基础。

输出每张图像的：

```text
score_CAMS(image, pair)
```

#### Step 2：预计算图像 latent

对 val/test 图像：

```text
image -> SD VAE -> latent z
```

这个 latent 可以缓存，避免每个 pair 重复编码图像。

#### Step 3：预计算 pair prompt embedding

对所有 candidate pairs 构造 prompt：

```text
a photo of a {attr} {obj}
```

用 SD2.1 的 text encoder 编码，缓存：

```text
pair -> text_condition
```

#### Step 4：计算 top-K pair 的 SD energy

对每张图像：

1. 从 CAMS score 中选 top-K pairs。
2. 采样固定 timestep 和 noise。
3. 对每个 top-K pair 计算 denoising error。
4. 得到：

   ```text
   score_SD = -E_SD
   ```

为减少随机性，可以用固定 seed，并对多个 timestep 取平均：

```text
N_timestep = 1 / 2 / 4
```

先从 `N_timestep=1` 开始，跑通后再增加。

#### Step 5：分数融合

对每张图像的 top-K pairs：

```text
score_final = score_CAMS + alpha * zscore(score_SD)
```

其中 `alpha` 在 val set 上搜索：

```text
alpha = 0.0, 0.05, 0.1, 0.2, 0.5, 1.0
```

也可以同时搜索：

```text
K = 20, 50, 100
N_timestep = 1, 2, 4
```

#### Step 6：用 val 选超参，用 test 汇报

流程：

1. 在 val set 上选 `alpha, K, N_timestep`。
2. 固定超参。
3. 在 test set 上评估 closed/open world。

这样实验更干净，因为 SD 不参与训练，只参与后处理或 reranking。

### 8.5 为什么这个方向比之前更合理

之前 v1-v3 的 SD 信号是训练正则：

```text
SD loss -> CAMS feature
```

它的问题是：优化 SD loss 不等于优化 CZSL ranking。

新方案是推理时 reranking：

```text
SD score(image, pair) -> final pair score
```

它的优点：

1. 目标更对齐：SD 直接影响 pair ranking。
2. 不破坏 CAMS：CAMS baseline 能力完整保留。
3. 更可解释：可以查看某张图像 top-K pair 的 CAMS score、SD energy、final score。
4. 更像 VRDiff：冻结扩散模型作为视觉关系/组合的外部 judge，而不是训练一个 adapter 去迎合它。
5. 更适合写成论文故事：SD 提供 generative compatibility prior，CAMS 提供 discriminative compositional score。

### 8.6 可能的风险

这个方向也不保证一定提升，主要风险有：

1. SD2.1 对 MIT/UT 的细粒度属性可能仍然不敏感。
2. prompt 质量会影响 SD score。
3. denoising energy 的尺度可能不稳定，需要 per-image normalization。
4. 计算成本较高，尤其是 MIT/CGQA pair 数量大。
5. 如果 CAMS top-K 中没有正确 pair，reranking 很难救回来。

因此第一轮实验应该设计成低成本诊断，而不是直接大规模训练。

### 8.7 第一轮推荐实验

我建议先在 MIT 上做，因为 MIT 比 UT 更接近自然图像属性组合，SD 的通用先验可能更有用。

第一轮只做 closed-world evaluation-only：

```text
dataset: MIT-States
base model: CAMS seed0
world: closed
K: 50
N_timestep: 1
alpha sweep: 0, 0.05, 0.1, 0.2, 0.5
prompt: "a photo of a {attr} {object}"
```

判断标准：

1. val AUC 是否能超过 CAMS baseline。
2. test AUC 是否至少不下降。
3. SD score 是否能把正确 pair 在 top-K 中往前推。
4. 对成功和失败样例做可视化分析。

如果第一轮有效，再扩展：

1. 多 timestep average。
2. 多 prompt template ensemble。
3. open-world setting。
4. CGQA。
5. 将 SD score 蒸馏成轻量 pair prior，减少推理成本。

### 8.8 如果 SD Reranking 仍然无效怎么办

如果 SD native prompt reranking 也没有提升，那么说明 SD2.1 的 denoising energy 对当前 CZSL benchmark 的判别性不足。

这时不建议继续在 SD denoising loss 上投入太多，可以转向：

1. CLIP/text semantic prior：用语言模型或 CLIP text embedding 建 pair feasibility。
2. 图像 caption/BLIP prior：先生成图像描述，再辅助判断 attr-object。
3. 组合原型增强：在 CAMS 的文本 prompt 或 pair prototype 上做更强的组合约束。
4. I2CD 式 invertible/disentangled module：不依赖 SD，而是在 CAMS feature space 内做可逆解耦和重组。

## 9. 当前建议

短期最值得做的是：

1. 不再继续调当前 v3 diffusion training loss。
2. 新建一个 evaluation-only 的 SD reranking 分支。
3. 先在 MIT closed-world 上验证：

   ```text
   CAMS score + alpha * SD native prompt denoising score
   ```

4. 如果 val/test 有正向趋势，再考虑写成方法。

如果这个方向有效，论文故事可以表述为：

```text
CAMS learns discriminative compositional representations.
Frozen diffusion model provides generative compatibility feedback at pair-ranking level.
The two scores are fused to improve compositional generalization without retraining the generative model.
```

这个故事比“让 SD 直接解耦 CAMS feature”更稳，也更容易解释当前实验现象。

## 10. SD Native Prompt Top-K Reranking 实验

相关 commit：

```text
282ec8b add SD top-k reranking evaluation
```

主要脚本：

```text
sd_rerank_eval.py
scripts/run_mit_sd_rerank.sh
```

主要输出目录：

```text
output/mit_sd_rerank_k20_t1
```

### 10.1 实验设置

这一版不再训练 CAMS，也不训练 diffusion adapter，而是 evaluation-only。

基础模型：

```text
<CAMS 复现目录>/output/mit/final_model.pt
```

数据集：

```text
MIT-States
Closed-World evaluation
```

SD 模型：

```text
<SD 权重目录>/stable-diffusion-2-1
```

Prompt：

```text
a photo of a {attr} {obj}
```

Reranking 设置：

```text
topK pairs: 20
num_timesteps: 1
alpha_grid: 0, 0.02, 0.05, 0.1, 0.2, 0.5
score_final = score_CAMS + alpha * zscore(-E_SD)
```

其中 `E_SD` 是 frozen SD 在图像 latent 和 pair text prompt 条件下的 denoising error。只对 CAMS top-20 candidate pairs 计算 SD energy。

### 10.2 实验结果

Val set：

| alpha | S | U | HM | AUC | attr_acc | obj_acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 52.28 | 57.87 | 43.53 | 26.48 | 39.14 | 58.13 |
| 0.02 | 52.28 | 57.94 | 43.60 | 26.57 | 39.14 | 58.17 |
| 0.05 | 52.17 | 57.93 | 43.57 | 26.52 | 39.14 | 58.15 |
| 0.10 | 52.17 | 57.91 | 43.66 | 26.47 | 39.16 | 58.21 |
| 0.20 | 51.74 | 58.01 | 43.60 | 26.27 | 39.28 | 58.21 |
| 0.50 | 51.25 | 57.57 | 42.88 | 25.85 | 39.35 | 57.95 |

按 val AUC 选择，best alpha 是：

```text
alpha = 0.02
```

Test set：

| alpha | S | U | HM | AUC | attr_acc | obj_acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 50.25 | 54.12 | 39.90 | 23.11 | 37.87 | 55.84 |
| 0.02 | 50.29 | 54.29 | 39.93 | 23.18 | 37.93 | 55.92 |
| 0.05 | 50.38 | 54.25 | 39.99 | 23.27 | 37.95 | 55.91 |
| 0.10 | 50.25 | 54.35 | 40.03 | 23.25 | 38.05 | 55.94 |
| 0.20 | 50.55 | 54.57 | 40.19 | 23.48 | 38.26 | 56.11 |
| 0.50 | 49.96 | 54.30 | 39.98 | 23.07 | 37.92 | 55.94 |

如果严格按 val AUC 选 alpha：

```text
alpha = 0.02
test AUC: 23.11 -> 23.18
test HM: 39.90 -> 39.93
```

如果看 test oracle 最优 alpha：

```text
alpha = 0.20
test AUC: 23.11 -> 23.48
test HM: 39.90 -> 40.19
```

### 10.3 诊断统计

Top-20 覆盖率：

| split | GT pair in CAMS top-20 | mean GT rank | median GT rank |
|---|---:|---:|---:|
| val | 85.53% | 22.33 | 3 |
| test | 84.16% | 23.81 | 3 |

这说明 top-20 reranking 有可操作空间：大约 84%-86% 的样本真实 pair 已经在 CAMS top-20 中，SD reranker 有机会在候选集合内部调整顺序。

Top-1 和 rank 变化：

| split | alpha | top1 | mean GT rank | improved | worsened |
|---|---:|---:|---:|---:|---:|
| val | 0.00 | 26.68 | 22.33 | 0.00 | 0.00 |
| val | 0.02 | 26.70 | 22.32 | 2.28 | 1.71 |
| val | 0.10 | 26.79 | 22.31 | 8.21 | 6.82 |
| val | 0.20 | 26.91 | 22.30 | 12.37 | 10.42 |
| val | 0.50 | 27.32 | 22.37 | 18.18 | 17.25 |
| test | 0.00 | 25.46 | 23.81 | 0.00 | 0.00 |
| test | 0.02 | 25.54 | 23.80 | 2.39 | 1.68 |
| test | 0.10 | 25.71 | 23.78 | 8.64 | 6.17 |
| test | 0.20 | 25.95 | 23.76 | 12.84 | 10.13 |
| test | 0.50 | 25.89 | 23.80 | 18.55 | 16.93 |

这里的 `improved/worsened` 表示在全体样本中，真实 pair 在 top-20 内被 reranker 提升或降低排名的比例。

### 10.4 结果分析

这次结果和前面的 training-time diffusion 不一样：SD native prompt reranking 出现了稳定的轻微正向信号。

主要观察：

1. `alpha=0.02` 在 val AUC 上最好，test 也有轻微提升。
2. `alpha=0.2` 在 test 上提升更明显，但它不是 val AUC 选择出的最优 alpha。
3. 过大的 `alpha=0.5` 会使 val AUC 和 test AUC 下降，说明 SD score 不能单独强势主导，只适合作为轻量 prior。
4. SD reranking 对 unseen 更友好一些，例如 test unseen 从 `54.12` 到 `54.57`，seen 从 `50.25` 到 `50.55`。
5. attr/object accuracy 也有小幅提升，说明 reranker 不只是改变 seen/unseen calibration，也确实调整了一部分 pair 排名。

和 v1-v3 training-time diffusion 相比，这个方向更有希望：

| 方法 | Test AUC |
|---|---:|
| CAMS seed0 | 23.11 |
| training-time v3 | 22.30 |
| lambda ablation best | 22.38 |
| SD rerank, val-selected alpha=0.02 | 23.18 |
| SD rerank, test oracle alpha=0.20 | 23.48 |

这说明 SD 的原生文本条件确实含有一点 pair compatibility 信息；之前直接训练 adapter 失败，主要是因为目标没有进入最终 ranking，且 adapter 可能学了与 CZSL 不对齐的 shortcut。

不过，这次提升仍然偏小，还不能直接作为强结论。

严格按 val AUC 选 alpha 时：

```text
AUC +0.07
HM +0.03
```

这只是一个弱正向趋势。

如果使用 test oracle alpha=0.20：

```text
AUC +0.37
HM +0.29
```

说明潜力存在，但当前 val selection 和 test 最优不完全一致，需要进一步稳定。

### 10.5 下一步建议

接下来不建议回到 training-time diffusion loss，而应该继续沿着 reranking 方向做更细的验证。

优先级最高的实验：

1. 复用已有 cache，做更细的 alpha sweep：

   ```text
   alpha = 0.00, 0.02, 0.04, ..., 0.30
   ```

   这不需要重新跑 SD，只需要重新评估融合分数。

2. 扩大 topK：

   ```text
   K = 50
   ```

   当前 top-20 覆盖率已经有 84%-86%，K=50 可能给 SD 更多纠错空间，但计算成本会变高。

3. 多 timestep 平均：

   ```text
   num_timesteps = 2 或 4
   ```

   当前只采样 1 个 timestep，SD energy 随机性比较强。多 timestep 可能让 val/test alpha 选择更稳定。

4. Prompt ensemble：

   ```text
   a photo of a {attr} {obj}
   a photo of the {obj} is {attr}
   a {attr} {obj}
   ```

   MIT 的属性有很多状态词，单一 prompt 可能不是最优。

5. 样例分析：

   取 alpha=0.2 时被提升/被伤害的样本，看 SD 是在修正哪些组合，还是只改变同一 object 下的属性排序。

当前最合理的下一步是先做第 1 点，因为已有 cache，不需要重新跑 SD，成本最低。如果 fine-grained alpha 显示 val/test 都在某个小区间稳定提升，再跑 K=50 或多 timestep。

## 11. K=20/t=1 Fine Alpha Sweep

相关 commit：

```text
9f6a12e add cached rerank alpha sweep
```

主要脚本：

```text
rerank_alpha_sweep.py
```

主要结果文件：

```text
output/mit_sd_rerank_k20_t1/alpha_sweep_k20_t1.json
output/mit_sd_rerank_k20_t1/alpha_sweep_k20_t1.csv
output/mit_sd_rerank_k20_t1/alpha_sweep_k20_t1.log
```

这一步复用已有 cache，不重新计算 SD energy，只重新融合分数并评估。

### 11.1 设置

```text
dataset: MIT-States
world: closed
topK: 20
num_timesteps: 1
alpha: 0.00, 0.02, 0.04, ..., 0.30
select_metric: val AUC
```

### 11.2 关键结果

Val AUC 最优：

```text
alpha = 0.02
val AUC = 26.575
test AUC = 23.183
test HM  = 39.935
```

Test oracle AUC 最优：

```text
alpha = 0.28
test AUC = 23.530
test HM  = 40.199
```

相比 CAMS baseline：

| 设置 | alpha | Test S | Test U | Test HM | Test AUC |
|---|---:|---:|---:|---:|---:|
| CAMS baseline | 0.00 | 50.25 | 54.12 | 39.90 | 23.11 |
| Val-selected | 0.02 | 50.29 | 54.29 | 39.93 | 23.18 |
| Test oracle | 0.28 | 50.67 | 54.58 | 40.20 | 23.53 |

更细的 test 趋势：

| alpha | Test AUC | Test HM | Test U | Test S |
|---:|---:|---:|---:|---:|
| 0.00 | 23.113 | 39.895 | 54.121 | 50.252 |
| 0.02 | 23.183 | 39.935 | 54.291 | 50.294 |
| 0.06 | 23.239 | 39.984 | 54.301 | 50.378 |
| 0.10 | 23.249 | 40.033 | 54.348 | 50.252 |
| 0.12 | 23.351 | 40.106 | 54.395 | 50.462 |
| 0.16 | 23.433 | 40.268 | 54.508 | 50.546 |
| 0.18 | 23.526 | 40.170 | 54.621 | 50.630 |
| 0.20 | 23.483 | 40.190 | 54.574 | 50.546 |
| 0.24 | 23.483 | 40.112 | 54.574 | 50.504 |
| 0.28 | 23.530 | 40.199 | 54.583 | 50.672 |
| 0.30 | 23.487 | 40.199 | 54.517 | 50.630 |

### 11.3 诊断结果

K=20 的真实 pair 覆盖率保持不变：

| split | GT pair in top-20 | median GT rank |
|---|---:|---:|
| val | 85.53% | 3 |
| test | 84.16% | 3 |

随着 alpha 增大，SD reranker 对 top-20 内真实 pair 的改动更多：

| split | alpha | improved_all | worsened_all | top1 |
|---|---:|---:|---:|---:|
| val | 0.02 | 2.28 | 1.71 | 26.70 |
| val | 0.10 | 8.21 | 6.82 | 26.79 |
| val | 0.20 | 12.37 | 10.42 | 26.91 |
| val | 0.30 | 14.94 | 13.29 | 27.13 |
| test | 0.02 | 2.39 | 1.68 | 25.54 |
| test | 0.10 | 8.64 | 6.17 | 25.71 |
| test | 0.20 | 12.84 | 10.13 | 25.95 |
| test | 0.28 | 14.77 | 12.20 | 25.98 |

### 11.4 分析

fine alpha sweep 进一步确认了两个现象。

第一，SD reranking 的确有正向信号。

Test AUC 从 `23.11` 稳定上升到 `23.35~23.53` 区间，尤其 `alpha=0.16~0.30` 整体都高于 baseline。这说明 SD native prompt energy 不是纯噪声，它在 top-20 candidate 内提供了一定组合兼容性信息。

第二，val-selected alpha 仍然偏保守。

按 val AUC 选择得到 `alpha=0.02`，test 只提升到 `23.18`；但 test oracle 最优在 `alpha=0.28`，AUC 到 `23.53`。这说明当前 val AUC 对 alpha 的选择不够稳定，可能因为：

1. val/test 的 pair 分布不完全一致。
2. AUC 对 seen/unseen bias 曲线比较敏感，小幅排序变化会让选择偏保守。
3. 单 timestep 的 SD energy 随机性较强。
4. K=20 限制了 reranker 的作用空间，val 上较大 alpha 可能更容易伤害局部排序。

因此，下一步最值得验证的是：

```text
K=50, num_timesteps=1
```

如果 K=50 后 val-selected alpha 也移动到更大的区间，且 test 继续提升，就说明扩大候选集可以让 SD prior 更稳定地发挥作用。

## 12. 第八版：K=50 的 SD Top-K Reranking

### 12.1 改动

第八版不是新的训练版本，而是继续沿用 frozen SD native prompt reranking，只把候选数从 K=20 扩大到 K=50：

```text
score_final = score_CAMS + alpha * zscore(-E_SD(image, attr, obj))
```

主要设置：

| 设置项 | 值 |
|---|---|
| dataset | MIT-States |
| setting | closed-world |
| CAMS checkpoint | CAMS seed0 final_model.pt |
| SD model | stable-diffusion-2-1 |
| top-k pairs | 50 |
| num_timesteps | 1 |
| prompt | a photo of a {attr} {obj} |
| output dir | output/mit_sd_rerank_k50_t1 |

这一版的目的不是继续证明训练式 diffusion loss，而是验证一个更直接的问题：

```text
扩大 CAMS 候选集合以后，SD prior 是否能更有效地修正 pair 排序？
```

### 12.2 实验结果

K=50 的 val 最优 alpha：

```text
best val alpha by AUC = 0.04
```

对应 test 结果：

| 设置 | alpha | Test S | Test U | Test HM | Test AUC |
|---|---:|---:|---:|---:|---:|
| CAMS baseline | 0.00 | 50.25 | 54.12 | 39.90 | 23.11 |
| K50 val-selected | 0.04 | 50.34 | 54.28 | 39.93 | 23.20 |
| K50 test oracle | 0.30 | 50.59 | 54.55 | 40.27 | 23.50 |

K50 的 test alpha sweep：

| alpha | Test S | Test U | Test HM | Test AUC |
|---:|---:|---:|---:|---:|
| 0.00 | 50.25 | 54.12 | 39.90 | 23.11 |
| 0.02 | 50.34 | 54.26 | 39.91 | 23.17 |
| 0.04 | 50.34 | 54.28 | 39.93 | 23.20 |
| 0.06 | 50.38 | 54.29 | 39.98 | 23.21 |
| 0.08 | 50.34 | 54.27 | 40.00 | 23.20 |
| 0.10 | 50.34 | 54.31 | 40.03 | 23.22 |
| 0.12 | 50.29 | 54.36 | 40.06 | 23.26 |
| 0.14 | 50.50 | 54.37 | 40.13 | 23.37 |
| 0.16 | 50.46 | 54.48 | 40.15 | 23.35 |
| 0.18 | 50.55 | 54.52 | 40.15 | 23.45 |
| 0.20 | 50.42 | 54.55 | 40.09 | 23.41 |
| 0.22 | 50.29 | 54.61 | 40.05 | 23.39 |
| 0.24 | 50.38 | 54.65 | 40.14 | 23.40 |
| 0.26 | 50.38 | 54.59 | 40.09 | 23.41 |
| 0.28 | 50.46 | 54.55 | 40.17 | 23.45 |
| 0.30 | 50.59 | 54.55 | 40.27 | 23.50 |

### 12.3 与 K=20 对比

| 设置 | val-selected alpha | Test AUC at val alpha | Test oracle alpha | Test oracle AUC |
|---|---:|---:|---:|---:|
| K20, t=1 | 0.02 | 23.18 | 0.20/0.28 | 23.48/23.53 |
| K50, t=1 | 0.04 | 23.20 | 0.30 | 23.50 |

K50 相比 K20 的收益很小：

```text
val-selected AUC: 23.18 -> 23.20
test-oracle AUC: 23.53 -> 23.50
```

虽然 K50 提高了真实 pair 落在候选集合里的覆盖率，但最终 test AUC 没有明显提升。这说明当前瓶颈不主要是候选数量，而更可能是 SD energy 的估计噪声、prompt 表达能力，以及 val alpha 选择不稳定。

### 12.4 当前结论

到目前为止，整体结论可以概括为：

1. 训练阶段把 frozen SD energy/ranking loss 直接接进 CAMS，不稳定，且没有带来可靠提升。
2. 推理阶段把 frozen SD 当作 native prompt reranker，有稳定但较小的正向信号。
3. SD reranker 在 test oracle alpha 下可以把 MIT closed AUC 从 `23.11` 提升到约 `23.50`。
4. val-selected alpha 只能提升到 `23.18~23.20`，说明选择权重的可靠性仍是问题。
5. 继续单纯增大 K 不太值得，K50 相比 K20 没有明显额外收益。

### 12.5 下一步建议

下一步最值得做的是提高 SD energy 的稳定性，而不是继续扩大候选集。

优先级最高的实验：

```text
K=20, num_timesteps=4
```

原因：

1. K20 已经覆盖了大部分真实 pair，继续加到 K50 收益很弱。
2. 单 timestep 的 diffusion energy 方差较大，可能导致 val/test alpha 不一致。
3. 多 timestep averaging 可以在不改变 CAMS 主模型的情况下，让 SD score 更稳。
4. 如果 K20/t=4 后 val-selected alpha 向 `0.10~0.30` 靠近，说明 SD prior 的可用性更强。

建议后续验证顺序：

| 优先级 | 实验 | 目的 |
|---:|---|---|
| 1 | K20, timesteps=4 | 降低 SD energy 噪声 |
| 2 | K20, timesteps=8 | 看稳定性是否继续提升 |
| 3 | prompt ensemble | 降低单 prompt 偏差 |
| 4 | val metric selection 对比 | 比较用 AUC、HM 或 combined metric 选 alpha |
| 5 | MIT 多 seed | 确认提升是否稳健 |

暂时不建议继续做：

1. 继续扩大到 K100。
2. 继续调训练式 diffusion loss 的 lambda。
3. 在 UT 上反复调 diffusion reranking。

当前更合理的主线是：

```text
CAMS 提供可靠候选集合
Frozen SD native prompt energy 提供组合兼容性先验
通过多 timestep / prompt ensemble 提高这个先验的稳定性
最后再考虑是否把稳定后的 SD score 蒸馏回 CAMS
```

## 13. 第九版：Multi-Timestep SD Reranking

### 13.1 改动

这一版验证的是上一节提出的核心问题：

```text
单 timestep 的 SD energy 是否太噪，multi-timestep averaging 能否让 rerank 更稳定？
```

具体做了两个并行实验：

| 数据集 | 设置 | 输出目录 |
|---|---|---|
| MIT-States | K=20, num_timesteps=4 | output/mit_sd_rerank_k20_t4 |
| UT-Zappos | K=20, num_timesteps=4 | output/ut_sd_rerank_k20_t4 |

因为 `num_timesteps=4` 会显著增加显存和计算量，脚本中把 batch 改小：

```text
eval_batch_size = 8
sd_batch_size = 4
```

对应代码版本：

```text
29af210 reduce-K20-t4-rerank-batches
6f62dc0 add-UT-K20-timestep-4-rerank-runner
```

### 13.2 MIT-States 结果

MIT K20/t4 的结果：

| 设置 | alpha | Test S | Test U | Test HM | Test AUC |
|---|---:|---:|---:|---:|---:|
| CAMS baseline | 0.00 | 50.25 | 54.12 | 39.90 | 23.11 |
| K20/t1 val-selected | 0.02 | 50.29 | 54.29 | 39.93 | 23.18 |
| K20/t4 val-selected | 0.30 | 50.21 | 54.68 | 40.09 | 23.31 |
| K20/t4 test oracle | 0.28 | 50.38 | 54.63 | 40.13 | 23.41 |

MIT K20/t4 的 val 最优 alpha 从 t1 的 `0.02` 移到了 `0.30`：

```text
t1 best val alpha = 0.02
t4 best val alpha = 0.30
```

这说明 multi-timestep averaging 确实改变了 val 上对 SD score 的信任程度，验证了前面的判断：单 timestep 的 SD energy 偏噪，t=4 后 val 不再极端保守。

但是 test 上的最终提升仍然有限：

```text
baseline AUC       = 23.11
K20/t4 val AUC     = 23.31
K20/t4 oracle AUC  = 23.41
K20/t1 oracle AUC  = 23.48
K50/t1 oracle AUC  = 23.50
```

因此，t=4 的主要收益不是把 oracle 上限继续抬高，而是让 val-selected alpha 更接近一个有用区间。

### 13.3 UT-Zappos 结果

UT K20/t4 的结果：

| 设置 | alpha | Test S | Test U | Test HM | Test AUC |
|---|---:|---:|---:|---:|---:|
| CAMS baseline | 0.00 | 68.62 | 76.10 | 56.56 | 43.76 |
| K20/t4 val-selected | 0.26 | 68.43 | 75.20 | 56.31 | 43.22 |
| K20/t4 test oracle by AUC | 0.00 | 68.62 | 76.10 | 56.56 | 43.76 |
| K20/t4 test oracle by HM | 0.12 | 68.72 | 75.52 | 56.70 | 43.56 |

UT 的结论比较明确：

```text
按 AUC 看，SD reranking 没有带来提升。
```

val 上最优 alpha 是 `0.26`，但这个 alpha 到 test 上反而使 AUC 从 `43.76` 降到 `43.22`。test AUC 最优仍然是 `alpha=0.00`，也就是不用 SD reranking。

有一个小现象是 HM 在 `alpha=0.12` 时从 `56.56` 到 `56.70`，但 AUC 降到 `43.56`，这个收益太小，不能算可靠提升。

### 13.4 分析

这一轮把 MIT 和 UT 的差异暴露得比较清楚。

MIT 上：

1. SD prior 有正信号。
2. t=4 让 val-selected alpha 从 `0.02` 移到 `0.30`。
3. val-selected test AUC 从 `23.18` 提升到 `23.31`。
4. 但 oracle 上限没有超过 K20/t1 和 K50/t1。

UT 上：

1. SD prior 对 AUC 基本是负作用。
2. val 上看似可用的 alpha 泛化到 test 变差。
3. SD 更倾向改善 object accuracy 或局部 HM，但损害 unseen/AUC 平衡。
4. 这符合 UT-Zappos 是细粒度鞋子数据集的直觉：SD2.1 的通用图文先验很难稳定区分这类材质、款式、颜色组合。

### 13.5 当前判断

到目前为止，SD reranking 方向的定位应该调整为：

```text
它不是一个普适提升模块。
它在 MIT-States 这类自然属性-物体组合上有弱正信号。
它在 UT-Zappos 这种细粒度鞋类组合上不稳定，甚至会拉低 AUC。
```

因此，下一步不建议继续在 UT 上调 SD reranking。

更合理的方向是：

1. 在 MIT 上继续做 prompt ensemble，看看能不能把 `23.31~23.50` 的弱提升变得更稳定。
2. 比较不同 alpha 选择策略，尤其是 val AUC、val HM、combined metric。
3. 如果 prompt ensemble 仍然只能带来很小提升，就把 SD reranking 定位为分析性辅助模块，而不是主贡献。
4. 若要形成论文式方法，需要考虑把 SD prior 变成可学习但轻量的 pair compatibility prior，而不是直接用 SD energy 做最终 rerank。

## 14. 第十版：Prompt Ensemble SD Reranking

### 14.1 验证动机

前面的实验显示：

1. MIT 上 frozen SD reranking 有弱正信号。
2. multi-timestep averaging 能让 val-selected alpha 从 `0.02` 移到更合理的区间。
3. 但 MIT 的提升仍然较小，UT 上甚至会下降。

因此，下一步验证的问题是：

```text
单一 prompt 是否限制了 SD energy 的稳定性？
```

之前只使用：

```text
a photo of a {attr} {obj}
```

这一版改成 prompt ensemble，对多个文本表达下的 SD denoising energy 求平均。

### 14.2 代码改动

新增参数：

```text
--prompt_templates
```

多个模板用 `||` 分隔。如果传入该参数，就覆盖原来的单模板 `--prompt_template`。

当前使用 4 个 prompt：

```text
a photo of a {attr} {obj}
a photo of the {obj} is {attr}
a {attr} {obj}
an image of a {attr} {obj}
```

energy 计算方式从：

```text
E_SD(image, pair)
```

变为：

```text
E_SD(image, pair) = mean_p E_SD(image, prompt_p(attr, obj))
```

对应代码版本：

```text
427ead0 add-prompt-ensemble-reranking
```

### 14.3 当前运行实验

先在 MIT 上验证 prompt ensemble，暂时不和 t=4 叠加，避免一次改变太多因素：

| 设置项 | 值 |
|---|---|
| dataset | MIT-States |
| setting | closed-world |
| top-k pairs | 20 |
| num_timesteps | 1 |
| prompt count | 4 |
| output dir | output/mit_sd_rerank_k20_t1_prompt4 |
| tmux session | mit_k20_t1_prompt4 |

这轮主要和两个结果对比：

| 设置 | val-selected AUC | test oracle AUC |
|---|---:|---:|
| K20/t1 single prompt | 23.18 | 23.48 |
| K20/t4 single prompt | 23.31 | 23.41 |
| K20/t1 prompt4 | pending | pending |

判断标准：

1. 如果 prompt4 的 val-selected AUC 明显高于 `23.18`，说明 prompt ensemble 能降低单 prompt 偏差。
2. 如果 prompt4 的 test oracle AUC 超过 `23.48`，说明多 prompt 能提高 SD prior 上限。
3. 如果 prompt4 只改变 alpha 但 AUC 不升，说明当前瓶颈不主要是 prompt 表达。
4. 如果 prompt4 下降，说明更多 prompt 可能引入噪声，需要考虑 prompt 筛选而不是简单平均。

### 14.4 实验结果

MIT K20/t1 prompt4 已完成：

| 设置 | alpha | Test S | Test U | Test HM | Test AUC |
|---|---:|---:|---:|---:|---:|
| CAMS baseline | 0.00 | 50.25 | 54.12 | 39.90 | 23.11 |
| K20/t1 single prompt val-selected | 0.02 | 50.29 | 54.29 | 39.93 | 23.18 |
| K20/t1 single prompt test oracle | 0.20 | 50.55 | 54.57 | 40.19 | 23.48 |
| K20/t1 prompt4 val-selected | 0.04 | 50.29 | 54.30 | 39.81 | 23.16 |
| K20/t1 prompt4 test oracle | 0.28 | 50.84 | 54.42 | 40.26 | 23.47 |

prompt4 的 val 最优：

```text
best val alpha = 0.04
val AUC        = 26.54
test AUC       = 23.16
```

prompt4 的 test oracle：

```text
best test alpha = 0.28
test AUC        = 23.47
test HM         = 40.26
```

### 14.5 分析

prompt ensemble 的结果不理想。

和单 prompt 相比：

```text
val-selected AUC: 23.18 -> 23.16
test-oracle AUC: 23.48 -> 23.47
```

也就是说，简单平均 4 个 prompt 的 SD energy 没有提高上限，也没有改善 val-selected 结果。它的 oracle HM 从 `40.19` 到 `40.26` 有微小提升，但 AUC 没有提升，因此不构成可靠收益。

这说明：

1. prompt 多样性本身不是充分条件。
2. 简单平均可能把有用 prompt 和有噪声 prompt 混在一起。
3. 下一步应该做 prompt selection，而不是继续增加 prompt 数量。

## 15. 第十一版：Individual Prompt Selection

### 15.1 验证动机

prompt4 平均没有提升，但这并不能说明每个 prompt 都没用。可能存在这种情况：

```text
某个 prompt 单独有效，但和其他 prompt 平均后被稀释。
```

所以第十一版改为分别验证单个 prompt。

已验证过的默认 prompt：

```text
a photo of a {attr} {obj}
```

接下来验证另外 3 个 prompt：

| prompt name | template |
|---|---|
| obj_is_attr | a photo of the {obj} is {attr} |
| plain | a {attr} {obj} |
| image | an image of a {attr} {obj} |

### 15.2 实验设置

保持其他设置不变，只替换 prompt：

| 设置项 | 值 |
|---|---|
| dataset | MIT-States |
| setting | closed-world |
| top-k pairs | 20 |
| num_timesteps | 1 |
| eval_batch_size | 8 |
| sd_batch_size | 4 |

输出目录：

```text
output/mit_sd_rerank_k20_t1_prompt_obj_is_attr
output/mit_sd_rerank_k20_t1_prompt_plain
output/mit_sd_rerank_k20_t1_prompt_image
```

判断标准：

1. 如果某个单 prompt 的 oracle AUC 高于 `23.48`，说明 prompt 表达确实影响 SD prior 上限。
2. 如果某个单 prompt 的 val-selected AUC 高于 `23.31`，说明它比 t=4 更有实用价值。
3. 如果所有单 prompt 都接近或低于默认 prompt，说明当前瓶颈不是 prompt 文本，而是 SD energy 本身与 CZSL 排序目标不够对齐。

### 15.3 实验结果

三个单 prompt 变体均已完成：

| prompt | val-selected alpha | Test AUC at val alpha | Test HM at val alpha | Test oracle alpha | Test oracle AUC | Test oracle HM |
|---|---:|---:|---:|---:|---:|---:|
| default: a photo of a {attr} {obj} | 0.02 | 23.18 | 39.93 | 0.20 | 23.48 | 40.19 |
| obj_is_attr: a photo of the {obj} is {attr} | 0.06 | 23.23 | 39.82 | 0.28 | 23.52 | 40.25 |
| plain: a {attr} {obj} | 0.04 | 23.19 | 39.84 | 0.18 | 23.25 | 39.91 |
| image: an image of a {attr} {obj} | 0.02 | 23.17 | 39.90 | 0.18 | 23.30 | 40.08 |
| prompt4 average | 0.04 | 23.16 | 39.81 | 0.28 | 23.47 | 40.26 |

最好的单 prompt 是：

```text
a photo of the {obj} is {attr}
```

它的 test oracle AUC 达到：

```text
23.52
```

这是目前所有 K20/t1 prompt 实验里最高的 AUC，略高于默认 prompt 的 `23.48`，也略高于 K50/t1 的 `23.50`。

但是它的 val-selected 结果仍然不够强：

```text
best val alpha = 0.06
test AUC       = 23.23
```

### 15.4 分析

prompt selection 的结论比 prompt ensemble 更有价值。

第一，prompt 确实会影响 SD prior 的上限。`obj_is_attr` 的 oracle AUC 从默认 prompt 的 `23.48` 提高到 `23.52`，说明文本表达不是完全无关。

第二，简单 prompt averaging 不合适。prompt4 平均没有超过最好的单 prompt，说明不同 prompt 的 energy 不是等价噪声，直接平均会混入弱 prompt。

第三，val-selected alpha 仍然偏小。`obj_is_attr` 的 oracle alpha 是 `0.28`，但 val AUC 选择的是 `0.06`。这个现象和默认 prompt 的 t=1 结果一致，说明单 timestep 仍然会让 val 选择偏保守。

因此，下一步最自然的是把两个有效因素组合起来：

```text
obj_is_attr prompt + num_timesteps=4
```

## 16. 第十二版：obj_is_attr Prompt + Multi-Timestep

### 16.1 验证动机

目前分别验证出的两个有效因素是：

| 因素 | 观察 |
|---|---|
| num_timesteps=4 | 默认 prompt 下让 val-selected alpha 从 `0.02` 到 `0.30`，test AUC 从 `23.18` 到 `23.31` |
| obj_is_attr prompt | 单 timestep 下 oracle AUC 从默认 prompt `23.48` 到 `23.52` |

因此第十二版验证：

```text
更好的 prompt + 更稳定的 timestep averaging 是否能叠加？
```

### 16.2 实验设置

| 设置项 | 值 |
|---|---|
| dataset | MIT-States |
| setting | closed-world |
| top-k pairs | 20 |
| num_timesteps | 4 |
| prompt | a photo of the {obj} is {attr} |
| eval_batch_size | 8 |
| sd_batch_size | 4 |
| output dir | output/mit_sd_rerank_k20_t4_prompt_obj_is_attr |

对比目标：

| 设置 | val-selected AUC | oracle AUC |
|---|---:|---:|
| default prompt, t=1 | 23.18 | 23.48 |
| default prompt, t=4 | 23.31 | 23.41 |
| obj_is_attr, t=1 | 23.23 | 23.52 |
| obj_is_attr, t=4 | pending | pending |

如果这一版有效，理想情况是：

```text
val-selected AUC > 23.31
oracle AUC       >= 23.52
```

如果只提升 val-selected、不提升 oracle，说明 t=4 主要改善 alpha 选择。

如果二者都不提升，说明 prompt/timestep 两个因素不能简单叠加，SD prior 的有效上限可能已经接近当前水平。

### 16.3 实验结果

`obj_is_attr + t4` 已完成，输出目录：

```text
output/mit_sd_rerank_k20_t4_prompt_obj_is_attr
```

关键结果：

| 设置 | val-selected alpha | Test AUC at val alpha | Test HM at val alpha | Test oracle alpha | Test oracle AUC | Test oracle HM |
|---|---:|---:|---:|---:|---:|---:|
| default prompt, t=1 | 0.02 | 23.18 | 39.93 | 0.20 | 23.48 | 40.19 |
| default prompt, t=4 | 0.30 | 23.31 | 40.09 | 0.28 | 23.41 | 40.13 |
| obj_is_attr, t=1 | 0.06 | 23.23 | 39.82 | 0.28 | 23.52 | 40.25 |
| obj_is_attr, t=4 | 0.18 | 23.27 | 39.97 | 0.26/0.28 | 23.29 | 39.94 |

`obj_is_attr + t4` 的 val 最优：

```text
best val alpha = 0.18
test AUC       = 23.27
test HM        = 39.97
```

test oracle：

```text
best test AUC = 23.29
best test HM  = 40.00
```

### 16.4 分析

这一版没有达到预期。

原本希望 `obj_is_attr prompt` 的更高 oracle 上限和 `t=4` 的更稳定 alpha 选择能够叠加，但实际结果是：

```text
obj_is_attr, t=1 oracle AUC = 23.52
obj_is_attr, t=4 oracle AUC = 23.29
```

也就是说，t=4 没有保留 obj_is_attr prompt 在 t=1 下的上限，反而把 test oracle 压低了。

不过它确实改变了 val alpha：

```text
obj_is_attr, t=1 best val alpha = 0.06
obj_is_attr, t=4 best val alpha = 0.18
```

这说明 multi-timestep averaging 仍然会让 val 更信任 SD score，但这个更稳定的 score 未必更符合 test 排序目标。

当前判断：

1. prompt 会影响 SD prior，但收益非常小。
2. multi-timestep 可以改变 alpha 选择，但不一定提高 oracle 上限。
3. prompt selection 和 multi-timestep 不能简单叠加。
4. 当前 frozen SD reranking 在 MIT 上的可靠提升大约只有 `+0.1~0.2 AUC`，oracle 上限约 `+0.4 AUC`。

因此，继续在 SD energy 本身上堆 prompt/timestep 的性价比已经变低。

下一步更合理的方向不是继续加 prompt 或 timestep，而是验证：

```text
能不能把 SD score 作为弱先验，用更稳定的方式和 CAMS/CLIP score 融合或蒸馏。
```
