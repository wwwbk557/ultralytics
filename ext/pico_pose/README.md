# Compact 四关键点 Pose 头（Scheme C）

`ext.pico_pose` 提供 **Scheme C 14 类**装甲板四关键点 Pose 头，挂在 YOLO11-pose 的 P3/P4/P5 颈部。

```text
14 类 = 2 色 × 7 role
0 blue_sentry   1 blue_hero   2 blue_engineer   3 blue_3
4 blue_4        5 blue_outpost 6 blue_base
7 red_sentry    8 red_hero    9 red_engineer   10 red_3
11 red_4       12 red_outpost 13 red_base
class = color * 7 + role
```

## 结构

每个 FPN 层：

| 分支 | 结构 | 用途 |
|---|---|---|
| `cv2` | Conv3x3(64) → Conv3x3(64) → Conv1x1(4×reg_max) | DFL 框分布，训练/验证 |
| `cv_shared` | DWConv3x3 → Conv1x1(32) | 共享特征 |
| `cv_compact` | Conv1x1(32→18) | 8 关键点 + 1 quality + 2 colour + 7 role |
| `cv_joint` | Conv3x3(32→64) → Conv3x3(64→64) → Conv1x1(64→14) | TAL 分类 logits，仅训练 |

`cv3`/`cv4` 不使用，P3 不加额外卷积块。导出只执行 `cv_shared + cv_compact`。

> `cv_joint` 结构已从 1×1 换成 3×3 两层，**旧 checkpoint 不兼容**，直接从头训练。

## 训练（480×640）

```bash
python -m ext.pico_pose.train_compact \
  --data /path/to/data_schemeC.yaml \
  --name schemeC_480x640 \
  --epochs 250 --batch 32 --imgsz 480,640 --device 0
```

- `--imgsz 480,640` 会走 `PicoRectTrainer`，实际 batch 是 `(B,3,480,640)`；
- `check_imgsz` 不再把 480,640 强制成 640；
- train/val 都固定 480×640；
- 因为上游 Mosaic 只支持方形，矩形训练改用 `RectMosaic`（2H×2W 画布，affine 直接裁到 H×W），`mosaic/mixup/cutmix` 仍然可用；`copy_paste` 需要分割标签，姿态数据集上是 no-op；
- 不加 `--weights` 就是从头训练。

## 损失槽位

```text
[box, pose, kobj, cls, dfl, color, role, quality, geom]
```

CSV 里的 `number_loss` 是 role 分类损失（历史列名，导出通道 11:18 也叫 number）。

第 8 个槽位 `quality` 现在训练成 **objectness**：TAL 匹配到的正样本目标为 1，
其余 anchor 为 0（以前是 OKS 回归目标）。`colour`/`role` 只在正样本上做交叉熵，
用来判断种类；`quality` 负责“是不是目标”，并以**软分数**参与排序。

objectness 的 BCE 用 `pos_weight = (n_neg / n_pos) ** obj_pw_pow`（mean reduction）：

- `obj_pw_pow = 1.0` 是类别平衡，每个正样本等价于 `n_neg/n_pos` 个负样本；
  均衡后，真实后验只要大于 `1/pos_weight` 的 anchor 都会被推到 0.5 以上，
  靠近目标的背景也容易漏成 FP；
- 默认 `obj_pw_pow = 0.5`（开方），正样本仍然很高，但负样本在总量上占优，
  背景压得更低，适合把 objectness 当门控用；
- 想更强背景抑制调到 0.25，想找回召调到 0.75。

## 验证即部署（deploy parity，默认开）

`deploy_boxes=True`（`PicoRectTrainer` 从训练参数里推到 head 上）让 **val/NMS 用 4 个预测角点
合成框**，而不是训练用的 `cv2`+DFL 分支——导出图里本来就没有 box。同一个 checkpoint、同一
768 张 val 子集，两种框的差别：

| NMS 框 | mAP50(P) | mAP50-95(P) | P | R |
|---|---|---|---|---|
| DFL box 分支 | 0.8522 | 0.8129 | 0.723 | 0.818 |
| 4 角点合成 | **0.9085** | **0.8604** | 0.871 | 0.838 |

关掉用 `--no-deploy-boxes`。注意这会让 `results.csv` 的 mAP 口径变化，跨版本比较时先对齐。

## 目标质量与 objectness 校准（可选）

- `--quality-soft 0.3~0.5`：objectness 正样本目标从硬 1.0 改成
  `(1-s) + s * IoU(角点框, GT 框)`。负样本仍是 0，所以门控语义不变，但正样本之间按
  **部署 NMS 实际使用的 IoU** 排序（默认 0 = 旧行为）。
- `--obj-logit-bias auto`：训练时 `pos_weight ≈ 26`，`sigmoid(logit)` 不是概率
  （0.5 只对应真实前景 3.7%）。auto 用训练集先验算出 `log(pos_weight)` 并写进 head；
  验证时算 `sigmoid(z - bias)`，导出时**直接减在通道 8 的 logit 上**，端侧 sigmoid 不变。
- 导出：`python scripts/export_pico.py --weights best.pt --format onnx`，会在
  `deepcopy(state_dict())` 上把 bias 加进 `cv_compact.bias[8]`，导出完恢复，ckpt 不动。
- **一次性校准护栏**：校准是 logit 空间的可加偏移，同一个 z 只能校准一次。head 上的
  `_obj_logit_calibrated` 在权重烘好后置位，`obj_prob()` / 导出分支 / `PicoCompactPoseLoss`
  三处都会断言它未置位——否则就是双重校准（症状：mAP 掉、loss 正常、阈值全偏）。

## 导出

`head.export = True` 时只执行 `cv_compact`，输出 `[B, A, 18]`：

```text
0:8    raw keypoint offsets      p = (raw + anchor) * stride
8      objectness logit          sigmoid   (旧名 quality)
9:11   colour logits             softmax
11:18  role logits               softmax
obj    = sigmoid(objectness)                        # 二分类前景/背景
score  = obj * softmax(colour)[c] * softmax(role)[r]   # 排序 / mAP 都用它
class  = argmax(colour) * 7 + argmax(role)
```

- **排序和 mAP 必须用软分数** `obj * class_prob`：如果写成
  `class_prob * (obj >= 0.5)`，obj=0.51 和 obj=0.99 的 anchor 分数一样，
  objectness 的排序信息就丢了（NMS/mAP 只剩 P(class|object)，没学背景）。
- 部署端可以在软分数之外**额外**用 `quality_threshold` 过滤 anchor，
  但过滤只决定“留不留”，留下来的框分数仍然是 `obj * class_prob`。
  `decodeRaw18` 里就是 `if (quality < quality_threshold) continue;`
  之后 `score = quality * class_prob`。
- 训练侧 TAL 用的是 `preds["scores"]`（`cv_joint` 的 sigmoid 软分数），
  从来没用过硬门控后的 compact 分数，所以不需要改。

## 监控指标说明（每个 epoch 写进 results.csv）

新增的列分三组，**计算位置不同**，不要混着比较：

| 组 | 在哪儿算 | 统计对象 |
|---|---|---|
| `obj/*` | val 集（EMA 模型前向时顺带统计，不额外跑一遍验证） | objectness 通道（导出后唯一的候选来源） |
| `mon/*` | 训练 batch（TAL 分配之后） | `cv_joint` 的 TAL 排序质量 + 两套分类器一致性 |
| `assign/*` | 训练 batch（TAL 分配之后） | 每个 GT 实际分到多少正样本 anchor |

### obj/* —— objectness 独立评估（和 TAL 解耦）

先给 anchor 打标签，**不使用 TAL 的正负样本**（避免循环论证）：

- 正样本：anchor 中心落在**恰好 1 个** GT 框内；
- 歧义：落在 **≥2 个** GT 框内（重叠目标），正负都不算，直接剔除；
- 负样本：中心落在所有 GT 框外。

所有坐标都换算到 480×640 的 letterbox 坐标系（和网络输入一致）。

| 列 | 含义 |
|---|---|
| `obj/AP` | 用上面标签，按 objectness 概率排序的 101 点 AP（1000 个直方图 bin 近似）。衡量 objectness 把“目标区域 anchor”和“背景 anchor”分开的能力 |
| `obj/P@0.5` | obj>0.5 的 anchor 里，真的是正样本的比例 |
| `obj/R@0.5` | 正样本 anchor 里，obj>0.5 的比例 |
| `obj/GT_R@0.5` | **按 GT 统计**：该 GT 框内至少有一个 obj>0.5 的 anchor 的比例，即导出后候选集能覆盖多少 GT（pose 召回上限） |
| `val/P@0.5` / `val/R@0.5` | val 集上**固定 conf≥0.5** 的 precision / recall（IoU 0.5、类内贪心匹配）。和部署阈值同口径，不受日志 max-F1 阈值漂移影响 |
| `val/P@0.8` / `val/R@0.8` | 同上，conf≥0.8 的工作点 |

读法：`obj/GT_R@0.5` 高说明“候选集能盖住目标”，问题在后面的分类打分；它低说明 objectness 本身漏目标。
注意它是**区域型**指标（框内所有 anchor 都算正），数值不能和 `metrics/mAP50` 直接比较。

### mon/* —— TAL 排序与两条分类路径的一致性

在**训练 batch**上、TAL 分配之后统计。记 `s = max_k sigmoid(cv_joint_k)`（每个 anchor 的 14 类最大分），
正/负样本取 TAL 的 `fg_mask`。

| 列 | 含义 |
|---|---|
| `mon/s_auc` | 正样本 vs 负样本的 `s` 直方图 AUC。0.5 = 完全分不开（TAL 排序等于随机），越接近 1 越好 |
| `mon/s_gap` | 正样本 `s` 均值 − 负样本 `s` 均值（概率差）。>0 才有区分度，越大越稳 |
| `mon/attr_agree` | TAL 正样本上，`argmax(cv_joint)`（训练/TAL 用）与 `argmax(colour)×7+argmax(role)`（推理用）的一致率。1 = 两套分类器完全一致；低 = 两条路径在打架，compact 打分（导出用的那套）还没跟上 |

读法：`s_auc` 长期 0.5 或 `s_gap` ≈ 0 → 问题在 cv_joint/共享特征，TAL 分配质量差；
`s_auc` 高但 `attr_agree` 低 → 训练用的 cv_joint 和推理用的 colour×role 不一致，导出后精度会掉。

### assign/* —— 正样本分配饥饿（每个 GT 分到多少 anchor）

对每个 GT 统计“`fg_mask` 为真且 `target_gt_idx == 该 GT`”的 anchor 数；**一个都没分到的 GT 记 0**，不剔除。
GT 面积按 letterbox 后的像素框算：`(w_norm×W) × (h_norm×H)`，单位 px²。

分桶（边长）：**tiny < 32²、small 32²–96²、medium 96²–192²、large ≥ 192²**（可在 `monitor.ASSIGN_EDGES` 改）。
本仓库 TAL 的 `topk = 10`（`v8PoseLoss` 默认），所以“分满”就是 10。

| 列 | 含义 |
|---|---|
| `assign/{tiny,small,medium,large}_med` | 该面积桶内“每 GT 分到 anchor 数”的**中位数** |
| `assign/{tiny,small,medium,large}_iqr` | 同桶的 **q75 − q25**（IQR），看分布是否稳定；中位数够但 IQR 大说明部分 GT 被饿死 |
| `assign/{tiny,small,medium,large}_n` | 该桶的 GT 数量；太小（<20）时中位数不可信 |
| `assign/zero_frac` | 一个 anchor 都没分到的 GT 比例（最直接的饥饿指标） |
| `assign/mean_ratio` | 平均每 GT anchor 数 / topk |

读法（topk=10）：

- `med ≥ 8` 基本吃饱；`med < 5` 说明这一档目标在特征图上覆盖的格子太少，TAL 给不够正样本；
- `zero_frac > 0.05~0.10` 就是明显饥饿；
- `mean_ratio < 0.6` 说明整体分配不足；
- TAL 用的是**预测框**做对齐，所以训练早期（框还很差）这一项会偏悲观，重点看**趋势**；
- `large` 桶可能一直是 0，因为数据里几乎没有 >192 px 的目标，属正常。

### 其他已有列（简要）

| 列 | 含义 |
|---|---|
| `train/color_loss` / `train/number_loss` | 颜色（2 类）/ 角色（7 类）在 TAL 正样本上的交叉熵；推理打分直接来自这两个头 |
| `train/quality_loss` | 通道 8 objectness 的 BCE（全 anchor，带 pos_weight）；不是“质量回归”了 |
| `train/cls_loss` | `cv_joint` 的 14 类 BCE（全 anchor，TAL 软目标）；**只用于训练/TAL，不参与导出推理** |
| `train/geom_loss` | 几何先验正则，`--geom 0` 时恒为 0 |
| `metrics/*` | 标准 val 指标，按 `obj × colour × role` 软分数排序计算 |

`val/*` 用 EMA 模型在 val 集上算，`mon/*`、`assign/*` 用训练模型在**增强后的训练 batch**上算，
两者分布不同，只看趋势、不要横向比绝对值。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  ext/pico_pose/tests/test_smoke.py tests/test_pico_compact_p0.py -q -o addopts=""
```
