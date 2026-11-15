# Compact 四关键点 Pose 头（Scheme C）

`ext.pico_pose` 提供 **Scheme C 14 类**装甲板四关键点 Pose 头，骨干/颈部换成 YOLO-ULM 的
`D3C2f` + `RepDown` 块，输出仍是 P3/P4/P5 三层。

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
| `cv_compact` | Conv1x1(32→9) | 8 关键点 + 1 colour logit |
| `cv3` | DWConv3x3 → DWConv3x3 → Conv1x1(32→7) | 7 role logits：TAL 分类 + 导出置信度 |

没有 objectness 通道。7 个 role logit 就是 YOLOv8 语义的 cls 头：sigmoid + BCE 在**所有**
anchor 上训练，TAL 分配和导出图用同一套分数。colour 是共享特征 `f` 上的条件属性：
`p_red = sigmoid(z)`，`color_p = [1 - p_red, p_red]`。

`cv4` 不使用，P3 不加额外卷积块。导出执行 `cv_shared + cv_compact + cv3`。

> 头部已从旧的 18 通道（objectness + `cv_joint`）改掉，**旧 checkpoint 不兼容**，直接从头训练。

## 骨干/颈部：D3C2f + RepDown（YOLO-ULM）

`ext/pico_pose/ulm_blocks.py` 按 YOLO-ULM（Han et al., CVPR 2026）实现两个块，cfg 里把
8 个 `C3k2` 换成 `D3C2f`、5 个 stride-2 `Conv` 换成 `RepDown`（其余层 index 不变）：

| 位置 | 旧 | 新 |
|---|---|---|
| backbone 2/4/6/8 | C3k2 | D3C2f |
| backbone 3/5/7 | Conv k3 s2 | RepDown |
| neck 13/16/19/22 | C3k2 | D3C2f |
| neck 17/20 | Conv k3 s2 | RepDown |

- `D3Block` = DSConv(3) → DWConv(7) → DSConv(3) + 残差；`D3C2f` 是 C2f 式聚合。
- `RepDown` = 7×7 / 3×3 两条 stride-2 深度卷积分支 concat 后再各接 1×1 PW。
  **没有真正的 rep fusion**：两条分支都留在推理图里，`RepDown.fuse()` 只折叠 PW 的
  Conv+BN（幂等，BN 已被 `BaseModel.fuse` 删掉也安全）。
- 与参考实现的有意差异：`D3C2f(c1, c2, n, shortcut, e, g)` 参数顺序修正了参考仓库的
  g/e 错位（参考 yaml 的 `0.25` 落在未使用的 g 上、layer 22 的 `True` 被当成 e=1.0）。
  我们统一 e=0.5，因此 layer 22 比参考实现窄。
- 代价：D3C2f 的内部 key 和 C3k2 不同，**换块前的 checkpoint 不能按名加载**；
  官方 `yolo11n-pose.pt` 能按名对齐 193/649 个 key（1.361M / 1.649M 参数）：layer
  6/8/13/16/19/22 的 `cv1/cv2` 和 head 的 `cv2+dfl` 可迁移；layer 2/4 因 e 不同
  （官方 0.25）只有 BN 统计量对齐，D3Block/PW 内部随机初始化。

实测（`ultralytics.utils.torch_utils.get_flops`，输入 480×640 / 640，官方权重同口径）：

| 模型 | 参数 | 480×640 | 640 |
|---|---|---|---|
| 新骨干 + compact 头 | **1.649M** | **3.886G** | **5.181G** |
| 旧 C3k2 骨干 + compact 头 | 2.564M | 4.745G | 6.327G |
| 官方 yolo11n-pose | 2.874M | 5.657G | 7.542G |

导出仍是 `[1, 6300, 16]`（480×640），ONNX Runtime 与 PyTorch 最大差 7.5e-7。

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

## 曝光/运动增广与 role mask（v2 微调用）

> 本节的微调命令针对旧 18 通道头（含 objectness/`cv_joint`）。当前头结构不兼容，
> 用新代码必须从头训练；数据增广和 role mask 参数仍然适用。

部署相机比训练集主体暗得多（帧均值 5~10 vs 30~45），role 头会把“暗、没有亮数字”的板子
判成 outpost。`RandomPhotometric` 在 train split 上按**目标均值**采样 gamma：

```text
target_mean ~ U(5, 60)
gamma = log(target_mean/255) / log(mean/255)
```

每张图独立采样，所以每个类都会同时出现在暗和亮的样本里，切断“亮度/类别”的相关性；再叠加
motion blur 和 Gaussian noise。它插在 mosaic/affine/flip 之后、`Format` 之前，输入仍是
uint8 HWC，只加训练算力，导出图不变。

```bash
--photometric-p 0.6 --photometric-target 5,60 \
--photometric-blur-p 0.3 --photometric-noise-p 0.2
```

`--consistency 0.1` 再加一条 **clean → 暗光/模糊 view 的 role/colour KL**：clean 视图做
teacher（detach），photometric 视图做 student，只走 `cv_shared + cv_compact + cv3`（临时
`aux_box=False`，不跑 `cv2`）。它让导出的 colour/role 头在暗光/模糊输入下保持
同一个后验。训练算力约 ×2，val 不跑（`val/consistency_loss` 为 0）；`train/consistency_loss`
进 CSV。模型引用用 module 级 `_MODEL_REFS`（criterion id → weakref）记录，避免 criterion 作为
model 子模块时形成引用环或让 checkpoint 不可 pickle；从旧 checkpoint 微调时
`PicoRectTrainer.setup_model` 会丢掉 checkpoint 里的旧 criterion，保证新的
`color/number/consistency` 权重生效。

`--role-mask-area 20,80` / `--role-mask-ratio 0.10,0.20` 用 GT 四角点的面积和短/长边比给
colour CE 一个软权重：太小或太侧视的板子只训 box/pose/role，不逼颜色头猜。
`mon/role_mask` 是正样本上的平均权重（1 = 完全没 mask）。`--role-label-smoothing 0.05`
给 colour/role CE 加 label smoothing（不要用上游已删除的 `label_smoothing` 键）。

微调时先裁 val，避免 outpost 主导 `best.pt`：

```bash
python scripts/cap_schemeC_val.py \
  --src datasets/schemeC_split_bal \
  --out datasets/schemeC_split_bal_capval \
  --max-per-group 120
```

它保留全部 train，只对每个 val capture group 最多保留 120 张，丢掉的帧不会进 train，
所以 group 级 train/val 隔离不变。

从 v2 checkpoint 微调（**不要 `--resume`**，`--resume` 会恢复旧 schedule/args）：

```bash
python train_schemeC.py \
  --weights runs/pose/runs/pose/schemeC_480x640_v2/weights/best.pt \
  --data datasets/schemeC_split_bal_capval/data.yaml \
  --epochs 40 --batch 16 --imgsz 480,640 --device 0 --workers 8 \
  --lr0 1e-4 --lrf 0.05 --warmup-epochs 2 --close-mosaic 10 \
  --photometric-p 0.6 --photometric-target 5,60 \
  --photometric-blur-p 0.3 --photometric-noise-p 0.2 \
  --consistency 0.1 --color 0.2 \
  --role-label-smoothing 0.05 \
  --role-mask-area 20,80 --role-mask-ratio 0.1,0.2 \
  --name schemeC_v2_ft_aug
```

## 环境解耦增广（P0）

role 头会把“场景”当特征：同一批单板 val，原图 role 96.8%，只留板贴灰底 27.1%，
贴到别的真实场景 46.6%，color 始终 100%。`RandomEnvDecouple` 在 train split 上做四种
可开关的破坏（`--env-p` 是总开关，默认 0）：

- `relocate`：抠出板（四角 mask），inpaint 原位置，再贴到画面内随机空位；
- `context`：把板周围 1.6~2.6 倍板尺寸的邻域换成画面别处的背景；
- `solo`：只留一块板，其余连标签一起 inpaint；
- `scene`：用训练集里无板的 crop（或合成渐变）替换整张背景，再把板贴回去。

它插在 photometric 之后、`Format` 之前，只改训练图，导出图不变。`--env-donors` 控制
donor bank 数量（32 张 1/4 分辨率 crop，约 2 MB，fork 后 worker 共享）。

贴回/擦除都用 `_painted_extent`（四关键点 + 数字上下各 20% 长边、左右 6% 的**真实板面**），
不是关键点 AABB：`context` / `scene` 之前只贴回 quad+1px，板子上下数字被背景替换而
标签还在（实测 96% / 100% 的亮像素被换掉）；`solo` 之前只 inpaint quad+2px，被删的板
会留下大片无标签数字。两处都已修正，点级回归见 `tests/test_env_decouple.py`。

```bash
--env-p 0.4 --env-relocate-p 0.35 --env-context-p 0.5 \
--env-solo-p 0.25 --env-scene-p 0.35 --env-flat-p 0.4 --env-donors 32
```

选模不能只看 mAP：`scripts/val_pico_env_swap.py` 在同一批单板 val 上比较
orig / local（灰底）/ swap（换真实场景），输出 match、role@hit、role@all 和逐类 role。
`--mask plate` 只贴板面像素（更严格，默认），`--mask rect` 是旧的带邻域 patch：

```bash
python scripts/val_pico_env_swap.py \
  --weights runs/pose/runs/pose/schemeC_480x640_v2/weights/best.pt --n 400
```

v2 小样本冒烟（n=64，`--mask plate`）：orig 96.8% / local 27.1% / swap 46.6% role@hit，
color 三档都是 100%。

## 目标级缩放抖动（plate scale jitter）

`--scale` 是整图仿射缩放，同一张图里所有板随场景一起变大变小。`RandomPlateScale`
只改**单块板**的视在尺寸：用与 env relocate 相同的 margin 把板抠出来（含数字上下
出界部分），缩/放 patch 后在原中心贴回，关键点跟着同一仿射变换，框重算为四关键点
AABB。插在 env 之后、`Format` 之前，只作用于 train split。

```bash
--plate-scale-p 0.35 --plate-scale-range 0.85,1.15 --plate-scale-max 2 --plate-scale-min-px 16
```

- 小目标保护：四关键点 AABB 长边 < 16px 的板**只保持或放大**，缩放因子从
  `[max(lo,1), max(hi,1)]` 重采样，不会被缩到 16px 以下；
- 放大后与同图其他板重叠、或缩放后越出画布，这张图上的该板跳过（不裁边、不重叠）。
  重叠判定用双方的真实板面范围（含数字上下悬出部分），不是关键点 AABB；贴回越界的
  检查放在 inpaint 之前，避免“板被擦掉但标签还在”；
- 场景尺度不变，模型不能再用“目标大小 ↔ 距离/场景”的捷径；`0.85,1.15` 是 ±15%
  的小幅抖动，每图最多 `--plate-scale-max` 块板。

单板数据实测（`/tmp` 临时脚本）：300 张图 284 次缩放，贴回范围与其他板真实板面
最近距离 15.7px，0 次覆盖；全 pipeline（env+p 0.4、plate 0.35、mosaic）121 次缩放
同样 0 次覆盖。

训练数据 `datasets/schemeC_pose23_ws_bysize`：非 ws 全部进 train；ws 采集 session
按“每图最大目标长边”的中位数切分，≥24px 整段进 train、其余进 val；ws 中长边
<12px 的目标删掉，删空的图整张不进数据集。

```bash
python scripts/build_pose23_ws_bysize.py
```

## 大角度 roll（±60°）与框同步

`--degrees` 是整图 in-plane 旋转（关键点一起转），和部署时相机 roll 是同一种变换。
v2_ft_env 只用了 ±5°，roll 超过 ±15° 就出误识别/分类错，roll 微调直接开满：

```bash
--degrees 60 --translate 0.1 --scale 0.5
```

- 旋转发生在 mosaic 的 2H×2W 画布上，最后裁回 480×640。120 张 train 探针：
  `degrees=60` 的 keep 数与 5° 持平（2.60 vs 2.48 个/图），没有额外丢标签；
  close-mosaic 之后（单图 letterbox 路径）被边界裁掉关键点的实例从 2.1% 升到 3.2%；
- 上游 `RandomPerspective` 转的是**框的四个角**。板子的四角是透视四边形，旋转后
  GT 框会比“四关键点 AABB”大（60° 时 60% 实例 IoU<0.9，5° 时 31%）。所以
  `degrees > 0` 时 `PicoRectTrainer` 自动在 `Format` 前插入 `KeypointBoxSync`：
  四角全可见的实例把框重算成关键点 AABB（IoU=1.0），被边界裁掉的实例保留上游
  裁边框（角点已 clamp，再取 AABB 会偏小）。这样 TAL 分配和部署角点框用同一套
  几何口径；
- `--fliplr` / `--flipud` 仍然必须是 0（数字会镜像），roll 不受影响。

## 损失槽位

```text
[box, pose, kobj, cls, dfl, color, geom, consistency]
```

- `cls` 就是 7 个 role logit 的 TAL BCE（全 anchor、软目标），**同一组 logit 就是导出置信度**；
  14 类标签只在 TAL 调用里映射成 role（`attrs.ROLE_BY_CLASS`，`assigner.num_classes = 7`）。
- `color` 是单 logit BCE，只作用于 TAL 正样本，权重来自 role mask。
- `geom` 是几何先验正则（`--geom 0` 时恒为 0）。
- `consistency` 是训练期 clean/photometric 视图的 per-logit Bernoulli KL（`--consistency`，
  val 为 0），不是导出通道的一部分。

验证/NMS/导出统一用
`score[c*7+r] = color_p[c] * sigmoid(role_logit[r])`，`class = (p_red>=0.5)*7 + argmax(role)`。

## 验证即部署（deploy parity，默认开）

`deploy_boxes=True`（`PicoRectTrainer` 从训练参数里推到 head 上）让 **val/NMS 用 4 个预测角点
合成框**，而不是训练用的 `cv2`+DFL 分支——导出图里本来就没有 box。同一个 checkpoint、同一
768 张 val 子集，两种框的差别：

| NMS 框 | mAP50(P) | mAP50-95(P) | P | R |
|---|---|---|---|---|
| DFL box 分支 | 0.8522 | 0.8129 | 0.723 | 0.818 |
| 4 角点合成 | **0.9085** | **0.8604** | 0.871 | 0.838 |

（表是旧 18 通道头的实测，只用来说明两种框口径的差异量级；新头的绝对值会变。）
关掉用 `--no-deploy-boxes`。注意这会让 `results.csv` 的 mAP 口径变化，跨版本比较时先对齐。

## 导出

`head.export = True` 时只执行 `cv_shared + cv_compact + cv3`，输出 `[B, A, 16]`：

```text
0:8    raw keypoint offsets      p = (raw + anchor) * stride
8      colour logit              p_red = sigmoid(z)
9:16   role logits              sigmoid（内置置信度，没有额外 objectness）
color_p = [1 - p_red, p_red]
score   = color_p[c] * sigmoid(role_logit[r])     # 排序 / mAP / TAL 都用它
class   = (p_red >= 0.5) * 7 + argmax(role)
```

- 导出图没有 box 分支：部署端用 4 个解码角点合成 AABB，和 `deploy_boxes=True` 的
  验证口径一致（见上一节）。
- 训练侧 TAL 用的 `preds["scores"]` 就是这 7 个 role logit，和导出完全同源，
  不存在 train/deploy 打分路径不一致的问题。
- **C++ 解码契约已变**：旧 `decodeRaw18` 的 quality 通道和 softmax 都不再适用，
  端侧需要按 16 通道重写（`ros1_ws` 里的节点尚未同步）。

## 监控指标说明（每个 epoch 写进 results.csv）

新增的列分三组，**计算位置不同**，不要混着比较：

| 组 | 在哪儿算 | 统计对象 |
|---|---|---|
| `anchor/*` | val 集（EMA 模型前向时顺带统计，不额外跑一遍验证） | 7 个 role logit 的最大 sigmoid（导出后唯一的候选来源） |
| `mon/*` | 训练 batch（TAL 分配之后） | role 头的 TAL 排序质量 + role mask 覆盖率 |
| `assign/*` | 训练 batch（TAL 分配之后） | 每个 GT 实际分到多少正样本 anchor |

### anchor/* —— role 置信度独立评估（和 TAL 解耦）

先给 anchor 打标签，**不使用 TAL 的正负样本**（避免循环论证）：

- 正样本：anchor 中心落在**恰好 1 个** GT 框内；
- 歧义：落在 **≥2 个** GT 框内（重叠目标），正负都不算，直接剔除；
- 负样本：中心落在所有 GT 框外。

所有坐标都换算到 480×640 的 letterbox 坐标系（和网络输入一致）。

| 列 | 含义 |
|---|---|
| `anchor/AP` | 用上面标签，按 role 置信度排序的 101 点 AP（1000 个直方图 bin 近似）。衡量置信度把“目标区域 anchor”和“背景 anchor”分开的能力 |
| `anchor/P@0.5` | conf>0.5 的 anchor 里，真的是正样本的比例 |
| `anchor/R@0.5` | 正样本 anchor 里，conf>0.5 的比例 |
| `anchor/GT_R@0.5` | **按 GT 统计**：该 GT 框内至少有一个 conf>0.5 的 anchor 的比例，即导出后候选集能覆盖多少 GT（pose 召回上限） |
| `val/P@0.5` / `val/R@0.5` | val 集上**固定 conf≥0.5** 的 precision / recall（IoU 0.5、类内贪心匹配）。和部署阈值同口径，不受日志 max-F1 阈值漂移影响 |
| `val/P@0.8` / `val/R@0.8` | 同上，conf≥0.8 的工作点 |

读法：`anchor/GT_R@0.5` 高说明“候选集能盖住目标”，问题在后面的分类打分；它低说明 role 置信度本身漏目标。
注意它是**区域型**指标（框内所有 anchor 都算正），数值不能和 `metrics/mAP50` 直接比较。

### mon/* —— TAL 排序质量与 role mask 覆盖

在**训练 batch**上、TAL 分配之后统计。记 `s = max_k sigmoid(role_k)`（每个 anchor 的最大
role 分——导出/NMS 用的就是它），正/负样本取 TAL 的 `fg_mask`。

| 列 | 含义 |
|---|---|
| `mon/s_auc` | 正样本 vs 负样本的 `s` 直方图 AUC。0.5 = 完全分不开（TAL 排序等于随机），越接近 1 越好 |
| `mon/s_gap` | 正样本 `s` 均值 − 负样本 `s` 均值（概率差）。>0 才有区分度，越大越稳 |
| `mon/role_mask` | 正样本上 role mask 的平均权重（1 = 完全没 mask，接近 0 = 大量小/侧视板被降权） |

读法：`s_auc` 长期 0.5 或 `s_gap` ≈ 0 → 问题在共享特征或 role 头，TAL 分配质量差；
`role_mask` 长期很低 → 数据里小目标/侧视板占比高，colour/role 实际监督量不足。

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
| `train/color_loss` | 单 logit colour BCE（TAL 正样本，role mask 加权）；推理的 `p_red` 直接来自它 |
| `train/cls_loss` | 7 role 的 BCE（全 anchor，TAL 软目标）；**同一组 logit 就是导出置信度** |
| `train/geom_loss` | 几何先验正则，`--geom 0` 时恒为 0 |
| `train/consistency_loss` | clean/photometric 视图的 Bernoulli KL，`--consistency 0` 时恒为 0 |
| `metrics/*` | 标准 val 指标，按 `color_p[c] × sigmoid(role[r])` 分数排序计算 |

`val/*` 用 EMA 模型在 val 集上算，`mon/*`、`assign/*` 用训练模型在**增强后的训练 batch**上算，
两者分布不同，只看趋势、不要横向比绝对值。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  ext/pico_pose/tests/test_smoke.py tests/test_pico_compact_p0.py -q -o addopts=""
```
