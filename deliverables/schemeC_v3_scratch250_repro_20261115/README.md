# schemeC_v3_scratch250 服务器训练复现包

生成于 2026-11-15。目标：在服务器已关机、本地工作区代码已被改动的前提下，
把 2026-11-12 22:14 → 11-13 11:48 那次 250 轮训练的
**代码 / 数据集构成 / 训练命令** 尽可能 1:1 复刻。

## 结论速览

| 项目 | 复刻依据 | 状态 |
|---|---|---|
| 代码 | git 提交 `5e9869ae8`（`code/` 内的 970 文件整树快照） | 已解包冒烟通过；61 个关键超参与 checkpoint 完全一致 |
| 数据集 | `build_pose23_schemeC_full.py` + `build_pose23_ws_bysize.py` + 全套元数据/checksum | 用 HEAD 脚本重建后，16,144 标签 md5、16,144 图片 md5、split_manifest 全部逐字节相同 |
| 命令 | 聊天中确认的原文（`train/TRAIN_COMMAND.sh`） | `--obj-logit-bias auto` 重算值 = checkpoint 记录值 3.2141383793739027 |
| 超参 | checkpoint 内 `train_args`（140 项）与最后一轮指标 | `train/args_resolved.yaml`、`train/final_metrics.json` |

---

## 1. 当时的真实时间线（+0800）

| 时间 | 事件 |
|---|---|
| 11-12 21:54 | 数据集上传完成并校验：`train 11219/11219`、`val 4925/4925`、`*.cache = 0`、2.1 GB |
| 11-12 21:58 | 服务器把 `data.yaml` 的 `path` 改写为 `/hy-tmp/ultralytics_2027/datasets/schemeC_pose23_ws_bysize` |
| 11-12 22:11 | 拷贝最终版 `ext/pico_pose/rect_train.py`（含“最后 N 轮关闭全部增强”补丁）到服务器 |
| 11-12 22:12 | 2 轮 8 图冒烟：日志出现 `closing mosaic and every train augmentation for the final epochs`，0 traceback |
| 11-12 22:14 | 启动 250 轮 |
| 11-13 11:48 | 训练完成（`ck['date'] = 2026-11-13T11:48:07.970177`） |
| 11-13 11:48–11:56 | 27 个 checkpoint 回传本地并逐个 md5 对服务器校验（ALL_27_MATCH） |
| 11-13 11:56 | 本地 CPU 导出 3 个 fp16 ONNX（last / best / epoch200） |

运行输出目录是 **`runs/pose/runs/pose/schemeC_v3_scratch250`**（双 `runs/pose`，原因见 §4），
不是笔误。

## 2. 代码：用 `5e9869ae8`，不要用当前工作区

```
code/ultralytics_2027_5e9869ae8.tar.gz   29 MB, 970 files
  sha256 4af75cf220daa6186c371fe3cd3a6f81aca3dc85a22d4e222a58c7c7efe417b7
code/HEAD.txt                            提交信息 + 该提交包含的文件清单
```

- 快照提交：`5e9869ae8bf1139f2c369aa362d727dd9b3dcecb`，父提交 `bb87ecaa8`。
- 该提交是一次**事后补提交**（提交时间 11-15 16:33），但内容就是 11-12 训练时的状态：
  它一次性收录了 `env_decouple.py` / `plate_scale.py` / `box_sync.py` / `rect_train.py` 的
  训练期补丁、两个数据集构建脚本、`train_schemeC.py` 的新增参数与 `default.yaml` 的新键。
- **当前工作区的 `armor-ulm` 改动（11-14，`ulm_blocks.py`、`tasks.py`、D3C2f/RepDown、
  1.649M 参数模型）是训练之后才做的**，不在快照里。用当前工作区训练得到的是另一个模型。
- 训练时模型规模：104 层、**2,724,104 参数、7.1 GFLOPs**，与 `5e9869ae8` 中的
  `ext/pico_pose/cfg/yolo11-pose-pico-compact.yaml` 一致。

快照中与这次训练直接相关的代码：

| 文件 | 作用 |
|---|---|
| `ext/pico_pose/rect_train.py` | `PicoRectTrainer`：480×640 矩形训练；`_close_dataloader_mosaic()` 在最后 20 轮把 **全部 18 个增强键清零**并重装 `KeypointBoxSync` |
| `ext/pico_pose/rect_augment.py` | `RectMosaic`，mosaic 画布 2H×2W |
| `ext/pico_pose/env_decouple.py` | `--env-p 0.4`：relocate / context / solo / scene / flat 五种环境解耦 |
| `ext/pico_pose/plate_scale.py` | `--plate-scale-*`：目标级缩放抖动（<16px 只放大） |
| `ext/pico_pose/box_sync.py` | 始终把 bbox 重算为四关键点 AABB（部署框语义） |
| `train_schemeC.py` | 训练入口；`--project` 默认 `runs/pose`；`--fliplr` 默认 0（镜像会翻装甲数字） |
| `ultralytics/cfg/default.yaml` | 新增 `env_p`、`plate_scale_*` 等键 |

已验证：解包后 `python train_schemeC.py --help` 正常，`PicoRectTrainer` /
`RandomEnvDecouple` / `RandomPlateScale` / `KeypointBoxSync` 均可导入；
用该快照解析同一条命令，61 个运行关键超参与 checkpoint 的 `train_args` **0 处不一致**
（含 `obj_logit_bias auto` 的 3.2141383793739027）。

## 3. 数据集构成

血缘（两级构建，脚本都在快照和本包 `dataset/` 中）：

```
datasets/pose23_current
  └─ scripts/build_pose23_schemeC_full.py      -> datasets/schemeC_pose23_full_v5
       label-only 过滤：白系列 / 5 号 / unknown 标签丢弃，visibility=0 的目标丢弃，
       混合图保留；14 类重映射；按 unit（源 train/val、source_json 捕获组、workspace 会话）
       做防泄漏切分
  └─ scripts/build_pose23_ws_bysize.py         -> datasets/schemeC_pose23_ws_bysize
       非 ws 行全部进 train（pose + source_json）；
       ws 会话按“每图最大目标长边（480×640 画布）的中位数”切：>= 24px 进 train，否则 val；
       ws 目标长边 < 12px 丢弃；丢空的 ws 图整张移除；非 ws 行保留全部目标
```

中间视图 `schemeC_pose23_full_v5`：266 个 unit（train 253 / val 19，overlap 0），
13,173 / 3,521 图，22,557 / 5,500 目标；`dataset/upstream_full_v5/` 保存了它的
`data.yaml`、`summary.json`、`README.md` 和 16,694 行 `split_manifest.tsv` 作为对账依据。

最终规模（`schemeC_pose23_ws_bysize`）：

| | train | val |
|---|---|---|
| 图片 / 标签 | 11,219 / 11,219 | 4,925 / 4,925 |
| 目标数 | 17,097 | 7,811 |
| ws 会话 | 106 | 155 |

丢弃：ws 目标 3,149 个；因目标全丢而移除的 ws 图 550 张。val 是纯 ws 域。

类别（14 类，`nc: 14`，`kpt_shape: [4,2]`，关键点顺序 LT/RT/RB/LB）：

| id | 名称 | train 目标 | val 目标 |
|---|---|---|---|
| 0 | blue_sentry | 574 | 529 |
| 1 | blue_hero | 1,507 | 401 |
| 2 | blue_engineer | 1,524 | 743 |
| 3 | blue_3 | 1,428 | 847 |
| 4 | blue_4 | 2,407 | 846 |
| 5 | blue_outpost | 747 | 588 |
| 6 | blue_base | 115 | 414 |
| 7 | red_sentry | 658 | 229 |
| 8 | red_hero | 1,082 | 236 |
| 9 | red_engineer | 2,005 | 432 |
| 10 | red_3 | 1,740 | 852 |
| 11 | red_4 | 1,846 | 887 |
| 12 | red_outpost | 1,403 | 590 |
| 13 | red_base | 61 | 217 |

标签格式：每行 13 列 = `cls cx cy w h`（归一化）+ 4 个关键点 × (x, y)（归一化）。
bbox 严格等于四关键点 AABB。

本包 `dataset/` 内容：

| 文件 | 说明 |
|---|---|
| `data.server.yaml` | 服务器上的 `data.yaml` 原文（`path: /hy-tmp/...`） |
| `data.local.yaml` | 本地版本（`path: /home/wbk/...`），其余字段相同 |
| `summary.json` | 构建统计（计数、类别分布、丢弃数） |
| `split_manifest.tsv` | 16,144 行逐图清单：源路径、unit、来源 split、目标 split、图名、标签名、目标数 |
| `labels_md5.txt.gz` | 16,144 个标签文件的 md5（相对路径） |
| `images_md5.txt.gz` | 16,144 张图（跟随软链）的 md5 |
| `image_symlinks.txt.gz` | 本地软链视图：每个条目 `images/... -> ../../../pose23_current/...` |
| `build_pose23_schemeC_full.py` / `build_pose23_ws_bysize.py` | 构建脚本原文 |
| `upstream_full_v5/` | 中间视图 `schemeC_pose23_full_v5` 的 data.yaml / summary.json / README.md / split_manifest.tsv（16,694 行） |
| `rebuild_and_verify.sh` | 一键重建 + 逐项校验 |
| `rebuild_check_20261115.json` | 本次复跑 `--dry-run` 的输出，与 `summary.json` 完全一致 |

已验证（2026-11-15）：以 `schemeC_pose23_full_v5` 为源重建，
`split_manifest.tsv` 与包内文件逐字节相同，16,144 个标签 md5 全对，
16,144 张图片 md5 全对，计数/类别分布全对。

注意：本地这份是**软链视图**（磁盘 138 MB），图片实体在 `datasets/pose23_current`；
服务器上是 `rsync -L` / `tar -h` 物化后的 2.1 GB 副本，两者内容由上面的图片 md5 保证一致。

## 4. 训练命令

```bash
cd /hy-tmp/ultralytics_2027
python train_schemeC.py \
  --data datasets/schemeC_pose23_ws_bysize/data.yaml \
  --epochs 250 --batch 16 --imgsz 480,640 --device 0 --workers 8 \
  --degrees 35 --env-p 0.4 \
  --plate-scale-p 0.35 --plate-scale-range 0.85,1.15 --plate-scale-max 2 --plate-scale-min-px 16 \
  --quality-soft 0.4 --obj-logit-bias auto --deploy-boxes \
  --name schemeC_v3_scratch250
```

`train/TRAIN_COMMAND.sh` 原样保存了这条命令与当时的 tmux 备选写法。

两个必须知道的细节：

1. **没有 `--project` 是有意的**。`train_schemeC.py` 默认 `--project runs/pose`，
   而 Ultralytics 的 `get_save_dir()` 会把相对 project 再拼到 `runs/<task>/` 下，
   所以真实落点是 `runs/pose/runs/pose/schemeC_v3_scratch250`。
   当时下载的权重就来自这个双路径。
2. `close_mosaic` 默认 20：最后 20 轮 `PicoRectTrainer._close_dataloader_mosaic()`
   把 mosaic/mixup/cutmix/copy_paste/degrees/translate/scale/shear/perspective/flip/HSV/
   erasing/photometric/env/plate-scale **全部置 0** 后重建 transforms，并重新安装
   关键点→框同步。上游原版只关 mosaic 三项，会漏掉旋转和 HSV，这一点当时特意补上了。

## 5. 超参（以 checkpoint 的 `train_args` 为准）

完整 140 项见 `train/args_resolved.yaml`（其中 `model:` 保留服务器历史路径）。
运行关键项：

| 分组 | 值 |
|---|---|
| 模型 / 数据 | `ext/pico_pose/cfg/yolo11-pose-pico-compact.yaml`；480×640 矩形；`rect=False`（由 PicoRectTrainer 接管） |
| 优化 | AdamW，lr0 `5e-4`，lrf `0.01`，cos_lr，weight_decay `5e-4`，warmup 5 ep（momentum 0.8 / bias_lr 0.1），nbs 64 |
| 轮次 / 保存 | epochs 250，patience 0，save_period 10，close_mosaic 20 |
| 精度 / 复现 | amp **False**，deterministic True，seed 0，pretrained True，从零初始化 |
| 几何增强 | mosaic 1.0，degrees 35，translate 0.1，scale 0.5，shear/perspective 0，fliplr **0**，flipud 0 |
| 颜色增强 | hsv_h 0.015，hsv_s 0.7，hsv_v 0.4，bgr 0；erasing 0；photometric_p **0** |
| 环境解耦 | env_p 0.4；relocate 0.35 / context 0.5 / solo 0.25 / scene 0.35 / flat 0.4；donors 32；context_scale 1.6,2.6 |
| 目标缩放 | plate_scale_p 0.35，range 0.85–1.15，max 2，min_px 16 |
| 损失权重 | box 7.5，cls 0.5，dfl 1.5，pose 12，kobj 1，color 0.1，number 0.1，quality 0.05，consistency 0，geom 0 |
| 属性监督 | quality_soft 0.4；obj_logit_bias = +3.2141383793739027（`auto` 由数据集重算，已核对一致）；deploy_boxes True |
| 其它 | 8 workers，device 0，plots，val，save，exist_ok False，single_cls False |

当时的服务器**只回传了权重**，没有 `args.yaml` / `results.csv`；上面参数是从
`weights/last.pt` 的 `train_args` 原样导出的，不是手抄。逐轮曲线/日志无法恢复，
`train/final_train_results.json` 是最后一轮的完整结果字典。

## 6. 环境

见 `env/environment.txt`。要点：RTX 3090 24GB、Python 3.11.11、torch 2.5.1+cu124、
ultralytics 8.4.30（editable，`/hy-tmp/ultralytics_2027`）、96 核、`/hy-tmp` 50GB。
导出 ONNX 的是本地机器（Py 3.9.23 / torch 2.8.0 CPU），不是训练机。

## 7. 复现步骤

```bash
# 1) 代码：解到任意目录（脚本会 chdir 到自己所在目录，所以它必须就是仓库根）
mkdir -p /hy-tmp && cd /hy-tmp
tar xzf ultralytics_2027_5e9869ae8.tar.gz
mv ultralytics_2027_5e9869ae8 ultralytics_2027
cd ultralytics_2027
pip install -e . --no-deps        # 保留机器自带 torch，避免被依赖解析改版本

# 2) 数据集：二选一
#    A. 直接使用现成树：把 datasets/schemeC_pose23_ws_bysize 物化拷贝过来
#       （本地是软链树，需 rsync -aL 或 tar -h），再把 data.yaml 的 path 改成新根；
#    B. 从源数据重建（需要 datasets/pose23_current 及其 split/ 元数据）：
#       build_pose23_schemeC_full.py 把 ROOT=/home/wbk/ultralytics_2027 写死在
#       rel_of()/SPLIT_DIR 里（训练时就是这样用），所以 full_v5 这一级要在该
#       路径下重建；build_pose23_ws_bysize.py 支持 --src/--out，位置随意。
python scripts/build_pose23_schemeC_full.py --src datasets/pose23_current --out datasets/schemeC_pose23_full_v5
python scripts/build_pose23_ws_bysize.py --src datasets/schemeC_pose23_full_v5 --out datasets/schemeC_pose23_ws_bysize
#       或一键重建+校验（已有 full_v5 时自动复用，实测 manifest/labels 全部一致）：
PY=python CODE=<解包后的代码根> bash dataset/rebuild_and_verify.sh /home/wbk/ultralytics_2027/datasets /tmp/ws_bysize_repro

# 3) 清缓存（服务器训练前是 0 个 cache；Ultralytics 首轮会自己重建）
find datasets/schemeC_pose23_ws_bysize -name '*.cache' -delete

# 4) 训练
bash <pack>/train/TRAIN_COMMAND.sh
#    或直接复制 §4 中的命令；产物在 runs/pose/runs/pose/schemeC_v3_scratch250
```

硬件/驱动不同不可能位级复现，但数据顺序、增强采样、初始化都由 seed 0 + deterministic
固定到同一分布；同型号 3090 + 同 torch 版本可复现到非常接近的曲线。

## 8. 那次训练的产物与结果

权重在 `runs/pose/schemeC_v3_scratch250/weights/`（27 个 `.pt`，md5 见
`train/weights_md5.txt`），另有本地导出的 `1113_last.onnx` / `1113_best.onnx` /
`1113_200.onnx`（各 4,435,593 B，fp16 权重 + fp32 I/O，通道 8 已烘焙 +3.214 的 Sub）。

最后一轮验证（纯 ws 域 val）：

| 指标 | 值 |
|---|---|
| Pose P / R | 0.86163 / 0.66664 |
| Pose mAP50 / mAP50-95 | 0.77669 / **0.76756** |
| Box P / R | 0.85808 / 0.66902 |
| Box mAP50 / mAP50-95 | 0.77239 / 0.60901 |
| obj/P@0.5 / obj/R@0.5 / obj/AP | 0.38757 / 0.84067 / 0.70032 |
| mon/s_auc / attr_agree / role_mask | 0.93881 / 0.93116 / 0.97979 |
| fitness | 1.37657 |

`best.pt` 与 `last.pt` 的 md5 不同，但两者导出后的 ONNX 输出逐元素相同
（best 正好落在最后一轮，两者都被 strip 成 `epoch=-1`）；`epoch200` 与 last 的
最大输出差 1.453125。

## 9. 已知边界

- 服务器上的 `args.yaml`、`results.csv`、训练日志没有回传；参数从 checkpoint 恢复，
  逐轮数值不可恢复（最终轮指标在 `train/final_metrics.json`）。
- 代码快照排除了工作区 11-14 的 `armor-ulm` 改动；那是之后的实验，模型结构不同。
- 数据集图片的 mtime 在服务器上是上传时间，内容一致性以本包 md5 为准。
- `data.server.yaml` 中的 `/hy-tmp/...` 是历史路径，换机器时改成新的数据集根即可。
