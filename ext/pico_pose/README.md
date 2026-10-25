# ext/pico_pose — 无框轻量 Pose 头

这是 `PicoPoseNoBox`（纯无框）和 `PicoPoseAuxBox`（训练期辅助框、导出无框）的
**唯一真实实现**。上游 `ultralytics/` 里只剩少量带 `# EXT:pico_pose` 的接入点，不再有 shim。

> 2026-10 起不再兼容旧权重：`runs/0913poseNoBox.pt` 里 pickle 的是
> `ultralytics.nn.modules.picopose.PicoPoseNoBox`，shim 删除后无法加载。
> 新权重统一走 `ext.pico_pose.head`，用当前代码重新训练/导出即可。
>
> 2026-10-24 起输出块从单层 1x1 改为 `DW 3x3 -> 1x1`（`head_k=3`）。结构变了之后，
> 即使设 `head_k=1`，state_dict key 也是 `cls_pred.0.pred.*`，旧的 1x1 权重不能直接加载；
> 当前仓库里没有需要兼容的 Pico 训练权重。

## 目录

```text
ext/pico_pose/
├─ head.py      # PicoPoseNoBox / PicoPoseAuxBox / PicoDWBlock / PicoSE / PicoHeadBlock
├─ loss.py      # PicoPoseNoBoxLoss(ATSS/Wing) + PicoAuxPoseLoss(复用 v8PoseLoss)
├─ cfg/         # yolo12 / yolo11 各两份：nobox 与 auxbox
├─ train.py     # 训练入口（默认用本目录的 yaml）
├─ tests/       # 结构、前向、导出通道的冒烟测试
└─ README.md
```

## 上游为什么还留着东西

| 上游位置 | 内容 | 为什么还留着 |
| --- | --- | --- |
| `ultralytics/nn/tasks.py` | 4 处 `# EXT:pico_pose` | `parse_model` 按类名从 globals 解析模块，且 head 参数注入 / `legacy` 赋值 / `init_criterion` 是硬编码的，只能在此接入 |
| `pyproject.toml` | `packages.find.include` 加 `ext*` | 让 `pip install -e .` 同时映射 `ext`，否则在仓库外 `import ultralytics` 会找不到 `ext` |

安装（改了 `ext/` 结构后需要重跑一次）：

```bash
cd /home/wbk/ultralytics
pip install -e . --no-deps --no-build-isolation
```

导入时请用 `from ext.pico_pose import PicoPoseNoBox`；不要把
`import ext.pico_pose.head` 作为进程第一个相关导入（会和 `ultralytics.nn`
初始化期间的反向导入循环）。

新增自定义头时，**不要**再往 `ultralytics/cfg/models/` 或 `ultralytics/utils/` 里塞文件；
按同样方式在 `ext/<feature>/` 建目录，上游只留带 `# EXT:` 标记的接入行。

## 与 YOLO11 / YOLO12 的兼容性

头只依赖颈部输出的三层特征图（P3/8、P4/16、P5/32）和 `nc`/`kpt_shape`，
与骨干类型无关，所以同一份 `PicoPoseNoBox` 可以直接挂到：

| 配置 | 头输入层索引 | 4 点单类参数量（实测，`head_k=3`） | `head_k=1`（旧结构） |
| --- | --- | --- | --- |
| `cfg/yolo12-pose-pico-nobox.yaml` | 14 / 17 / 20 | 2,211,399 | 2,207,175 |
| `cfg/yolo11-pose-pico-nobox.yaml` | 16 / 19 / 22 | 2,233,191 | 2,228,967 |
| `cfg/yolo12-pose-pico-auxbox.yaml` | 14 / 17 / 20 | 2,334,871 | 2,330,647 |
| `cfg/yolo11-pose-pico-auxbox.yaml` | 16 / 19 / 22 | 2,356,663 | 2,352,439 |

3x3 输出块固定只多 4,224 参数（3 层 × cls/reg × 704），480×640 下 GFLOPs 增加约 0.015
（`model.info()` 两种结构都显示 5.1）。

两个骨干的 `Pose` 头接口相同，yaml 里只把 `Pose` 换成 `PicoPoseNoBox` / `PicoPoseAuxBox` 即可；
`PoseModel` / `PoseTrainer` / 损失都不用改。

## 输出块：DW 3x3 -> 1x1

早期版本 cls/reg 都是单层 1x1（逐点线性映射），空间上下文完全依赖前面的 5x5 DW stem；
类别数一多，分类边界容易欠拟合。现在每个分支各有一个 `PicoHeadBlock`：

```text
共享 stem(2 × DW5x5+PW) -> SE -> ┬ DW 3x3 (groups=feat_out) + BN + Hardswish -> 1x1 -> nc
                                 └ DW 3x3 (groups=feat_out) + BN + Hardswish -> 1x1 -> nk
```

- 只改 head 内部，验证/导出的通道契约不变（仍是 `4+nc+nk` / `nc+2*nkpt` 等）；
- 相邻网格先做非线性融合再投影，类别数变大时 `nc` 通道共享同一份空间上下文；
- `head_k=1` 退回旧的单层 1x1（结构等价，state_dict key 不同，见文件头说明）；
- `head_dw=False` 把 DW 3x3 换成完整 3x3，每层每分支比 DW 版多 36,288 参数（相比旧 1x1 +36,992），
  用于容量上限对比；
- 通过 `PICO_HEAD_CFG` 或 `ULTRALYTICS_PICO_HEAD="head_k=1"` 切换，不需要改 yaml。

## 训练期辅助框（推荐）

纯无框训练时，pose 一项占总加权损失的 ~99.8%，cls 只有 ~0.18%，所以 precision/recall
在前期会抖。`PicoPoseAuxBox` 在训练/验证时额外挂一个**不进入导出图**的框分支：

- 共享 PicoFeat stem → 3×3 Conv → 1×1 Conv(`4*reg_max`)，配 DFL；
- 训练 `forward` 多返回 `boxes`，`PicoAuxPoseLoss` 直接复用上游 `v8PoseLoss`
  （TAL + box/DFL + KeypointLoss），只重写 `kpts_decode` 适配 Pico 的
  `(anchor + raw) * stride` 约定；
- 验证时也会算 `boxes` 供 validator 的 val loss 使用，但指标仍走 `_inference`
  的伪框 + 关键点输出；
- `export=True` 时完全不执行 cv2/dfl。实测 aux 与 no-aux 的 ONNX 输出都是
  `[1, 9, 6300]`，图里没有 `model.21.cv2*` / `model.21.dfl*` 初始化器。

训练命令：

```bash
python -m ext.pico_pose.train \
  model=ext/pico_pose/cfg/yolo12-pose-pico-auxbox.yaml \
  data=/home/wbk/train_data/0802_1/windmill_4pt_pose.yaml \
  imgsz=480,640 epochs=300 batch=16 device=0
```

`PicoPoseNoBox`（纯无框、ATSS/Wing）保留作为 A/B 对照组，不要删。

## 输出通道契约（部署端必须对齐）

设 `nc` 为类别数，`nkpt` 为关键点数，`ndim` 为每点维度（2 或 3）。

| 阶段 | 返回 | 通道数 |
| --- | --- | --- |
| 训练（纯无框） | `dict(scores, kpts, feats)` | `nc` / `nkpt*ndim` |
| 训练（aux box） | `dict(scores, kpts, boxes, feats)` | 额外 `4*reg_max`，仅 loss 使用 |
| 验证 | `Tensor`（伪框 + 分数 + 解码关键点） | `4 + nc + nkpt*ndim` |
| 导出（`export_xy_only=True`，默认） | `Tensor`（分数 sigmoid + 原始 xy） | `nc + 2*nkpt` |
| 导出（`export_xy_only=False`） | `Tensor`（分数 sigmoid + 原始 xy + 可见性 logits） | `nc + nkpt*ndim` |

注意：导出分支默认不会解码关键点，也不会对可见性做 sigmoid；C++/部署端需要自己完成
`(anchor + raw) * stride` 和 `sigmoid(vis)`。这是和旧 `PoseNoBox`（导出解码后的像素坐标）
最大的契约差异。

### 部署导向的导出选项（默认关闭，不改变旧契约）

| 选项 | 效果 | 代价 |
| --- | --- | --- |
| `export_row_major=True` | 输出 `[B, A, C]`（行优先，每个候选 C 个 float 连续） | 末端 1 个 Transpose，约 0.02 ms |
| `export_decode=True` | 图内把 kpt xy 解码成像素：`(anchor + raw) * stride` | 约 0.2–0.3 ms |

开启后 `C = nc + 2*nkpt`（`export_xy_only=True`）或 `nc + nkpt*ndim`；
C++ 侧直接从 `row = data + i*C` 读，若同时 `export_decode=True` 则不用再查 anchor/stride。
`PicoPoseAuxBox` 的辅助框分支仍然不会进导出图。

```bash
ULTRALYTICS_PICO_HEAD="export_row_major=1,export_decode=1" \
python -m ext.pico_pose.train ...   # 训练时写进配置；导出时也会读该环境变量
```

旧 checkpoint 的 head 对象没有这两个属性时，`_inference` 会回退读 `PICO_HEAD_CFG`，
所以加载旧 `.pt` 直接导出也能用环境变量打开。

精度方面：**CPU 目标用 `half=False`**，导出图里 0 个 Cast；FP16 I/O（`to_fp16.py` 默认的
`keep_io_types=False`）也是 0 Cast，但 ORT CPU 会插入内部转换、实测更慢；`half=True`
默认保留 FP32 I/O + 2 个边界 Cast。GPU/TensorRT 再考虑 FP16 I/O。

**验证态伪框是 `xywh`**（`_inference` 里由关键点 min/max 合成，中心+宽高），因为
Ultralytics 的 NMS 会按 `xywh -> xyxy` 再转一次。2026-10-24 修复了此前误输出 `xyxy`
导致框被二次放大、box 指标恒为 0 的问题（`tests/test_smoke.py::test_validation_pseudo_box_is_xywh`）。
另外，box 指标衡量的是“关键点 hull”与 GT 平板框：实测 hull/GT IoU 均值约 0.34，
只有 0.3% 超过 0.5，所以 **box mAP 接近 0 是预期**，4 点任务看 Pose 指标即可。

⚠️ 目前 `runs/0913poseNoBox.onnx` 是 `nc=1, kpt_shape=(4,3)` 且默认 `export_xy_only=True`，
输出为 `[1, 9, 6300]`（1 + 4×2）。如果部署端要的是 `nc + nkpt*3`（例如 3 类 5 点 = 18），
必须显式设置 `export_xy_only=False`。修改前先更新 `tests/test_smoke.py` 里的
`test_export_channel_contract`。

`PICO_HEAD_CFG` 支持 `ULTRALYTICS_PICO_HEAD="feat_out=96,use_se=1,head_k=3,head_dw=1,aux_box=1,export_row_major=1,export_decode=1"`
环境变量；该变量会改变网络结构或导出契约，训练时请写进实验 README，避免复现时对不上。
**默认结构已含 `head_k=3`，只有做 A/B 时才需要显式写 `head_k=1`。**

## 运行

```bash
cd /home/wbk/ultralytics

# 方式一：直接用扩展入口（model 默认已指向 yolo12 auxbox 配置）
python -m ext.pico_pose.train \
  data=/home/wbk/train_data/0802_1/windmill_4pt_pose.yaml \
  imgsz=480,640 epochs=300 batch=16 device=0

# YOLO11-pose 版本：同一个头，换成 11 的骨干/颈部
python -m ext.pico_pose.train \
  model=ext/pico_pose/cfg/yolo11-pose-pico-auxbox.yaml \
  data=/home/wbk/train_data/0802_1/windmill_4pt_pose.yaml \
  imgsz=480,640 epochs=300 batch=16 device=0

# 对照组：纯无框（ATSS + Wing）
python -m ext.pico_pose.train \
  model=ext/pico_pose/cfg/yolo12-pose-pico-nobox.yaml \
  data=/home/wbk/train_data/0802_1/windmill_4pt_pose.yaml \
  imgsz=480,640 epochs=300 batch=16 device=0

# 冒烟测试
pytest ext/pico_pose/tests -q
```

## 新增实验的规则

1. 一次性实验放 `projects/YYYYMMDD_<name>/`，写清 commit、命令、数据 yaml、结果；
2. 可复用的代码才进 `ext/pico_pose/`，并补测试；
3. 改上游接入点必须带 `# EXT:pico_pose` 标记，并在 `patches/` 留 patch；
4. 权重、ONNX、`runs/` 不进 git。
