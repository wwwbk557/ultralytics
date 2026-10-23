# ext/pico_pose — 无框轻量 Pose 头

这是 `PicoPoseNoBox`（PicoDet-S 风格、去掉框回归分支）的**唯一真实实现**。
上游 `ultralytics/` 里只保留 re-export shim 和少量接入点。

## 目录

```text
ext/pico_pose/
├─ head.py      # PicoPoseNoBox / PicoDWBlock / PicoSE
├─ loss.py      # PicoPoseNoBoxLoss + ATSS / VFL / Wing
├─ cfg/         # 模型结构 yaml
├─ train.py     # 训练入口（默认用本目录的 yaml）
├─ tests/       # 结构、前向、导出通道的冒烟测试
└─ README.md
```

## 上游为什么还留着东西

| 上游位置 | 内容 | 不能搬走的原因 |
| --- | --- | --- |
| `ultralytics/nn/modules/picopose.py` | 2 行 re-export | 旧 checkpoint 的 pickle 路径就是 `ultralytics.nn.modules.picopose.PicoPoseNoBox`；`parse_model` 也按模块 globals 解析类名 |
| `ultralytics/utils/pico_loss.py` | 2 行 re-export | `PoseModel.init_criterion` 运行时从这里导入 |
| `ultralytics/nn/tasks.py` | 4 处 `# EXT:pico_pose` | `parse_model` 的 head 集合 / `legacy` 赋值 / `init_criterion` 是硬编码的，只能在此接入 |
| `ultralytics/nn/modules/__init__.py` | 1 行 import | 让 `from ultralytics.nn.modules import PicoPoseNoBox` 继续可用 |
| `pyproject.toml` | `packages.find.include` 加 `ext*` | 让 `pip install -e .` 同时映射 `ext`，否则在仓库外 `import ultralytics` 会找不到 `ext` |

安装（改了 `ext/` 结构后需要重跑一次）：

```bash
cd /home/wbk/ultralytics
pip install -e . --no-deps --no-build-isolation
```

导入时请用 `from ext.pico_pose import PicoPoseNoBox`；不要把
`import ext.pico_pose.head` 作为进程第一个相关导入（会和上游 shim 循环）。


新增自定义头时，**不要**再往 `ultralytics/cfg/models/` 或 `ultralytics/utils/` 里塞文件；
按同样方式在 `ext/<feature>/` 建目录，上游只留 shim + 带 `# EXT:` 标记的接入行。

## 输出通道契约（部署端必须对齐）

设 `nc` 为类别数，`nkpt` 为关键点数，`ndim` 为每点维度（2 或 3）。

| 阶段 | 返回 | 通道数 |
| --- | --- | --- |
| 训练 | `dict(scores, kpts, feats)` | `nc` / `nkpt*ndim` |
| 验证 | `Tensor`（伪框 + 分数 + 解码关键点） | `4 + nc + nkpt*ndim` |
| 导出（`export_xy_only=True`，默认） | `Tensor`（分数 sigmoid + 原始 xy） | `nc + 2*nkpt` |
| 导出（`export_xy_only=False`） | `Tensor`（分数 sigmoid + 解码关键点） | `nc + nkpt*ndim` |

⚠️ 目前 `runs/0913poseNoBox.onnx` 是 `nc=1, kpt_shape=(4,3)` 且默认 `export_xy_only=True`，
输出为 `[1, 9, 6300]`（1 + 4×2）。如果部署端要的是 `nc + nkpt*3`（例如 3 类 5 点 = 18），
必须显式设置 `export_xy_only=False`。修改前先更新 `tests/test_export_contract.py`。

`PICO_HEAD_CFG` 支持 `ULTRALYTICS_PICO_HEAD="feat_out=96,use_se=1"` 环境变量；
该变量会改变网络结构，训练时请写进实验 README，避免复现时对不上。

## 运行

```bash
cd /home/wbk/ultralytics

# 方式一：直接用扩展入口（model 默认已指向 ext 里的 yaml）
python -m ext.pico_pose.train \
  data=/home/wbk/train_data/0802_1/windmill_4pt_pose.yaml \
  imgsz=480,640 epochs=300 batch=16 device=0

# 方式二：仍用 ultralytics CLI，显式给 yaml 路径
python -m ultralytics yolo pose train \
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
