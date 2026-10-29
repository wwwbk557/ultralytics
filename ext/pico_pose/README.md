# Compact 四关键点 Pose 头

`ext.pico_pose` 提供 14 类装甲板四关键点 Pose 头，挂在 YOLO11-pose 的 P3/P4/P5 颈部。

## 结构

每个 FPN 层：

| 分支 | 结构 | 用途 |
|---|---|---|
| `cv2` | Conv3x3(64) → Conv3x3(64) → Conv1x1(4×reg_max) | DFL 框分布，训练/验证 |
| `cv_shared` | DWConv3x3 → Conv1x1(32) | 共享特征 |
| `cv_compact` | Conv1x1(32→18) | 8 关键点 + 1 quality + 2 colour + 7 number |
| `cv_joint` | Conv1x1(32→14) | TAL 分类 logits |

`cv3`/`cv4` 不使用。P3 不额外加卷积块。

## 训练

```bash
python -m ext.pico_pose.train_compact \
  --data /path/to/data.yaml \
  --name compact_640 \
  --epochs 250 --batch 32 --imgsz 640 --device 0 \
  --geom 0.0
```

`--geom` 是几何正则总开关：0 关闭，大于 0 启用。几何项由
`--geom-par`（平行四边形）、`--geom-opp`（对边等长）、`--geom-angle`（直角）、
`--geom-axis`（首边水平）加权，全部在尺度归一化后计算。

损失槽位：`box, pose, kobj, cls, dfl, color, number, quality, geom`。
`pose` 使用 OKS 形式的关键点损失；`quality` 是与预测关键点 OKS 重合度做目标的 BCE。

## 导出

`head.export = True` 时只执行 `cv_compact`，输出 `[B, A, 18]`：

```text
0:8    raw keypoint offsets      p = (raw + anchor) * stride
8      quality logit             sigmoid
9:11   colour logits             softmax
11:18  number logits             softmax
score  = sigmoid(quality) * max(softmax(colour)) * max(softmax(number))
class  = argmax(colour) * 7 + argmax(number)
```

## 测试

```bash
python -m pytest ext/pico_pose/tests/test_smoke.py tests/test_pico_compact_p0.py -q
```
