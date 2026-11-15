# Pico 装甲板姿态模型 — 部署交付包（2026-11-14）

本包用于把 `schemeC_ws123val_scratch250` 的推理部署到另一台机器。包内只有 ONNX 权重（自包含，无需 ultralytics/自定义层）、契约文件、参考解码器和一张黄金样例。参考解码器已在本机用 onnxruntime 1.19.2 / numpy 2.0.2 / OpenCV 4.13 跑通并通过自检。

**读这个包时不需要训练代码，也不需要 PyTorch。**

## 1. 包内容

```
pico_pose_deploy_20261114/
├── README_DEPLOY.md               # 本文
├── model_io.json                  # 机器可读契约（通道、anchor、公式、类别、阈值、限制）
├── weights/
│   ├── schemeC_ws123val_scratch250_20261114_best_fp16.onnx   # 推荐部署权重
│   ├── schemeC_ws123val_scratch250_20261114_last_fp16.onnx   # 最后一轮，黄金样例上与 best 完全一致
│   └── md5sums.txt
└── reference/
    ├── decode.py                  # 参考解码器（onnxruntime + numpy + cv2，无 torch）
    ├── selfcheck.py               # 自检：md5 + 输入输出 + 黄金样例复现
    ├── expected_detections.json   # 黄金样例期望输出
    ├── sample.jpg                 # 黄金样例（1440x1080，3 个 GT）
    ├── sample_label.txt           # 黄金样例的 YOLO-pose 标签
    └── requirements.txt
```

## 2. 快速验证

```bash
pip install -r reference/requirements.txt
cd reference
python selfcheck.py                       # 期望最后打印 PASS
python decode.py --onnx ../weights/schemeC_ws123val_scratch250_20261114_best_fp16.onnx \
    --image sample.jpg --draw /tmp/dets.jpg
```

## 3. 模型 IO

| 项 | 值 |
|---|---|
| 输入 | `images`，`[1, 3, 480, 640]`，float32，NCHW，**RGB**，值域 `[0,1]` |
| 输出 | `output0`，`[1, 6300, 16]`，float32，`[batch, anchor, channel]` |
| opset / IR | 17 / 8 |
| 权重精度 | fp16（输入输出仍是 fp32） |
| batch / 动态轴 | 固定 1 / 无 |
| 大小 / md5 | best `b6db5ef096c4ede416e29c577f33f37b`，last `a704c32502dae755fa32331c34e3f901` |

16 个通道的含义（**没有 objectness，没有 box 分支**）：

| 通道 | 含义 |
|---|---|
| 0:8 | 4 个关键点角的原始偏移 `(dx,dy)×4`，顺序 **TL, TR, BR, BL** |
| 8 | 颜色 logit `z` → `p_red = sigmoid(z)`，`colour_p = [1-p_red, p_red] = [p_blue, p_red]` |
| 9:16 | 7 个角色 logit → `role_p = sigmoid(r)`，7 个**相互独立**的概率（多标签，**不是 softmax**） |

## 4. 预处理（必须逐字复刻）

```python
r = min(480 / h, 640 / w)                       # 原始图 h,w
new_w, new_h = round(w * r), round(h * r)
dw, dh = (640 - new_w) / 2, (480 - new_h) / 2
img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
top, bottom = round(dh - 0.1), round(dh + 0.1)
left, right = round(dw - 0.1), round(dw + 0.1)
img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blob = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
```

要点：先 resize（保持长宽比、`INTER_LINEAR`）→ 居中补边（值 114）→ BGR 转 RGB → `/255` → CHW → float32。顺序错了（比如先转 RGB 再 resize）会有细微数值差异。

## 5. 后处理（必须逐字复刻）

### 5.1 anchor 网格

三个 level，顺序 stride 8 → 16 → 32，level 内按行优先（`x` 变化最快）：

| level | stride | 特征图 h×w | anchor 数 | 起始下标 |
|---|---|---|---|---|
| P3 | 8 | 60×80 | 4800 | 0 |
| P4 | 16 | 30×40 | 1200 | 4800 |
| P5 | 32 | 15×20 | 300 | 6000 |

每个 cell 的 anchor 中心在网格坐标是 `(x + 0.5, y + 0.5)`，其中 `x = i % w`，`y = i // w`（`i` 是 level 内下标）。

### 5.2 解码

```python
kpts[i, j] = (raw[i, 2*j]   + ax[i],      # j = 0..3，TL,TR,BR,BL
              raw[i, 2*j+1] + ay[i]) * stride[i]      # 单位：480x640 输入图像素

x0, y0 = kpts[..., 0].min(), kpts[..., 1].min()
x1, y1 = kpts[..., 0].max(), kpts[..., 1].max()       # NMS 用的 xyxy 框

p_red    = sigmoid(raw[:, 8])
colour_p = [1 - p_red, p_red]                          # [blue, red]
role_p   = sigmoid(raw[:, 9:16])                       # 7 个独立概率
score[c*7 + r] = colour_p[c] * role_p[r]               # c∈{0,1}, r∈0..6
conf  = max(score)                                     # 14 类取最大
class = argmax(score)
```

### 5.3 NMS 与坐标还原

- 阈值：`conf >= 0.25`，`IoU <= 0.45`，`max_det = 300`（可用 `--conf/--iou/--max-det` 覆盖）。
- NMS：按 conf 降序的**类内**贪心 NMS（等价 ultralytics `non_max_suppression(multi_label=False)`；现有 C++ 节点按类别循环 NMS 也是同一语义）。
- 还原：`x_orig = (x_net - pad_left) / r`，`y_orig = (y_net - pad_top) / r`；最终框裁剪到图像边界，关键点保留在图像外（贴边目标可能为负）。
- 注意：类内 NMS **不会**跨颜色抑制同一个物理装甲板。颜色分支目前不可靠，同一块板可能出现两个颜色不同的框；若下游要求一板一框，需额外按 IoU 去重（例如 IoU>0.6 时保留得分高的）。

## 6. 类别表（14 类 = 2 颜色 × 7 角色，`class = colour_index * 7 + role_index`）

| role_index | role | class (blue) | class (red) |
|---|---|---|---|
| 0 | sentry | 0 blue_sentry | 7 red_sentry |
| 1 | hero | 1 blue_hero | 8 red_hero |
| 2 | engineer | 2 blue_engineer | 9 red_engineer |
| 3 | 3 | 3 blue_3 | 10 red_3 |
| 4 | 4 | 4 blue_4 | 11 red_4 |
| 5 | outpost | 5 blue_outpost | 12 red_outpost |
| 6 | base | 6 blue_base | 13 red_base |

颜色索引 0 = blue，1 = red；角色索引即上表行号。关键点顺序 TL,TR,BR,BL（`flip_idx=[1,0,3,2]`）。

## 7. 已知限制（部署方必须知道）

1. **颜色分支当前不可靠**：跨场景留出集颜色准确率约 54.5–59.4%（旧 v3 模型 99.86%），本模型自身的验证集 mAP50 为 0.547。黄金样例里就有 1 个框角色正确、颜色判错（GT 是 blue_outpost，输出 red_outpost）。角色分类正常：匹配上的检测里角色准确率约 92.8%，主要混淆是 3↔4 和 sentry→3/4。**不要假设颜色可信**；如果下游主要靠颜色区分敌我，需要重新训练颜色分支后再上线。
2. 角色输出是 7 个独立 sigmoid，不要做 softmax，也不要归一化。
3. 没有 objectness 通道，置信度就是 `colour_p × role_p`，不要去找第 9 个通道当 obj。
4. **旧 C++ 解码器（raw18）与本模型不兼容**：`OUTPUT_ATTRIBUTE_COUNT=18`、obj=row[8]、颜色 softmax(row[9:11])、角色 softmax(row[11:18]) 的代码必须改成 16 通道语义，否则解析全错。
5. 输入固定 480×640、batch 固定 1；需要别的尺寸/批量要重新导出。
6. `.pt` 权重不在包内（依赖项目内自定义 head 类，不适合搬机）；ONNX 已自包含。

## 8. 黄金样例

`reference/sample.jpg` + `sample_label.txt` 来自验证集，3 个 GT。参考解码器在 `conf=0.25 / iou=0.45` 下的输出保存在 `expected_detections.json`：

```
blue_4        score=0.606  box=[422.30, 952.42, 503.44, 990.76]
red_outpost   score=0.577  box=[985.82, 0.00, 1127.67, 61.22]   # GT 为 blue_outpost：颜色判错
blue_3        score=0.415  box=[3.02, 515.76, 34.95, 544.55]
```

三个框与 GT 的 IoU 均 > 0.9，框位偏差约 1–2 px。新写的解码器只要预处理/解码正确，这些框应在 0.5 px、分数 0.005 以内复现（`selfcheck.py` 的判定标准）。若框位整体偏移，优先检查 letterbox 的 padding 方向和 `(raw + anchor) * stride` 的 anchor 是否带 `+0.5`。

## 9. 校验和

见 `weights/md5sums.txt`。搬运后先跑：

```bash
cd weights && md5sum -c md5sums.txt
```
