"""Export the compact Pico pose head (16 channels: 8 keypoints + 1 colour + 7 roles).

The export graph drops ``cv2`` (box) and ``dfl``; deployment rebuilds the NMS box
from the four decoded corners, which is what ``deploy_boxes`` makes validation
do as well. There is no objectness channel, so no calibration offset is applied:
the seven role logits are the exported confidence.

Usage:
    python scripts/export_pico.py --weights runs/pose/.../weights/best.pt --format onnx
    python scripts/export_pico.py --weights ... --format engine --imgsz 480,640 --half
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True, help="trained checkpoint (best.pt / last.pt)")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", default="480,640", help="H,W (the exported graph has no letterbox)")
    p.add_argument("--half", action="store_true")
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--opset", type=int, default=17, help="17 keeps onnxruntime<=1.19 and OpenVINO happy")
    return p.parse_args()


def _toposort_onnx(path) -> None:
    """ONNX's CPU FP16 converter can leave Cast nodes out of topological order.

    ORT is forgiving but onnx.checker / OpenVINO / TensorRT are not.  Reorder the
    graph in place after the export, then re-run the checker.
    """
    try:
        import onnx
    except Exception:
        return
    path = Path(path)
    if path.suffix != ".onnx" or not path.is_file():
        return
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
        return
    except Exception:
        pass
    nodes = list(model.graph.node)
    producer = {o: i for i, n in enumerate(nodes) for o in n.output}
    n = len(nodes)
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for i, node in enumerate(nodes):
        for x in node.input:
            j = producer.get(x)
            if j is not None and j != i:
                adj[j].append(i)
                indeg[i] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    order = []
    while q:
        i = q.pop(0)
        order.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    if len(order) != n:
        print("[export_pico] WARNING: ONNX graph has a cycle; not reordering")
        return
    model.graph.node.clear()
    model.graph.node.extend(nodes[i] for i in order)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[export_pico] ONNX graph topologically sorted: {path}")


def main() -> None:
    a = parse()
    weights = Path(a.weights)
    model = YOLO(str(weights))
    head = model.model.model[-1]
    head.export = True
    h, w = (int(v) for v in a.imgsz.replace("x", ",").split(","))
    print(f"[export_pico] weights={weights} imgsz={(h, w)} export_channels={head.no_export}")
    path = model.export(format=a.format, imgsz=(h, w), half=a.half, simplify=a.simplify, opset=a.opset)
    if a.format == "onnx":
        _toposort_onnx(path)
    print(f"[export_pico] -> {path}")


if __name__ == "__main__":
    main()
