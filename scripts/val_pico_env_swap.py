"""Role/colour accuracy under environment changes.

The trained role head can use the scene as a shortcut, which the normal val
split cannot see (train and val share the scene distribution).  This script
re-runs the same weights on single-plate val images under three conditions:

* ``orig``  - untouched image (reference)
* ``local`` - plate pixels pasted on a flat 114-gray canvas (no environment)
* ``swap``  - plate pixels pasted onto a different val image (foreign scene)

All three use the deployment box path (``deploy_boxes=True``).  Matching is by
IoU against the GT corner AABB, so ``match`` (detection under the environment
change) is reported separately from ``role@hit`` (classification given a hit).
``role@all`` includes missed detections and is the end-to-end number.

Usage:
    python scripts/val_pico_env_swap.py --weights runs/pose/.../weights/best.pt --n 400
    python scripts/val_pico_env_swap.py --weights ... --mask rect --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLES = ["sentry", "hero", "engineer", "3", "4", "outpost", "base"]
GRAY = 114


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split_bal/data.yaml"))
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.1, help="pre-NMS confidence (keep low for recall)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--match-iou", type=float, default=0.3, help="IoU needed to count a detection as the target")
    p.add_argument("--mask", choices=("plate", "rect"), default="plate",
                   help="paste only the plate quad, or the whole AABB patch (old ablation protocol)")
    p.add_argument("--margin", type=float, default=0.5, help="AABB margin for --mask rect")
    p.add_argument("--imgsz", default="480,640")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def read_gt(label_path: Path, w: int, h: int):
    """Return ``[(cls, 4x2 corners in pixels)]`` from a YOLO pose label file."""
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        v = line.split()
        cls = int(v[0])
        kp = v[5:13] if len(v) >= 13 else v[1:9]
        pts = np.array([float(t) for t in kp], dtype=np.float32).reshape(4, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        out.append((cls, pts))
    return out


def aabb(pts: np.ndarray) -> np.ndarray:
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def overlaps(a, b, pad: float = 0.0) -> bool:
    return a[0] < b[2] + pad and a[2] > b[0] - pad and a[1] < b[3] + pad and a[3] > b[1] - pad


def cut_patch(img: np.ndarray, pts: np.ndarray, mask_mode: str, margin: float):
    """Return ``(patch, mask, (x, y))`` for one plate."""
    h, w = img.shape[:2]
    b = aabb(pts)
    bw = max(float(b[2] - b[0]), 2.0)
    bh = max(float(b[3] - b[1]), 2.0)
    x0 = max(0, int(b[0] - bw * margin) - 2)
    y0 = max(0, int(b[1] - bh * margin) - 2)
    x1 = min(w, int(np.ceil(b[2] + bw * margin)) + 3)
    y1 = min(h, int(np.ceil(b[3] + bh * margin)) + 3)
    patch = img[y0:y1, x0:x1].copy()
    if mask_mode == "rect":
        mask = np.full(patch.shape[:2], 255, np.uint8)
    else:
        mask = np.zeros(patch.shape[:2], np.uint8)
        poly = np.round(pts - np.array([x0, y0], np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return patch, mask, (x0, y0)


def paste(dst: np.ndarray, patch: np.ndarray, mask: np.ndarray, x: int, y: int) -> bool:
    h, w = dst.shape[:2]
    ph, pw = patch.shape[:2]
    if x < 0 or y < 0 or x + pw > w or y + ph > h:
        return False
    roi = dst[y : y + ph, x : x + pw]
    np.copyto(roi, patch, where=mask[..., None].astype(bool))
    return True


def resolve_val_dir(data_yaml: Path) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val = Path(str(cfg.get("val", "images/val")))
    return val if val.is_absolute() else root / val


def label_dir_for(img_dir: Path) -> Path:
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def preds_from_result(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    return [{"box": xyxy[i], "cls": int(cls[i])} for i in range(len(xyxy))]


def predict_chunked(model, images, batch: int, conf: float, iou: float, imgsz, device):
    out = []
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        out.extend(
            model.predict(
                chunk,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=300,
                device=device,
                batch=len(chunk),
                verbose=False,
                stream=False,
            )
        )
    return out


def pick_target(preds, pts: np.ndarray, thr: float):
    box = aabb(pts)
    best, best_iou = None, thr
    for pred in preds:
        value = iou(box, pred["box"])
        if value > best_iou:
            best, best_iou = pred, value
    return best


def evaluate(images, metas, model, args, shifted: bool = False):
    """Per-condition stats: total, matched, role/colour hits (overall and per role)."""
    imgsz = tuple(int(v) for v in str(args.imgsz).replace("x", ",").split(",") if v.strip())
    results = predict_chunked(model, images, args.batch, args.conf, args.iou, imgsz, args.device)
    overall = {"total": 0, "matched": 0, "role_ok": 0, "color_ok": 0}
    per_role = defaultdict(lambda: {"total": 0, "matched": 0, "role_ok": 0})
    for (cls, pts, offset), result in zip(metas, results):
        role, color = cls % 7, cls // 7
        target = pts + offset if shifted else pts
        hit = pick_target(preds_from_result(result), target, args.match_iou)
        overall["total"] += 1
        per_role[role]["total"] += 1
        if hit is None:
            continue
        overall["matched"] += 1
        overall["role_ok"] += int(hit["cls"] % 7 == role)
        overall["color_ok"] += int(hit["cls"] // 7 == color)
        per_role[role]["matched"] += 1
        per_role[role]["role_ok"] += int(hit["cls"] % 7 == role)
    return overall, dict(per_role)


def ratios(stat: dict) -> dict:
    total = max(stat["total"], 1)
    matched = max(stat["matched"], 1)
    return {
        "total": stat["total"],
        "match": stat["matched"] / total,
        "role_at_hit": stat["role_ok"] / matched,
        "role_end2end": stat["role_ok"] / total,
        "color_at_hit": stat.get("color_ok", 0) / matched,
    }


def main() -> None:
    args = parse()
    rng = np.random.default_rng(args.seed)
    img_dir = resolve_val_dir(Path(args.data))
    lab_dir = label_dir_for(img_dir)
    items = sorted(p for p in img_dir.glob("*.jpg") if (lab_dir / (p.stem + ".txt")).is_file())
    if not items:
        raise SystemExit(f"no labelled val images under {img_dir}")

    order = rng.permutation(len(items))
    targets, images, metas = [], [], []
    originals, locals_, swaps = [], [], []
    skipped = 0
    for idx in order:
        if len(targets) >= args.n:
            break
        path = items[int(idx)]
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        gts = read_gt(lab_dir / (path.stem + ".txt"), w, h)
        if len(gts) != 1:
            continue
        cls, pts = gts[0]
        patch, mask, (px, py) = cut_patch(img, pts, args.mask, args.margin)

        donor = None
        donor_boxes = None
        for _ in range(8):
            candidate = items[int(rng.integers(len(items)))]
            if candidate == path:
                continue
            dimg = cv2.imread(str(candidate))
            if dimg is None:
                continue
            dh, dw = dimg.shape[:2]
            dimg = cv2.resize(dimg, (w, h), interpolation=cv2.INTER_LINEAR)
            boxes = [aabb(p) * np.array([w / dw, h / dh, w / dw, h / dh], np.float32)
                     for _, p in read_gt(lab_dir / (candidate.stem + ".txt"), dw, dh)]
            donor, donor_boxes = dimg, boxes
            break
        if donor is None:
            skipped += 1
            continue

        ph, pw = patch.shape[:2]
        placed = None
        for _ in range(80):
            nx = int(rng.integers(0, max(w - pw, 0) + 1))
            ny = int(rng.integers(0, max(h - ph, 0) + 1))
            box = (nx, ny, nx + pw, ny + ph)
            if not any(overlaps(box, b, pad=2.0) for b in donor_boxes):
                placed = (nx, ny)
                break
        if placed is None:
            skipped += 1
            continue
        nx, ny = placed
        swap = donor.copy()
        paste(swap, patch, mask, nx, ny)
        local = np.full_like(img, GRAY)
        paste(local, patch, mask, px, py)

        originals.append(img)
        locals_.append(local)
        swaps.append(swap)
        targets.append(path)
        metas.append((cls, pts, np.array([nx - px, ny - py], np.float32)))

    if not metas:
        raise SystemExit("no usable single-plate images (all skipped)")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.model.model[-1].deploy_boxes = True

    conditions = {}
    for name, imgs in (("orig", originals), ("local", locals_), ("swap", swaps)):
        overall, per_role = evaluate(imgs, metas, model, args, shifted=(name == "swap"))
        conditions[name] = {"overall": ratios(overall), "per_role": per_role}

    if args.json:
        print(json.dumps({
            "weights": args.weights,
            "data": str(Path(args.data)),
            "n": len(metas),
            "skipped": skipped,
            "conf": args.conf,
            "match_iou": args.match_iou,
            "mask": args.mask,
            "conditions": conditions,
        }))
        return

    print(f"\n{Path(args.weights).parent.name}  n={len(metas)} (skipped {skipped})  "
          f"conf={args.conf}  match_iou={args.match_iou}  mask={args.mask}")
    print(f"  {'condition':10s} {'match':>8s} {'role@hit':>9s} {'role@all':>9s} {'color@hit':>10s}")
    for name in ("orig", "local", "swap"):
        v = conditions[name]["overall"]
        print(f"  {name:10s} {v['match']:8.2%} {v['role_at_hit']:9.2%} "
              f"{v['role_end2end']:9.2%} {v['color_at_hit']:10.2%}")

    print("\n  role@hit per class")
    print(f"  {'role':9s} {'n':>5s} " + " ".join(f"{name:>9s}" for name in ("orig", "local", "swap")))
    for role, name in enumerate(ROLES):
        row = [conditions[c]["per_role"].get(role) for c in ("orig", "local", "swap")]
        n = max((r["total"] if r else 0) for r in row)
        if n == 0:
            continue
        cells = [f"{r['role_ok'] / max(r['matched'], 1):9.2%}" if r and r["matched"] else f"{'-':>9s}" for r in row]
        print(f"  {name:9s} {n:5d} " + " ".join(cells))


if __name__ == "__main__":
    main()
