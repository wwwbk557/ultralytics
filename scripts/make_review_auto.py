#!/usr/bin/env python3
"""Build a review gallery for the auto-labelled own_kicked dataset.

Thumbnails show model predictions in green and the old (16-class) GT in red.
Entries default to the predicted objects; images with no prediction fall back to
the old GT (converted to 14 classes) and are marked as fallback.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ext.armor_pose.ref_infer import TARGET_NAMES

OLD16_TO_14 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13}
P_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
BOX_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 255), (0, 255, 128)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def brighten(img: np.ndarray) -> np.ndarray:
    table = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)], dtype=np.uint8)
    img = cv2.LUT(img, table)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def put_text(img, text, org, color=(255, 255, 255), scale=0.55, thick=1):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base), (15, 15, 15), -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, color, thick, cv2.LINE_AA)


def fit_scale(text, max_w, start=0.55, min_scale=0.30, thick=1):
    fs = start
    while fs > min_scale:
        (tw, _), _ = cv2.getTextSize(text, FONT, fs, thick)
        if tw <= max_w:
            return fs
        fs -= 0.02
    return min_scale


def draw_points(img, pts, scale=0.5, color=(220, 220, 220)):
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pt), 4, P_COLORS[i], -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (pt[0] + 6, pt[1] - 6), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (pt[0] + 6, pt[1] - 6), FONT, scale, P_COLORS[i], 1, cv2.LINE_AA)


def read_old_gt(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 13:
            continue
        old = int(p[0])
        if old not in OLD16_TO_14:
            continue
        out.append({
            "cls": OLD16_TO_14[old],
            "kpts": np.array([[float(p[5 + 2 * i]), float(p[6 + 2 * i])] for i in range(4)]),
        })
    return out


def read_label(path: Path, scores: list[float]) -> list[dict]:
    out = []
    if not path.exists():
        return out
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        p = line.split()
        if len(p) != 13:
            continue
        cls = int(p[0])
        pts = np.array([[float(p[5 + 2 * k]), float(p[6 + 2 * k])] for k in range(4)])
        out.append({"cls": cls, "kpts": pts, "score": scores[i] if i < len(scores) else None})
    return out


def make_zoom(img0, kpts_px, label, color, path: Path, zoom=240):
    h, w = img0.shape[:2]
    x1, y1 = kpts_px[:, 0].min(), kpts_px[:, 1].min()
    x2, y2 = kpts_px[:, 0].max(), kpts_px[:, 1].max()
    side = max(int(max(x2 - x1, y2 - y1) * 2.6), 64)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - w), max(0, y0 + side - h)
    crop = img0[max(0, y0):min(h, y0 + side), max(0, x0):min(w, x0 + side)]
    if crop.size == 0:
        return None
    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    crop = cv2.resize(crop, (zoom, zoom), interpolation=cv2.INTER_CUBIC)
    s = zoom / side

    def mp(p):
        return (int(round((p[0] - x0) * s)), int(round((p[1] - y0) * s)))

    pts = [mp(p) for p in kpts_px]
    b1, b2 = mp((x1, y1)), mp((x2, y2))
    cv2.rectangle(crop, b1, b2, color, 2)
    draw_points(crop, pts, scale=0.7)
    fs = fit_scale(label, zoom - 8, start=0.55, min_scale=0.32)
    put_text(crop, label, (4, 18), color, fs, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def render(task):
    cv2.setNumThreads(1)
    src_img, src_gt, lab_path, thumb_path, zoom_dir, max_side, npred, ngt, fallback, scores = task
    img0 = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if img0 is None:
        return None
    h, w = img0.shape[:2]
    display = brighten(img0)
    gts = read_old_gt(src_gt) if ngt else []
    labs = read_label(lab_path, scores)

    objs_src = "gt" if fallback == "gt" else "pred"
    objs = []
    for oi, o in enumerate(labs):
        pts_px = o["kpts"] * [w, h]
        name = TARGET_NAMES[o["cls"]] if 0 <= o["cls"] < len(TARGET_NAMES) else str(o["cls"])
        color = BOX_COLORS[oi % len(BOX_COLORS)] if objs_src == "pred" else (0, 0, 255)
        tag = f"PRED {o['score']:.2f}" if o.get("score") is not None else "GT"
        label = f"#{oi + 1} {name}({o['cls']}) {tag}"
        zoom_path = zoom_dir / (src_img.stem + "_%02d.jpg" % (oi + 1))
        make_zoom(display, pts_px, label, color, zoom_path)
        objs.append({
            "cls": int(o["cls"]), "name": name, "src": objs_src, "score": o.get("score"),
            "zoom": str(zoom_path.relative_to(zoom_dir.parents[1])),
            "bbox": [round(float(pts_px[:, 0].min()), 1), round(float(pts_px[:, 1].min()), 1),
                     round(float(pts_px[:, 0].max()), 1), round(float(pts_px[:, 1].max()), 1)],
            "kpts": [[round(float(x), 1), round(float(y), 1)] for x, y in pts_px],
        })

    scale = min(1.0, max_side / max(h, w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    thumb = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    for g in gts:
        pts = (g["kpts"] * [w, h] * scale).astype(int)
        cv2.polylines(thumb, [pts.reshape(-1, 1, 2)], True, (0, 0, 255), 2, cv2.LINE_AA)
        draw_points(thumb, pts, scale=0.45, color=(0, 0, 255))
        put_text(thumb, f"OLD GT {TARGET_NAMES[g['cls']]}", tuple(pts[0] + 4), (0, 0, 255), 0.5, 1)

    for oi, o in enumerate(objs):
        color = BOX_COLORS[oi % len(BOX_COLORS)] if o["src"] == "pred" else (0, 0, 255)
        pts = (np.array(o["kpts"]) * scale).astype(int)
        cv2.polylines(thumb, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        draw_points(thumb, pts, scale=0.45, color=color)
        label = f"#{oi + 1} {o['name']} {'%.2f' % o['score'] if o['score'] is not None else 'GT'}"
        fs = fit_scale(label, max(40, nw - int(pts[:, 0].min()) - 8), start=0.5)
        put_text(thumb, label, (max(2, int(pts[:, 0].min())), max(14, int(pts[:, 1].min()) - 3)), color, fs, 1)

    source = src_img.parent.name
    y = 16
    for text in [f"{source} | {src_img.name}", f"{w}x{h} | pred {npred} | old GT {ngt}" + (" | FALLBACK GT" if fallback == "gt" else "")]:
        fs = fit_scale(text, nw - 12, start=0.55, min_scale=0.3)
        put_text(thumb, text, (6, y), (255, 255, 255), fs, 1)
        (_, th), base = cv2.getTextSize(text, FONT, fs, 1)
        y += th + base + 6
    if not labs:
        txt = "NO PREDICTION" if not gts else "NO PREDICTION - GT shown"
        fs = fit_scale(txt, nw - 20, start=0.8, min_scale=0.45)
        put_text(thumb, txt, (max(6, nw // 2 - 150), nh // 2), (0, 0, 255), fs, 2)

    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        "file": f"{source}/{src_img.name}",
        "thumb": str(thumb_path.relative_to(thumb_path.parents[2])),
        "split": "kicked", "source": source, "width": w, "height": h,
        "n": len(objs), "npred": npred, "ngt": ngt, "fallback": fallback == "gt",
        "objects": objs,
    }


def build_page(out: Path, data: dict) -> None:
    tpl = (ROOT / "review_rm_pose" / "review_offline.html").read_text(encoding="utf-8")
    tpl = tpl.replace("<title>RM Pose 标注审阅</title>", "<title>RM Pose 自动标注审阅 — own_kicked</title>")
    tpl = tpl.replace("<b>RM Pose Review — unchecked + XJTLU</b>", "<b>RM Pose Auto-Label Review — own_kicked (last.pt)</b>")
    tpl = re.sub(
        r'<select id="fSplit">.*?</select>',
        '<select id="fSplit"><option value="">全部来源</option><option value="kicked">kicked</option></select>',
        tpl, count=1, flags=re.S,
    )
    tpl = tpl.replace(
        "删除记录保存在本机浏览器，导出 TSV 后发我。",
        "绿框=模型预测，红框=旧 GT（已转 14 类）；画面已提亮。<br>删除记录保存在本机浏览器，导出 TSV 后发我。",
    )
    tpl = tpl.replace("const KEY = 'rm_pose_review_v1';", "const KEY = 'rm_pose_review_auto14_v1';")
    tpl = tpl.replace(
        '<label><input id="fNeg" type="checkbox"> 负样本</label>',
        '<label><input id="fNeg" type="checkbox"> 负样本</label>\n  <label><input id="fNoPred" type="checkbox"> 仅看无预测</label>',
    )
    tpl = tpl.replace(
        "const neg = $('fNeg').checked;",
        "const neg = $('fNeg').checked;\n  const noPred = $('fNoPred').checked;",
    )
    tpl = tpl.replace(
        "if (neg && e.n !== 0) return false;",
        "if (neg && e.n !== 0) return false;\n    if (noPred && e.npred !== 0) return false;",
    )
    tpl = tpl.replace("$('fNeg').onchange = applyFilter;", "$('fNeg').onchange = applyFilter;\n$('fNoPred').onchange = applyFilter;")
    tpl = tpl.replace("$('fMulti').checked = false; $('fNeg').checked = false;", "$('fMulti').checked = false; $('fNeg').checked = false; $('fNoPred').checked = false;")
    tpl = tpl.replace(
        "return `split: <b>${e.split}</b> · source: <b>${e.source}</b> · ${e.width}×${e.height} · objects: <b>${e.n}</b>`;",
        "return `source: <b>${e.source}</b> · ${e.width}×${e.height} · pred: <b>${e.npred}</b> · old GT: <b>${e.ngt}</b>${e.fallback ? ' · <span style=\"color:var(--yellow)\">FALLBACK GT</span>' : ''}`;",
    )
    tpl = tpl.replace(
        "if (!e.objects.length) return '<div class=\"empty\">NEGATIVE：空标签（负样本）</div>';",
        "const warn = (e.npred === 0 && e.ngt > 0) ? '<div class=\"empty\" style=\"color:var(--yellow)\">模型未检出，列表显示旧 GT（已转 14 类）。请重点复核。</div>' : '';\n  if (!e.objects.length) return warn + '<div class=\"empty\">NEGATIVE：空标签（负样本）</div>';",
    )
    tpl = tpl.replace(
        "return `<div class=\"obj\">\n      <div class=\"objhead\">#${i + 1} ${o.name} (class ${o.cls})</div>",
        "const tag = o.src === 'gt' ? '<span style=\"color:var(--yellow)\">[GT]</span>' : `<span style=\"color:var(--green)\">[PRED ${o.score != null ? o.score.toFixed(2) : ''}]</span>`;\n    return `${warn}<div class=\"obj\">\n      <div class=\"objhead\">#${i + 1} ${o.name} (class ${o.cls}) ${tag}</div>",
    )
    tpl = tpl.replace(
        "return e.objects.map((o, i) => {",
        "return warn + e.objects.map((o, i) => {",
    )
    tpl = tpl.replace(".empty { color:var(--muted); padding:20px; }", ".empty { color:var(--muted); padding:20px; }\n  .warn { color:var(--yellow); }")
    payload = "window.__REVIEW_DATA__ = " + json.dumps(data, ensure_ascii=False) + ";"
    tpl = re.sub(r"window\.__REVIEW_DATA__ = .*?;</script>", payload + "</script>", tpl, count=1, flags=re.S)
    (out / "review.html").write_text(tpl, encoding="utf-8")
    (out / "index.html").write_text(tpl, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "own_kicked_20261025"))
    p.add_argument("--auto", default=str(ROOT / "own_kicked_20261025_auto14"))
    p.add_argument("--out", default=str(ROOT / "review_rm_pose" / "auto_kicked"))
    p.add_argument("--max-side", type=int, default=960)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--page-only", action="store_true", help="rebuild review.html from existing index.json")
    return p.parse_args()


def main():
    a = parse_args()
    src_root, auto_root, out = Path(a.src), Path(a.auto), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)
    (out / "zoom").mkdir(exist_ok=True)
    if a.page_only:
        data = json.loads((out / "index.json").read_text(encoding="utf-8"))
        build_page(out, data)
        print("page rebuilt", out)
        return

    tasks = []
    manifest = auto_root / "manifest.tsv"
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 7 or parts[5] == "unreadable":
            continue
        source, name, npred, ngt, kept, fallback, cls_txt = parts[:7]
        scores = []
        for item in cls_txt.split(","):
            if ":" in item:
                tail = item.rsplit(":", 1)[1]
                try:
                    scores.append(float(tail))
                except ValueError:
                    scores.append(None)
        img = src_root / source / name
        if not img.exists():
            continue
        tasks.append((
            img, img.with_suffix(".txt"), auto_root / "labels" / source / (Path(name).stem + ".txt"),
            out / "thumbs" / source / (Path(name).stem + ".jpg"),
            out / "zoom" / source, a.max_side, int(npred), int(ngt), fallback, scores,
        ))
    if a.limit:
        tasks = tasks[:a.limit]
    print("tasks", len(tasks), flush=True)

    entries = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(render, tasks, chunksize=8)):
            if r is not None:
                entries.append(r)
            if (i + 1) % 200 == 0:
                print("done", i + 1, flush=True)
    entries.sort(key=lambda e: e["file"])
    data = {"names": list(TARGET_NAMES), "entries": entries}
    (out / "index.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    build_page(out, data)
    print("entries", len(entries), "out", out)


if __name__ == "__main__":
    main()
