#!/usr/bin/env python3
"""Audit sentry (0/7) labels with a tiny armour-glyph classifier.

The sentry plate carries a shield-shaped emblem, while hero/engineer/3/4 plates
carry the digit 1/2/3/4.  A 5-class CNN (sentry, 1, 2, 3, 4) is trained on
area>=150 crops at 480x640, with per-crop gamma normalisation so dark drone
frames remain usable.  A sentry label whose prediction is confidently 2/3/4 is
a wrong label (the plate shows a normal infantry/hero digit).

Usage:
    python scripts/audit_sentry_digit.py --out runs/audit/sentry_digit.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/wbk/ultralytics_2027")
HW = (480, 640)
# scheme class -> glyph class: sentry=0, hero/1, engineer/2, 3, 4
GLYPH = {0: 0, 7: 0, 1: 1, 8: 1, 2: 2, 9: 2, 3: 3, 10: 3, 4: 4, 11: 4}
GLYPH_NAMES = ["sentry", "1", "2", "3", "4"]
SRC_NAMES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 7: 0, 8: 1, 9: 2, 10: 3, 11: 4}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(ROOT / "datasets/schemeC_split_bal"))
    p.add_argument("--out", default=str(ROOT / "runs/audit/sentry_digit.csv"))
    p.add_argument("--cache", default=str(ROOT / "runs/audit/sentry_glyph_cache.npz"))
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--min-area", type=float, default=150.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def gamma_to(gray: np.ndarray, target: float = 80.0) -> np.ndarray:
    m = float(gray.mean())
    if m <= 1:
        return gray
    g = math.log(target / 255.0) / math.log(max(m, 1.0) / 255.0)
    g = float(np.clip(g, 0.30, 3.0))
    return np.clip((gray.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)


def crop_box(img: Image.Image, cx, cy, bw, bh, pad=0.10):
    w0, h0 = img.size
    x1 = int((cx - bw / 2) * w0); x2 = int((cx + bw / 2) * w0) + 1
    y1 = int((cy - bh / 2) * h0); y2 = int((cy + bh / 2) * h0) + 1
    px = int((x2 - x1) * pad); py = int((y2 - y1) * pad)
    return max(0, x1 - px), max(0, y1 - py), min(w0, x2 + px), min(h0, y2 + py)


def build_cache(args):
    split = Path(args.split)
    rows = list(csv.DictReader((split / "split_manifest.tsv").open(), delimiter="\t"))
    by_img = defaultdict(list)
    for r in rows:
        by_img[(r["split"], r["stem"])].append(r)
    xs, ys, meta = [], [], []
    for (sp, stem), rs in by_img.items():
        img_path = split / "images" / sp / f"{stem}.jpg"
        if not img_path.is_file():
            cand = list((split / "images" / sp).glob(f"{stem}.*"))
            if not cand:
                continue
            img_path = cand[0]
        with Image.open(img_path) as im:
            w0, h0 = im.size
            s = min(HW[0] / h0, HW[1] / w0)
            for r in rs:
                for line in (split / "labels" / sp / r["label"]).read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    c = int(p[0])
                    g = GLYPH.get(c)
                    if g is None:
                        continue
                    cx, cy, bw, bh = map(float, p[1:5])
                    area = bw * w0 * s * bh * h0 * s
                    if area < args.min_area:
                        continue
                    x1, y1, x2, y2 = crop_box(im, cx, cy, bw, bh)
                    if x2 - x1 < 3 or y2 - y1 < 3:
                        continue
                    crop = im.crop((x1, y1, x2, y2)).convert("L")
                    crop = crop.resize((args.size, args.size), Image.BILINEAR)
                    arr = np.asarray(crop, dtype=np.uint8)
                    arr = gamma_to(arr)
                    xs.append(arr); ys.append(g)
                    meta.append((sp, stem, r["source"], r["group"], r["root"], c, area))
    x = np.stack(xs)[:, None, :, :].astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64)
    np.savez_compressed(args.cache, x=x, y=y,
                        meta=np.asarray(meta, dtype=object))
    print(f"cache {x.shape} classes={Counter(y.tolist())}")
    return x, y, meta


class Aug(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        # x uint8/float [B,1,H,W] 0..255
        out = []
        for t in x:
            a = t[0].numpy().astype(np.uint8)
            if random.random() < 0.8:
                a = gamma_to(a, random.uniform(40, 120))
            if random.random() < 0.5:
                a = np.clip(a.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-15, 15), 0, 255).astype(np.uint8)
            if random.random() < 0.4:
                ang = random.uniform(-10, 10); sc = random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, sc)
                a = cv2.warpAffine(a, M, (self.size, self.size), borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.15:
                a = cv2.GaussianBlur(a, (3, 3), 0)
            out.append(torch.from_numpy(a).float() / 255.0)
        return torch.stack(out)[:, None]


class GlyphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Linear(128, len(GLYPH_NAMES))

    def forward(self, x):
        return self.h(self.f(x).flatten(1))


class DS(Dataset):
    def __init__(self, x, y, train):
        self.x, self.y, self.train = x, y, train
        self.aug = Aug(x.shape[-1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = torch.from_numpy(self.x[i]).float()
        if self.train:
            t = self.aug(t[None])[0]
        else:
            t = t / 255.0
        return t, int(self.y[i])


def main():
    a = parse()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cache = Path(a.cache)
    if a.rebuild or not cache.is_file():
        x, y, meta = build_cache(a)
    else:
        z = np.load(cache, allow_pickle=True)
        x, y, meta = z["x"], z["y"], list(z["meta"])
        print(f"cache {x.shape} classes={Counter(y.tolist())}")
    # group split by root, force the known bad group into val
    roots = sorted({m[4] for m in meta})
    rng = random.Random(a.seed); rng.shuffle(roots)
    force_val = {r for r in roots if str(r).startswith("drone_20260801_05_01")}
    n_val = max(1, int(round(len(roots) * 0.15)))
    val_roots = set(force_val)
    for r in roots:
        if len(val_roots) >= n_val:
            break
        val_roots.add(r)
    val_idx = [i for i, m in enumerate(meta) if m[4] in val_roots]
    tr_idx = [i for i, m in enumerate(meta) if m[4] not in val_roots]
    print(f"roots={len(roots)} val_roots={len(val_roots)} train={len(tr_idx)} val={len(val_idx)}")
    tr = DS(x[tr_idx], y[tr_idx], True); va = DS(x[val_idx], y[val_idx], False)
    tl = DataLoader(tr, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)
    model = GlyphNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cnt = Counter(y[tr_idx].tolist())
    w = torch.tensor([len(tr_idx) / (len(cnt) * cnt[i]) for i in range(len(GLYPH_NAMES))], dtype=torch.float32)
    print("class weights", w.tolist(), "train counts", dict(cnt))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    for ep in range(a.epochs):
        model.train(); tot = 0.0; n = 0
        for xb, yb in tl:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        sched.step()
        model.eval(); correct = 0; totn = 0; per = defaultdict(lambda: [0, 0])
        sent_tp = sent_n = 0
        for xb, yb in vl:
            with torch.no_grad():
                pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item(); totn += len(yb)
            for p, t in zip(pred.tolist(), yb.tolist()):
                per[t][1] += 1; per[t][0] += int(p == t)
        acc = correct / max(1, totn)
        print(f"ep{ep:02d} loss={tot/max(1,n):.4f} val_acc={acc:.4f} " +
              " ".join(f"{GLYPH_NAMES[k]}:{v[0]}/{v[1]}" for k, v in sorted(per.items())))
    # predict all sentry crops
    model.eval()
    sent = [i for i, m in enumerate(meta) if m[5] in (0, 7)]
    scores = []
    for start in range(0, len(sent), 256):
        idx = sent[start:start + 256]
        xb = torch.from_numpy(x[idx]).float() / 255.0
        with torch.no_grad():
            logits = model(xb)
            prob = logits.softmax(1).numpy()
        for j, i in enumerate(idx):
            m = meta[i]
            scores.append((m[0], m[1], m[2], m[3], m[4], m[5], m[6], *prob[j].tolist()))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["split", "stem", "source", "group", "root", "scheme_cls", "area480",
                      "p_sentry", "p1", "p2", "p3", "p4"])
        wtr.writerows(scores)
    print(f"wrote {out} rows={len(scores)}")
    # summary of suspicious sentry labels
    susp = [s for s in scores if s[7] < 0.25 and max(s[8:]) > 0.55]
    print(f"suspicious (p_sentry<.25 & max digit>.55): {len(susp)}")
    for s in sorted(susp, key=lambda z: max(z[8:]), reverse=True)[:20]:
        print(f"  {s[1]:<70} area={s[6]:.0f} p_sent={s[7]:.3f} p1={s[8]:.2f} p2={s[9]:.2f} p3={s[10]:.2f} p4={s[11]:.2f}")


if __name__ == "__main__":
    main()
