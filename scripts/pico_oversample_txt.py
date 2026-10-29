#!/usr/bin/env python3
"""Build a weak-class oversampled train list for the 14-class Pico pose data.

The Ultralytics ``get_img_files`` reader preserves duplicate lines in a txt
list, so repeating an image is enough to oversample it.  Validation is never
touched; only a new train txt + yaml are written.

Modes:
  all2x: every image containing any weak class is repeated once (2x total).
  out3x: every outpost image is repeated twice (3x total), other weak classes 1x.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

WEAK = (2, 3, 6, 9, 13)
OUTPOST = (6, 13)


def _class_set(label: Path) -> set[int]:
    if not label.is_file():
        return set()
    return {int(float(line.split()[0])) for line in label.read_text().splitlines() if line.strip()}


def _images(root: Path, train: str) -> list[Path]:
    p = Path(train)
    if not p.is_absolute():
        p = root / p
    if p.is_file():
        return [Path(x) if Path(x).is_absolute() else p.parent / x for x in p.read_text().splitlines() if x.strip()]
    return sorted(p.glob("*.*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="source data.yaml")
    ap.add_argument("--mode", choices=("all2x", "out3x"), default="all2x")
    ap.add_argument("--out-yaml", required=True)
    ap.add_argument("--cap", type=float, default=1.5)
    args = ap.parse_args()

    data_path = Path(args.data)
    data = yaml.safe_load(data_path.read_text())
    root = Path(data.get("path", data_path.parent)).resolve()
    imgs = _images(root, str(data["train"]))
    weak = set(WEAK)
    outpost = set(OUTPOST)

    extras: list[Path] = []
    per_class = {c: 0 for c in weak}
    for img in imgs:
        label = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        cls = _class_set(label)
        hit = cls & weak
        for c in hit:
            per_class[c] += 1
        if args.mode == "all2x":
            if hit:
                extras.append(img)
        elif cls & outpost:
            extras.extend([img, img])
        elif hit:
            pass  # out3x: non-outpost weak classes stay 1x to respect the cap

    cap = int(len(imgs) * args.cap)
    if len(imgs) + len(extras) > cap:
        extras = extras[: max(0, cap - len(imgs))]
    lines = [str(p) for p in imgs] + [str(p) for p in extras]
    txt = Path(args.out_yaml).resolve().with_suffix(".txt")
    txt.write_text("\n".join(lines) + "\n")

    out = dict(data)
    out["path"] = str(root)
    out["train"] = str(txt)
    Path(args.out_yaml).write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))

    print(f"mode={args.mode} base={len(imgs)} total={len(lines)} x{len(lines)/len(imgs):.2f}")
    print(f"weak image counts: {per_class}")
    print(f"wrote {txt}\nwrote {args.out_yaml}")


if __name__ == "__main__":
    main()
