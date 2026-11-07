"""Live training dashboard for a running Pico pose run - one chart per metric.

Every refresh rewrites a small PNG per metric inside ``<run>/live/`` plus an
``index.html`` dashboard that refreshes those images in place (no page reload,
no scroll jump). A status banner (``status.png``) carries epoch / P / R / mAP /
ETA, so nothing has to be overlaid on the curves.

Usage:
    python scripts/live_plot.py                      # newest run under runs/
    python scripts/live_plot.py --run runs/pose/schemeC_480x640_obj
    python scripts/live_plot.py --interval 20        # keep watching
    python scripts/live_plot.py --once               # render once
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TRAIN_KEYS = [
    "box_loss",
    "pose_loss",
    "kobj_loss",
    "cls_loss",
    "dfl_loss",
    "color_loss",
    "number_loss",
    "quality_loss",
    "geom_loss",
]
METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(P)",
    "metrics/recall(P)",
    "metrics/mAP50(P)",
    "metrics/mAP50-95(P)",
]
MONITOR_KEYS = [
    "obj/AP",
    "obj/P@0.5",
    "obj/R@0.5",
    "obj/GT_R@0.5",
    "mon/s_auc",
    "mon/s_gap",
    "mon/attr_agree",
]
VAL_PR_KEYS = ["val/P@0.5", "val/R@0.5", "val/P@0.8", "val/R@0.8"]
ASSIGN_KEYS = [
    "assign/tiny_med",
    "assign/tiny_iqr",
    "assign/small_med",
    "assign/small_iqr",
    "assign/medium_med",
    "assign/medium_iqr",
    "assign/large_med",
    "assign/large_iqr",
    "assign/zero_frac",
    "assign/mean_ratio",
]
SCHEDULE_KEYS = ["lr/pg0", "lr/pg1", "lr/pg2"]

PALETTE = ["#4c9be8", "#e8a33d", "#5fbf77", "#e05d5d", "#b07cc6", "#8d6e63", "#d98cb3", "#7f8c8d", "#95a5a6"]

# One-line explanations shown under each card (Chinese, browser side only).
DESC = {
    "obj/AP": "anchor 级 objectness AP：中心落在单个 GT 框内=正、框外=负、多框=歧义剔除；衡量 object/background 分离度",
    "obj/P@0.5": "obj>0.5 的 anchor 中真正是正样本的比例",
    "obj/R@0.5": "正样本 anchor 中 obj>0.5 的比例",
    "obj/GT_R@0.5": "至少有一个 obj>0.5 anchor 落在框内的 GT 比例 = 导出后候选集召回上限",
    "mon/s_auc": "TAL 正/负样本的 s=max sigmoid(cv_joint) 直方图 AUC；0.5=随机，越接近 1 越好",
    "mon/s_gap": "s 的正样本均值 − 负样本均值",
    "mon/attr_agree": "TAL 正样本上 argmax(cv_joint) 与 argmax(colour)×7+argmax(role) 的一致率",
    "assign/tiny_med": "tiny(<32² px) 每 GT 分到的 anchor 数中位数（topk=10）",
    "assign/tiny_iqr": "tiny 桶 q75−q25，看分配是否稳定",
    "assign/small_med": "small(32²–96² px) 每 GT 分到 anchor 数中位数",
    "assign/small_iqr": "small 桶 q75−q25",
    "assign/medium_med": "medium(96²–192² px) 每 GT 分到 anchor 数中位数",
    "assign/medium_iqr": "medium 桶 q75−q25",
    "assign/large_med": "large(≥192² px) 每 GT 分到 anchor 数中位数",
    "assign/large_iqr": "large 桶 q75−q25",
    "assign/zero_frac": "一个 anchor 都没分到的 GT 比例（分配饥饿）",
    "assign/mean_ratio": "平均每 GT anchor 数 / topk",
    "val/P@0.5": "val 上固定 conf≥0.5 的 precision（部署口径，不再是 max-F1 点）",
    "val/R@0.5": "val 上固定 conf≥0.5 的 recall",
    "val/P@0.8": "val 上固定 conf≥0.8 的 precision",
    "val/R@0.8": "val 上固定 conf≥0.8 的 recall",
    "metrics/precision(B)": "Box 精确率（max-F1 阈值）",
    "metrics/recall(B)": "Box 召回率",
    "metrics/mAP50(B)": "Box mAP@0.5",
    "metrics/mAP50-95(B)": "Box mAP@0.5:0.95",
    "metrics/mAP50(P)": "Pose mAP@0.5",
    "metrics/mAP50-95(P)": "Pose mAP@0.5:0.95",
    "train/quality_loss": "通道 8 objectness 的 BCE（全 anchor，带 pos_weight）",
    "train/cls_loss": "cv_joint 14 类 BCE（全 anchor，TAL 软目标；只用于训练/TAL）",
    "train/color_loss": "颜色 2 类交叉熵（TAL 正样本）",
    "train/number_loss": "角色 7 类交叉熵（TAL 正样本）",
}

SECTION_HINTS = {
    "Objectness / TAL monitors": "obj/* 在 val 集上算（候选集质量），mon/* 在训练 batch 上算（TAL 排序与两条分类路径一致性）",
    "Val P/R at fixed confidence": "固定 conf 下、按 IoU 0.5 类内贪心匹配的 val precision/recall；看固定工作点是否退化，不受日志 max-F1 阈值漂移影响",
    "Assignment (anchors per GT)": "每个 GT 实际分到多少正样本 anchor，按面积分桶；topk=10，median <5 或 zero_frac >10% 即分配饥饿",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live per-metric dashboard from a YOLO results.csv")
    p.add_argument("--run", default=None, help="run dir (default: newest results.csv under runs/)")
    p.add_argument("--interval", type=float, default=20.0, help="refresh seconds")
    p.add_argument("--once", action="store_true", help="render once and exit")
    p.add_argument("--keep-zero", action="store_true", help="also plot columns that are always 0")
    return p.parse_args()


def find_latest_run() -> Path | None:
    files = list((ROOT / "runs").rglob("results.csv"))
    return max(files, key=lambda p: p.stat().st_mtime).parent if files else None


def read_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    try:
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                try:
                    int(float(row["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(row)
    except FileNotFoundError:
        return {}
    data = {}
    for c in (rows[0].keys() if rows else []):
        try:
            data[c] = np.array([float(r[c]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            continue
    return data


def _epoch_durations(t: np.ndarray) -> np.ndarray:
    """Positive per-epoch deltas; the negative jump at a --resume boundary is dropped."""
    d = np.diff(np.asarray(t, dtype=float))
    return d[d > 0]


def eta_text(epoch_now: int, total: int, t: np.ndarray) -> str:
    d = _epoch_durations(t)
    if not len(d):
        return "-"
    per = float(np.median(d[-20:]))
    h, m = divmod(int((total - epoch_now) * per), 3600)
    return f"{h}h{m // 60:02d}m"


def safe_name(key: str) -> str:
    return key.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def render(run: Path, keep_zero: bool = False, refresh_s: float = 20.0) -> str:
    run.mkdir(parents=True, exist_ok=True)
    out = run / "live"
    out.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = read_csv(run / "results.csv")
    if not data:
        (out / "index.html").write_text(
            "<h1 style='color:#ddd;background:#111;font-family:sans-serif'>waiting for results.csv ...</h1>"
        )
        return f"waiting for {run.name}/results.csv"

    ep = data["epoch"]
    total = 0
    args_yaml = run / "args.yaml"
    if args_yaml.is_file():
        for line in args_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("epochs:"):
                total = int(float(line.split(":", 1)[1]))
                break
    total = total or int(ep.max())

    written: set[str] = set()

    def draw(key: str, color: str, best_ep: int | None = None) -> str | None:
        if key not in data:
            return None
        y = data[key]
        finite = np.isfinite(y)
        if not finite.any():
            # no data yet (e.g. area bucket empty): keep the card but say so
            fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=100)
            ax.text(0.5, 0.5, "no data yet (bucket empty)", ha="center", va="center", fontsize=10, color="0.45")
            ax.set_axis_off()
            fname = f"{safe_name(key)}.png"
            fig.tight_layout()
            fig.savefig(out / fname)
            plt.close(fig)
            written.add(fname)
            return fname
        if not keep_zero and np.allclose(y[finite], 0):
            return None  # columns that are always 0 (kobj/geom) are noise
        fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=100)
        ax.plot(ep, y, marker=".", ms=3, lw=1.4, color=color)
        if best_ep is not None:
            ax.axvline(best_ep, ls="--", lw=1, color="0.6")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fname = f"{safe_name(key)}.png"
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)
        written.add(fname)
        return fname

    best_i = int(np.nanargmax(data["metrics/mAP50(B)"])) if "metrics/mAP50(B)" in data else -1
    best_ep = int(ep[best_i]) if best_i >= 0 else None

    train_cards = [(k, draw(f"train/{k}", PALETTE[i % len(PALETTE)])) for i, k in enumerate(TRAIN_KEYS)]
    val_cards = [(k, draw(f"val/{k}", PALETTE[i % len(PALETTE)])) for i, k in enumerate(TRAIN_KEYS)]
    metric_cards = [
        (k.split("/", 1)[1], draw(k, PALETTE[i % len(PALETTE)], best_ep)) for i, k in enumerate(METRIC_KEYS)
    ]
    monitor_cards = [(k, draw(k, PALETTE[i % len(PALETTE)])) for i, k in enumerate(MONITOR_KEYS)]
    valpr_cards = [(k, draw(k, PALETTE[i % len(PALETTE)])) for i, k in enumerate(VAL_PR_KEYS)]
    assign_cards = [(k, draw(k, PALETTE[i % len(PALETTE)])) for i, k in enumerate(ASSIGN_KEYS)]
    schedule_cards = [(k, draw(k, PALETTE[i % len(PALETTE)])) for i, k in enumerate(SCHEDULE_KEYS)]

    sections = []
    for title, cards in (
        ("Train loss", train_cards),
        ("Val loss", val_cards),
        ("Metrics (box / pose)", metric_cards),
        ("Val P/R at fixed confidence", valpr_cards),
        ("Objectness / TAL monitors", monitor_cards),
        ("Assignment (anchors per GT)", assign_cards),
        ("Schedule", schedule_cards),
    ):
        kept = [(label, f) for label, f in cards if f]
        if kept:
            sections.append((title, kept))

    # status banner (kept out of the curves)
    status_lines = []
    if "time" in data:
        total_train_s = float(_epoch_durations(data["time"]).sum())
        status_lines.append(
            f"elapsed {total_train_s / 3600:.2f}h  ETA {eta_text(int(ep[-1]), total, data['time'])}"
        )
    short = {
        "metrics/precision(B)": "P(B)",
        "metrics/recall(B)": "R(B)",
        "metrics/mAP50(B)": "mAP50(B)",
        "metrics/mAP50-95(B)": "mAP60-95(B)",
        "metrics/mAP50(P)": "mAP50(P)",
        "metrics/mAP50-95(P)": "mAP50-95(P)",
    }
    short["metrics/mAP50-95(B)"] = "mAP50-95(B)"
    for k, label in short.items():
        if k in data:
            status_lines.append(f"{label} {data[k][-1]:.4f}")
    if best_ep is not None:
        status_lines.append(f"best mAP50(B) {data['metrics/mAP50(B)'][best_i]:.4f}@ep{best_ep}")
    status = (
        f"[{time.strftime('%H:%M:%S')}] {run.name}  ep{int(ep[-1])}/{total}  "
        + "  ".join(status_lines)
    )

    fig = plt.figure(figsize=(16, 0.75), dpi=100)
    fig.patch.set_facecolor("#141414")
    fig.text(0.01, 0.5, status, color="#e8e8e8", fontsize=9.5, family="monospace", va="center")
    fig.savefig(out / "status.png", facecolor="#141414")
    plt.close(fig)

    # drop charts that are no longer produced (e.g. flat columns filtered out)
    for old in out.glob("*.png"):
        if old.name != "status.png" and old.name not in written:
            old.unlink()

    cards_html = []
    for title, cards in sections:
        inner = "".join(
            f"<div class='card'><img data-live src='{fname}' alt='{label}' loading='lazy'>"
            + (f"<div class='desc'>{html.escape(DESC[label])}</div>" if DESC.get(label) else "")
            + "</div>"
            for label, fname in cards
        )
        hint = SECTION_HINTS.get(title, "")
        hint_html = f"<div class='hint'>{html.escape(hint)}</div>" if hint else ""
        cards_html.append(f"<h2>{title}</h2>{hint_html}<div class='grid'>{inner}</div>")

    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>live " + run.name + "</title>"
        "<style>"
        "body{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif}"
        "h1{margin:0;padding:10px 14px 0;font-weight:500;font-size:15px;color:#9aa}"
        ".status img{width:100%;display:block;background:#141414}"
        "h2{margin:18px 14px 6px;font-size:13px;font-weight:600;color:#8fb9e8;text-transform:uppercase;letter-spacing:.04em}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:10px;padding:0 14px}"
        ".card{background:#1a1a1a;border-radius:8px;padding:6px}"
        ".card img{width:100%;height:auto;display:block;border-radius:4px}"
        ".desc{font-size:11px;color:#9aa4b0;padding:5px 4px 3px;line-height:1.3}"
        ".hint{margin:0 14px 4px;font-size:11px;color:#8090a0}"
        "</style>"
        "<h1>" + html.escape(run.name) + " &middot; live (one chart per metric)</h1>"
        "<div class='status'><img data-live src='status.png' alt='status'></div>"
        + "".join(cards_html)
        + "<script>"
        + f"const P={max(int(refresh_s), 5) * 1000};"
        + "setInterval(()=>{document.querySelectorAll('img[data-live]').forEach(im=>{"
        + "const b=im.getAttribute('src').split('?')[0];im.src=b+'?t='+Date.now();});},P);"
        + "</script>",
        encoding="utf-8",
    )
    return status


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve() if args.run else find_latest_run()
    if run is None or not run.is_dir():
        raise SystemExit("no run dir found; pass --run")
    print(f"watching {run}/results.csv -> {run}/live/index.html", flush=True)
    last_sig = None
    while True:
        csv_path = run / "results.csv"
        try:
            sig = (csv_path.stat().st_mtime_ns, csv_path.stat().st_size)
        except FileNotFoundError:
            sig = None
        if sig != last_sig:
            try:
                print(render(run, args.keep_zero, args.interval), flush=True)
                last_sig = sig
            except Exception as e:
                print(f"render error: {e}", flush=True)
        if args.once:
            return
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    main()
