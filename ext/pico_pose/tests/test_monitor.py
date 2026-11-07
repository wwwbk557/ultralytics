"""Tests for the validation-time objectness monitor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ext.pico_pose.monitor import AnchorStats, hist_auc


def test_hist_auc_perfect_and_chance():
    pos = np.zeros(1000)
    neg = np.zeros(1000)
    pos[900] = 10.0
    neg[100] = 10.0
    assert hist_auc(pos, neg) == pytest.approx(1.0)
    same = np.ones(1000)
    assert hist_auc(same, same) == pytest.approx(0.5)
    assert np.isnan(hist_auc(np.zeros(1000), neg))


def test_anchor_labels_and_gt_recall():
    st = AnchorStats(thr=0.5)
    # one GT box [0,0,10,10]; anchors 0 and 2 are inside, 1 and 3 are outside
    obj = torch.tensor([0.9, 0.8, 0.1, 0.6])
    centers = torch.tensor([[5.0, 5.0], [50.0, 50.0], [6.0, 5.0], [20.0, 20.0]])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    st.add(obj, centers, boxes)
    assert (st.n_pos, st.n_neg, st.n_amb) == (2, 2, 0)
    assert (st.tp, st.fp) == (1, 2)  # only 0.9 is a positive above 0.5
    res = st.result()
    assert res["obj/P@0.5"] == pytest.approx(1 / 3)
    assert res["obj/R@0.5"] == pytest.approx(0.5)
    assert res["obj/GT_R@0.5"] == pytest.approx(1.0)  # at least one anchor >0.5 inside
    assert 0.0 <= res["obj/AP"] <= 1.0


def test_ambiguous_anchors_are_excluded():
    st = AnchorStats(thr=0.5)
    obj = torch.tensor([0.9, 0.9, 0.9])
    centers = torch.tensor([[1.0, 1.0], [9.0, 9.0], [15.0, 15.0]])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [8.0, 8.0, 20.0, 20.0]])
    st.add(obj, centers, boxes)
    # anchor 0 only in box0, anchor 1 in both -> ambiguous, anchor 2 only in box1
    assert (st.n_pos, st.n_amb, st.n_neg) == (2, 1, 0)
    assert st.gt_total == 2 and st.gt_hit == 2


def test_empty_boxes_are_all_negatives():
    st = AnchorStats()
    st.add(torch.tensor([0.2, 0.9]), torch.tensor([[1.0, 1.0], [2.0, 2.0]]), torch.zeros((0, 4)))
    assert st.n_neg == 2 and st.n_pos == 0 and st.gt_total == 0
    assert st.result()["obj/GT_R@0.5"] == 0.0


def test_monitor_hooks_head_and_reads_anchors():
    """End-to-end plumbing: head hook -> validator.update_metrics -> AnchorStats."""
    from types import SimpleNamespace

    from ultralytics.nn.tasks import PoseModel

    from ext.pico_pose.monitor import ObjectnessMonitor

    cfg = "ext/pico_pose/cfg/yolo11-pose-pico-compact.yaml"
    model = PoseModel(cfg, nc=14, data_kpt_shape=(4, 2), verbose=False).eval()
    trainer = SimpleNamespace(
        ema=SimpleNamespace(ema=model),
        model=model,
        validator=SimpleNamespace(update_metrics=lambda preds, batch: None),
    )
    mon = ObjectnessMonitor(trainer)
    mon.attach()
    img = torch.rand(1, 3, 480, 640)
    with torch.no_grad():
        model(img)
    batch = {
        "img": img,
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        "cls": torch.tensor([[3.0]]),
        "batch_idx": torch.tensor([0]),
    }
    trainer.validator.update_metrics(None, batch)
    mon.detach()
    assert mon.stats.n_pos > 0, "anchors inside the GT box should be positives"
    res = mon.result()
    for k in ("obj/AP", "obj/P@0.5", "obj/R@0.5", "obj/GT_R@0.5", "val/P@0.5", "val/R@0.5"):
        assert k in res


def test_assignment_summary_buckets():
    from ext.pico_pose.monitor import assignment_summary

    areas = np.array([100.0, 2000.0, 20000.0, 50000.0])  # tiny / small / medium / large
    counts = np.array([1.0, 4.0, 8.0, 0.0])
    out = assignment_summary(areas, counts, topk=10)
    assert out["assign/tiny_n"] == 1 and out["assign/tiny_med"] == 1.0
    assert out["assign/small_n"] == 1 and out["assign/small_med"] == 4.0
    assert out["assign/medium_n"] == 1 and out["assign/medium_med"] == 8.0
    assert out["assign/large_n"] == 1 and out["assign/large_med"] == 0.0
    assert out["assign/zero_frac"] == pytest.approx(0.25)
    assert out["assign/mean_ratio"] == pytest.approx(13 / 4 / 10)


def test_assignment_summary_empty():
    from ext.pico_pose.monitor import assignment_summary

    out = assignment_summary(np.zeros(0), np.zeros(0), topk=10)
    assert out["assign/tiny_n"] == 0 and np.isnan(out["assign/tiny_med"])
    assert np.isnan(out["assign/zero_frac"])


def test_fixed_conf_pr_matching():
    from types import SimpleNamespace

    from ext.pico_pose.monitor import ObjectnessMonitor

    mon = ObjectnessMonitor(SimpleNamespace())
    gt = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    gc = torch.tensor([3])
    det = {
        "bboxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 140.0, 140.0]]),
        "conf": torch.tensor([0.9, 0.6]),
        "cls": torch.tensor([3.0, 5.0]),
    }
    mon.n_gt = 1
    mon._match_fixed_conf(det, gt, gc)
    res = mon.result()
    assert res["val/P@0.5"] == pytest.approx(0.5)  # 1 TP + 1 FP
    assert res["val/R@0.5"] == pytest.approx(1.0)
    assert res["val/P@0.8"] == pytest.approx(1.0)  # only the 0.9 detection passes
    assert res["val/R@0.8"] == pytest.approx(1.0)
