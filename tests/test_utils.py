# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

def test_get_box_metrics_shape_mismatch():
    # 1 batch, 3 anchors, 2 GT boxes, 5 classes
    bs, na, n_max, nc = 1, 3, 2, 5

    # Random predictions
    pd_scores = torch.rand(bs, na, nc)
    pd_bboxes = torch.rand(bs, na, 4)

    # Two ground‑truth boxes with labels 0..nc-1
    gt_labels = torch.tensor([[[0], [1]]], dtype=torch.long)           # shape (1,2,1)
    gt_bboxes = torch.rand(bs, n_max, 4)

    # A mask that doesn’t align: 2 GT × 3 anchors = 6 slots,
    # but we’ll only extract 4 IoUs by picking mask_gt.sum()=4 mismatches
    mask_gt = torch.tensor([[
        [True,  False, True ],   # 2 Trues here
        [True,  True,  False],   # 2 Trues here → total 4
    ]], dtype=torch.bool)        # shape (1,2,3)

    assigner = TaskAlignedAssigner(topk=1, num_classes=nc)
    # Set required internal attributes for get_box_metrics
    assigner.bs = bs
    assigner.n_max_boxes = n_max
    assigner.alpha = 1.0
    assigner.beta = 1.0
    assigner.eps = 1e-7

    # Call get_box_metrics and verify output shapes
    align_metric, overlaps = assigner.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt
    )
    assert align_metric.shape == (bs, n_max, na)
    assert overlaps.shape    == (bs, n_max, na)