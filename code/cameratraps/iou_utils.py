"""
iou_utils.py

Box-IoU helpers for Lab 3.

Boxes are 4-tuples (x1, y1, x2, y2) in image pixels.
"""
from typing import Iterable, List, Tuple


def iou(a, b) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def greedy_match(
    pred_boxes: List[Tuple[float, ...]],
    pred_scores: List[float],
    gt_boxes: List[Tuple[float, ...]],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int]:
    """Match predictions to ground truth greedily by descending score.

    Returns (tp, fp, fn).
    """
    order = sorted(range(len(pred_boxes)),
                   key=lambda i: pred_scores[i], reverse=True)
    matched_gt = set()
    tp, fp = 0, 0
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            v = iou(pred_boxes[i], g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def precision_recall_curve(
    pred_boxes_per_image: Iterable[List[Tuple[float, ...]]],
    pred_scores_per_image: Iterable[List[float]],
    gt_boxes_per_image:   Iterable[List[Tuple[float, ...]]],
    thresholds:           Iterable[float],
    iou_thresh: float = 0.5,
):
    """Sweep `thresholds` and return parallel lists (precisions, recalls)."""
    precisions, recalls = [], []
    pred_boxes_per_image = list(pred_boxes_per_image)
    pred_scores_per_image = list(pred_scores_per_image)
    gt_boxes_per_image = list(gt_boxes_per_image)

    for t in thresholds:
        TP = FP = FN = 0
        for boxes, scores, gts in zip(pred_boxes_per_image,
                                      pred_scores_per_image,
                                      gt_boxes_per_image):
            keep = [(b, s) for b, s in zip(boxes, scores) if s >= t]
            kb = [b for b, _ in keep]
            ks = [s for _, s in keep]
            tp, fp, fn = greedy_match(kb, ks, gts, iou_thresh)
            TP += tp; FP += fp; FN += fn
        p = TP / (TP + FP) if (TP + FP) else 0.0
        r = TP / (TP + FN) if (TP + FN) else 0.0
        precisions.append(p); recalls.append(r)
    return precisions, recalls
