"""Shared evaluation utilities for BUSI segmentation (Dice, IoU, HD95 + visualization)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

IMG_SIZE = 256


def load_mask(path):
    """Load a mask as a binary bool array (any value > 0 is tumor), resized to 256x256."""
    m = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return np.array(m) > 0


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2.0 * inter / total   # both empty -> perfect


def iou_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def hd95_score(pred, gt):
    """95th-percentile symmetric Hausdorff distance (in pixels) between mask boundaries."""
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.nan          # undefined; excluded by nanmean (report how many in the notebook)
    pred_b = pred ^ binary_erosion(pred)          # boundary pixels
    gt_b = gt ^ binary_erosion(gt)
    dist_to_gt = distance_transform_edt(~gt_b)    # distance of every pixel to GT boundary
    dist_to_pred = distance_transform_edt(~pred_b)
    d = np.concatenate([dist_to_gt[pred_b], dist_to_pred[gt_b]])
    return float(np.percentile(d, 95))


def evaluate_folder(pred_dir, gt_dir, names, save_csv=None):
    """names: list of mask filenames (e.g. from test.xlsx). Returns per-image DataFrame."""
    rows = []
    for n in names:
        pred = load_mask(Path(pred_dir) / n)
        gt = load_mask(Path(gt_dir) / n)
        rows.append({"name": n,
                     "dice": dice_score(pred, gt),
                     "iou": iou_score(pred, gt),
                     "hd95": hd95_score(pred, gt)})
    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(save_csv, index=False)
    print(f"Dice : {df.dice.mean():.4f} ± {df.dice.std():.4f}")
    print(f"IoU  : {df.iou.mean():.4f} ± {df.iou.std():.4f}")
    print(f"HD95 : {df.hd95.mean():.2f} ± {df.hd95.std():.2f} px "
          f"({df.hd95.isna().sum()} images with undefined HD95 skipped)")
    return df


def show_results(img_dir, gt_dir, pred_dir, names, start=0, count=8, save_path=None):
    """Rows = images; columns = image | ground truth | prediction | overlay.
    Overlay: green = GT boundary region, red = prediction, yellow = overlap."""
    names = names[start:start + count]
    fig, axes = plt.subplots(len(names), 4, figsize=(12, 3 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        img = np.array(Image.open(Path(img_dir) / n).convert("RGB")
                       .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        gt = load_mask(Path(gt_dir) / n)
        pred = load_mask(Path(pred_dir) / n)

        overlay = img.copy().astype(float)
        tp, fp, fn = gt & pred, pred & ~gt, gt & ~pred   # TP=yellow, FP(over-seg)=red, FN(under-seg)=green
        for mask, color in ((tp, (255, 255, 0)), (fp, (255, 0, 0)), (fn, (0, 255, 0))):
            overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array(color)

        d = dice_score(pred, gt)
        for c, (im, title, cmap) in enumerate([(img, n, None), (gt, "Ground truth", "gray"),
                                               (pred, f"Prediction (Dice {d:.2f})", "gray"),
                                               (overlay.astype(np.uint8), "Overlay", None)]):
            axes[r, c].imshow(im, cmap=cmap)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()