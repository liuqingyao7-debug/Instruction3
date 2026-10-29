"""Compare U-Net, DeepLabV3+ and nnU-Net on the BUSI test set.

Run it from the Instruction3 folder, with the same environment as the notebooks:
    /mnt/cv_data/users/seven/envs/ml/bin/python make_comparison_figures.py

It uses the shared evaluate.py and the saved predicted masks of the three models, and writes to comparison/:
    per_image_metrics_all_models.csv   Dice, IoU and HD95 of every test image for every model
    summary_table.md                   the result tables (copy the numbers into the report)
    easy_cases.png                     images that all models segment well (one benign, one malignant)
    model_differences.png              images where nnU-Net and U-Net differ most
    hard_cases.png                     images where even the best model has the lowest Dice
The images are selected by simple rules (see select_cases), not by hand.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                  # draw to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------
# Paths (adjust the run folders if you trained again)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path("/mnt/cv_data/users/seven/Instruction3")
DATA_ROOT = Path("/mnt/cv_data/users/seven/datasets/Dataset_BUSI_with_GT_Clean")

PRED_DIRS = {
    "U-Net": PROJECT_ROOT / "outputs" / "unet" / "job_383" / "pred_masks",
    "DeepLabV3+": PROJECT_ROOT / "outputs" / "deeplabv3plus" / "job_387" / "pred_masks",
    "nnU-Net": PROJECT_ROOT / "outputs" / "nnunet" / "pred_masks",
}
OUT_DIR = PROJECT_ROOT / "comparison"

sys.path.insert(0, str(PROJECT_ROOT))
import evaluate  # shared metric code, identical for all models  # noqa: E402

IMG_DIR = DATA_ROOT / "images"
GT_DIR = DATA_ROOT / "labels"
MODELS = list(PRED_DIRS)
DIAGONAL = 256 * np.sqrt(2)            # HD95 value used for undefined cases


def collect_metrics(test_names):
    """Per-image Dice, IoU and HD95 of all models in one wide table."""
    wide = pd.DataFrame({"name": test_names})
    for model, pred_dir in PRED_DIRS.items():
        print(f"\n=== {model} ===")
        df = evaluate.evaluate_folder(pred_dir, GT_DIR, test_names)
        for metric in ("dice", "iou", "hd95"):
            wide[f"{metric}_{model}"] = df[metric].values
    return wide


def summary_tables(wide):
    """Markdown tables with the numbers used in the report."""
    n = len(wide)
    rows = []
    for model in MODELS:
        dice, iou, hd = wide[f"dice_{model}"], wide[f"iou_{model}"], wide[f"hd95_{model}"]
        rows.append(
            f"| {model} | {dice.mean():.4f} ± {dice.std():.4f} | {iou.mean():.4f} ± {iou.std():.4f} "
            f"| {hd.mean():.2f} ± {hd.std():.2f} ({hd.isna().sum()} excluded) "
            f"| {hd.fillna(DIAGONAL).mean():.2f} |"
        )
    main = (
        "| Model | Dice | IoU | HD95 (px), undefined excluded | HD95 (px), undefined = 362 |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows)
    )

    rows = []
    for model in MODELS:
        dice, iou = wide[f"dice_{model}"], wide[f"iou_{model}"]
        rows.append(
            f"| {model} | {dice.median():.4f} | {iou.median():.4f} | {(dice >= 0.8).sum()} / {n} "
            f"| {(dice < 0.5).sum()} / {n} | {(dice == 0).sum()} / {n} |"
        )
    extra = (
        "| Model | Median Dice | Median IoU | Dice ≥ 0.8 | Dice < 0.5 | Dice = 0 |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(rows)
    )
    return main, extra


def paired_bootstrap(wide, n_resamples=10000, seed=42):
    """95% interval of the difference in mean Dice between two models (paired by image)."""
    rng = np.random.default_rng(seed)
    n = len(wide)
    lines = []
    for better, other in (("nnU-Net", "DeepLabV3+"), ("nnU-Net", "U-Net"), ("DeepLabV3+", "U-Net")):
        diff = (wide[f"dice_{better}"] - wide[f"dice_{other}"]).values
        means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
        low, high = np.percentile(means, [2.5, 97.5])
        lines.append(
            f"Mean Dice {better} - {other}: {diff.mean():+.4f} "
            f"(95% bootstrap interval {low:+.4f} to {high:+.4f})"
        )
    return lines


def select_cases(wide):
    """Select example images with fixed rules, so that the choice is not cherry-picked."""
    wide = wide.copy()
    wide["min_dice"] = wide[[f"dice_{m}" for m in MODELS]].min(axis=1)
    wide["max_dice"] = wide[[f"dice_{m}" for m in MODELS]].max(axis=1)
    wide["gain"] = wide["dice_nnU-Net"] - wide["dice_U-Net"]

    # Easy cases: highest worst-model Dice, one benign and one malignant image.
    ordered = wide.sort_values("min_dice", ascending=False)
    easy = [ordered[ordered["name"].str.startswith(kind)].iloc[0]["name"] for kind in ("benign", "malignant")]

    # Differences: where nnU-Net gains most over U-Net, and where it loses most (if it does).
    gains = list(wide.nlargest(2, "gain")["name"])
    losses = list(wide[wide["gain"] < 0].nsmallest(1, "gain")["name"])

    # Hard cases: even the best model has the lowest Dice.
    hard = list(wide.nsmallest(3, "max_dice")["name"])
    return {"easy_cases": easy, "model_differences": gains + losses, "hard_cases": hard}


def overlay(img, gt, pred):
    """Yellow = correct, red = false positive, green = missed tumor (same colors as the notebooks)."""
    out = img.astype(float)
    for mask, color in ((gt & pred, (255, 255, 0)), (pred & ~gt, (255, 0, 0)), (gt & ~pred, (0, 255, 0))):
        out[mask] = 0.5 * out[mask] + 0.5 * np.array(color)
    return out.astype(np.uint8)


def draw_figure(names, wide, path):
    """One row per image: image | ground truth | overlay of each model with its Dice."""
    cols = 2 + len(MODELS)
    fig, axes = plt.subplots(len(names), cols, figsize=(3 * cols, 3 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        img = np.array(Image.open(IMG_DIR / name).convert("RGB").resize((256, 256), Image.BILINEAR))
        gt = evaluate.load_mask(GT_DIR / name)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(name, fontsize=9)
        axes[r, 1].imshow(gt, cmap="gray")
        axes[r, 1].set_title("Ground truth", fontsize=9)
        for c, model in enumerate(MODELS, start=2):
            pred = evaluate.load_mask(PRED_DIRS[model] / name)
            dice = float(wide.loc[wide["name"] == name, f"dice_{model}"].iloc[0])
            axes[r, c].imshow(overlay(img, gt, pred))
            axes[r, c].set_title(f"{model} (Dice {dice:.2f})", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    test_names = pd.read_excel(DATA_ROOT / "test.xlsx")["Image"].tolist()

    wide = collect_metrics(test_names)
    wide.to_csv(OUT_DIR / "per_image_metrics_all_models.csv", index=False)

    main_table, extra_table = summary_tables(wide)
    bootstrap = paired_bootstrap(wide)
    text = main_table + "\n\n" + extra_table + "\n\n" + "\n".join(f"- {line}" for line in bootstrap) + "\n"
    (OUT_DIR / "summary_table.md").write_text(text)
    print("\n" + text)

    for figure_name, names in select_cases(wide).items():
        draw_figure(names, wide, OUT_DIR / f"{figure_name}.png")
        print(f"{figure_name}.png: {', '.join(names)}")
    print("\nSaved everything to", OUT_DIR)


if __name__ == "__main__":
    main()
