"""Train DeepLabV3+ from scratch on the BUSI dataset (binary tumor segmentation).

All settings are identical to the U-Net experiment: the train/test split from
train.xlsx / test.xlsx, 256 x 256 grayscale input in [0, 1], seed 42, batch size 8,
Adam with learning rate 0.001, 50 epochs, BCE + soft Dice loss, no augmentation.
Only the training set is used; the test set is never touched here.

Usage (from the Instruction3 folder):
    mkdir -p logs
    sbatch --job-name=deeplab submit.sh train_deeplabv3plus.py

Outputs are written to outputs/deeplabv3plus/job_<slurm id>/:
    config.json, training_history.csv, checkpoints/latest.pth, weights/deeplabv3plus_final.pth
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

# ----------------------------------------------------------------------------
# Settings (must stay identical to the U-Net experiment)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path("/mnt/cv_data/users/seven/Instruction3")
DATA_ROOT = Path("/mnt/cv_data/users/seven/datasets/Dataset_BUSI_with_GT_Clean")

MODEL_NAME = "deeplabv3plus"
WEIGHT_NAME = "deeplabv3plus_final.pth"

SEED = 42
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3


# ----------------------------------------------------------------------------
# Data (same preprocessing as the notebook: grayscale, 256 x 256, [0, 1])
# ----------------------------------------------------------------------------
class BUSISegmentationDataset(Dataset):
    def __init__(self, dataframe, root):
        self.names = dataframe["Image"].tolist()
        self.root = Path(root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        with Image.open(self.root / "images" / name) as img:
            original_size = img.size
            img = img.convert("L")
            img = img.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
            image = np.array(img, dtype=np.float32) / 255.0

        with Image.open(self.root / "labels" / name) as label:
            # Validate the original mask before resizing.
            assert label.size == original_size, f"Size mismatch: {name}"
            raw_mask = np.array(label)
            assert raw_mask.ndim == 2, f"Unexpected mask shape: {name}"
            assert set(np.unique(raw_mask)).issubset({0, 1}), f"Unexpected mask values: {name}"

            label = label.resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)
            mask = np.array(label, dtype=np.float32)

        # [height, width] -> [1, height, width]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask, name


# ----------------------------------------------------------------------------
# Model (identical to Section 4 of the notebook)
# ----------------------------------------------------------------------------
def conv_bn_relu(in_ch, out_ch, kernel_size=3, dilation=1):
    """Convolution + BatchNorm + ReLU; padding keeps height and width unchanged."""
    padding = 0 if kernel_size == 1 else dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: parallel branches with different dilation rates."""

    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()

        # One 1x1 branch plus one dilated 3x3 branch for every rate.
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_ch, out_ch, kernel_size=1)]
            + [conv_bn_relu(in_ch, out_ch, kernel_size=3, dilation=r) for r in rates]
        )

        # Global context branch. No BatchNorm: its feature map is 1 x 1, and
        # BatchNorm fails if the last batch contains a single image.
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Fuse the concatenated branches (len(rates) + 2 branches in total).
        self.project = nn.Sequential(
            conv_bn_relu(out_ch * (len(rates) + 2), out_ch, kernel_size=1),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]

        pooled = self.image_pool(x)  # [B, out_ch, 1, 1]
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        features.append(pooled)

        return self.project(torch.cat(features, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a ResNet-50 encoder for single-channel binary segmentation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ResNet-50 without pretrained weights. Dilation in the last stage gives
        # output stride 16 (feature map = 1/16 of the input size).
        # (With an old torchvision, replace `weights=None` by `pretrained=False`.)
        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        # The first convolution is replaced to accept grayscale input.
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256 channels, 1/4 resolution (low-level features)
        self.layer2 = backbone.layer2  # 512 channels, 1/8
        self.layer3 = backbone.layer3  # 1024 channels, 1/16
        self.layer4 = backbone.layer4  # 2048 channels, 1/16 (dilated)

        self.aspp = ASPP(in_ch=2048, out_ch=256)

        # Decoder: reduce the low-level features to 48 channels, then fuse.
        self.low_level = conv_bn_relu(256, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            conv_bn_relu(256 + 48, 256, kernel_size=3),
            conv_bn_relu(256, 256, kernel_size=3),
        )

        # One raw prediction score (logit) per pixel.
        self.classifier = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer4(self.layer3(self.layer2(low)))

        # Multi-scale context
        x = self.aspp(x)

        # Decoder: upsample to 1/4 resolution and fuse with low-level features
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low_level(low)], dim=1))

        # Logits at full input resolution
        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------
# Loss (identical to Section 5 of the notebook): 0.5 * BCE + 0.5 * soft Dice
# ----------------------------------------------------------------------------
bce_loss = nn.BCEWithLogitsLoss()


def segmentation_loss(logits, targets):
    bce = bce_loss(logits, targets)  # pixel-level binary cross-entropy

    probabilities = torch.sigmoid(logits)  # logits -> probabilities

    # Soft Dice computed separately for each image.
    dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dims)
    total = probabilities.sum(dim=dims) + targets.sum(dim=dims)

    smooth = 1e-6
    soft_dice = (2 * intersection + smooth) / (total + smooth)
    dice_loss = 1 - soft_dice.mean()

    return 0.5 * bce + 0.5 * dice_loss


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Submit this script with Slurm on a GPU node.")
    device = "cuda"
    print("Device:", torch.cuda.get_device_name(0), flush=True)

    # One folder per run; the Slurm job id keeps runs apart.
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    run_dir = PROJECT_ROOT / "outputs" / MODEL_NAME / f"job_{job_id}"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Fixed random seeds and a dedicated generator for the training shuffle.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)

    # Only the provided training split is used.
    train_df = pd.read_excel(DATA_ROOT / "train.xlsx")
    train_dataset = BUSISegmentationDataset(train_df, DATA_ROOT)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    print("Training images:", len(train_dataset), "| batches per epoch:", len(train_loader), flush=True)

    model = DeepLabV3Plus(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Save the settings of this run next to its results.
    config = {
        "model": MODEL_NAME,
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "0.5 * BCEWithLogits + 0.5 * soft Dice",
        "augmentation": "none",
        "pretrained": False,
        "train_images": len(train_dataset),
        "slurm_job_id": job_id,
    }
    with open(run_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2)

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()  # enable BatchNorm/Dropout training behavior
        running_loss = 0.0

        for batch_images, batch_masks, _ in train_loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()  # clear old gradients
            loss = segmentation_loss(model(batch_images), batch_masks)
            loss.backward()  # backpropagation
            optimizer.step()  # update the weights

            running_loss += loss.item() * batch_images.size(0)

        # Average loss per training image in this epoch.
        epoch_loss = running_loss / len(train_dataset)
        history.append({"epoch": epoch, "train_loss": epoch_loss})

        # Save progress after every epoch, so nothing is lost if the job stops.
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            run_dir / "checkpoints" / "latest.pth",
        )
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {epoch_loss:.4f} | {elapsed:.1f} min", flush=True)

    # Final weights go into their own folder.
    torch.save(model.state_dict(), run_dir / "weights" / WEIGHT_NAME)
    print("Training finished. Final weights saved to:", run_dir / "weights" / WEIGHT_NAME, flush=True)


if __name__ == "__main__":
    main()