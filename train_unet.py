#!/usr/bin/env python3
"""Standalone U-Net training for BUSI; independent of the notebook.

Run in a Slurm GPU allocation: python -u train.py
Validate files without training: python train.py --check-data
Dependencies: torch, numpy, pandas, Pillow, openpyxl.
This script trains only; test metrics and visualization are separate steps.
"""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Same base-32 U-Net and parameter names as the guided notebook."""
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)



class BUSIDataset(Dataset):
    def __init__(self, root, names):
        self.root = Path(root)
        self.names = list(names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / 'images' / name) as image:
            original_size = image.size
            image = image.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
            image_array = np.array(image, dtype=np.float32) / 255.0
        with Image.open(self.root / 'labels' / name) as mask:
            raw = np.array(mask)
            if mask.size != original_size or raw.ndim != 2:
                raise ValueError(f'Invalid mask size/shape: {name}')
            if not np.isin(raw, [0, 1]).all():
                raise ValueError(f'Mask must contain palette indices 0/1: {name}')
            mask = mask.resize((256, 256), Image.Resampling.NEAREST)
            mask_array = np.array(mask, dtype=np.float32)
        return (torch.from_numpy(image_array).unsqueeze(0),
                torch.from_numpy(mask_array).unsqueeze(0), name)


def read_splits(root):
    splits = {}
    for split in ['train', 'test']:
        df = pd.read_excel(root / f'{split}.xlsx')
        if 'Image' not in df or df.empty:
            raise ValueError(f'{split}.xlsx is empty or lacks the Image column')
        names = df['Image']
        if names.isna().any() or not names.is_unique:
            raise ValueError(f'Missing or duplicate names in {split}.xlsx')
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f'Expected a plain image filename: {name!r}')
            for folder in ['images', 'labels']:
                if not (root / folder / name).is_file():
                    raise FileNotFoundError(root / folder / name)
        splits[split] = names.tolist()
    if set(splits['train']) & set(splits['test']):
        raise ValueError('Training and test filenames overlap')
    return splits


def segmentation_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probability * targets).sum(dims)
    total = probability.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + 1e-6) / (total + 1e-6)
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/cv_data/users/seven/datasets/Dataset_BUSI_with_GT_Clean'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, help='Must be a new directory; defaults to outputs/unet/RUN_ID next to train.py')
    parser.add_argument('--check-data', action='store_true', help='Validate all files on CPU and exit without training')
    args = parser.parse_args()
    args.model = "unet"
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs >= 1, batch-size >= 1 and a positive finite lr are required')

    splits = read_splits(args.data_root)
    print(f"Train: {len(splits['train'])} | Test: {len(splits['test'])} | Overlap: 0", flush=True)
    if args.check_data:
        for split, names in splits.items():
            dataset = BUSIDataset(args.data_root, names)
            for i in range(len(dataset)):
                dataset[i]
            print(f'{split}: all image/mask pairs passed validation', flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Submit through Slurm; do not train on the login node.')
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(args.seed)
    dataset = BUSIDataset(args.data_root, splits['train'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    model = UNet().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_id = 'job_' + os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir or Path(__file__).resolve().parent / 'outputs' / args.model / run_id
    # Refuse to overwrite an earlier experiment.
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'weights'
    weights.mkdir()
    config = {**vars(args), 'data_root': str(args.data_root.resolve()), 'output_dir': str(output.resolve()),
              'image_size': [256, 256], 'normalization': 'grayscale / 255', 'mask_resize': 'nearest',
              'augmentation': 'none', 'optimizer': 'Adam', 'loss': '0.5 BCE + 0.5 soft Dice',
              'selection': 'final epoch; no test-driven selection', 'pretrained_weights': None,
              'batching': 'shuffle; retain all samples including the last partial batch',
              'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'split_sha256': {s: hashlib.sha256((args.data_root / f'{s}.xlsx').read_bytes()).hexdigest() for s in splits}}
    config['base_channels'] = 32
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'split_filenames.json').write_text(json.dumps(splits, indent=2))
    print('GPU:', config['gpu'], flush=True)
    print('Output directory:', output.resolve(), flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss, seen = 0.0, 0
        for images, masks, _ in loader:
            images, masks = images.to('cuda'), masks.to('cuda')
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != masks.shape:
                raise RuntimeError(f'Output/mask shape mismatch: {logits.shape} vs {masks.shape}')
            loss = segmentation_loss(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
        record = {'epoch': epoch, 'train_loss': total_loss / seen, 'seconds': time.time()-start}
        history.append(record)
        atomic_save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'history': history,
                     'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                     'loader_rng_state': generator.get_state()}, weights / 'latest_checkpoint.pth')
        temporary_csv = output / 'training_history.tmp'
        pd.DataFrame(history).to_csv(temporary_csv, index=False)
        temporary_csv.replace(output / 'training_history.csv')
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={record['train_loss']:.4f} | {record['seconds']:.1f}s", flush=True)
    atomic_save(model.state_dict(), weights / f'{args.model}_final.pth')
    print('Training completed. Final weights:', weights / f'{args.model}_final.pth', flush=True)
    print('Next: evaluate the final weights on test.xlsx and produce visualizations.', flush=True)


if __name__ == '__main__':
    main()
