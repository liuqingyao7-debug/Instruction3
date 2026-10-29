| Model | Dice | IoU | HD95 (px), undefined excluded | HD95 (px), undefined = 362 |
|---|---|---|---|---|
| U-Net | 0.6822 ± 0.2989 | 0.5843 ± 0.3001 | 43.03 ± 45.98 (4 excluded) | 52.84 |
| DeepLabV3+ | 0.7901 ± 0.2399 | 0.7019 ± 0.2527 | 24.25 ± 30.97 (1 excluded) | 26.85 |
| nnU-Net | 0.8245 ± 0.2204 | 0.7436 ± 0.2309 | 20.92 ± 29.78 (2 excluded) | 26.17 |

| Model | Median Dice | Median IoU | Dice ≥ 0.8 | Dice < 0.5 | Dice = 0 |
|---|---|---|---|---|---|
| U-Net | 0.8114 | 0.6826 | 69 / 130 | 32 / 130 | 7 / 130 |
| DeepLabV3+ | 0.8869 | 0.7968 | 91 / 130 | 15 / 130 | 3 / 130 |
| nnU-Net | 0.9031 | 0.8233 | 100 / 130 | 10 / 130 | 2 / 130 |

- Mean Dice nnU-Net - DeepLabV3+: +0.0344 (95% bootstrap interval +0.0114 to +0.0594)
- Mean Dice nnU-Net - U-Net: +0.1423 (95% bootstrap interval +0.1030 to +0.1831)
- Mean Dice DeepLabV3+ - U-Net: +0.1078 (95% bootstrap interval +0.0739 to +0.1438)
