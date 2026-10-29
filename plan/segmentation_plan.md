# BUSI Segmentation Project Plan

Tasks 6, 7, and 8: U-Net, nnU-Net, DeepLabV3+ on the BUSI dataset.

## 1. Goal

Train three segmentation models (U-Net, nnU-Net, DeepLabV3+) to produce a binary tumor mask on the BUSI breast ultrasound dataset. Compare them with Dice, IoU, and HD95, and document the results in `segmentation_comparison.md`.

## 2. Deliverables

| Item | File / Folder |
|---|---|
| U-Net notebook | `U_Net_segmentation.ipynb` |
| nnU-Net notebook | `nnU_Net_segmentation.ipynb` |
| DeepLabV3+ notebook | `DeepLabV3Plus_segmentation.ipynb` |
| Weights (per model) | `<Model>/weights/` |
| Predicted test masks (per model) | `<Model>/pred_masks/` |
| Final report | `segmentation_comparison.md` |

Suggested folder layout:

```
research_training/
├── BUSI/                      # images, masks, train.xlsx, test.xlsx
├── U_Net/{weights,pred_masks}/
├── nnU_Net/{weights,pred_masks}/
├── DeepLabV3Plus/{weights,pred_masks}/
├── U_Net_segmentation.ipynb
├── nnU_Net_segmentation.ipynb
├── DeepLabV3Plus_segmentation.ipynb
└── segmentation_comparison.md
```

## 3. Fixed Experimental Settings (same for all models)

| Setting | Value |
|---|---|
| Train / test split | From `train.xlsx` and `test.xlsx` (no new random split) |
| Input image and mask size | 256 x 256 |
| Preprocessing / normalization | Same for all models |
| Random seed | 42 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs | 50 |
| Data augmentation | Same for all models (e.g. flip, small rotation) |
| Metrics | Mean Dice, IoU, HD95 on the test set |

> nnU-Net configures itself automatically. Document any settings that cannot be matched exactly.

## 4. Environment Setup (Done / To Verify)

- [x] Connect VS Code to the cluster over SSH
- [x] Conda environment with PyTorch
- [ ] Verify the GPU on a compute node: `srun --gres=gpu:1 --pty python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Install extra packages: `segmentation-models-pytorch medpy albumentations openpyxl pandas matplotlib opencv-python`
- [ ] Start an interactive GPU session and connect the notebook kernel to it
- [ ] Create the output folders
- [ ] Confirm the BUSI dataset location on the server

## 5. Task 6: Training the Models

### 5.1 Shared building blocks (copied into every notebook)

1. Config cell: seed, image size, batch size, learning rate, epochs
2. Read `train.xlsx` / `test.xlsx` to get the file lists
3. Dataset class: load image and mask, resize to 256x256, normalize, binarize mask
4. Merge multiple masks per image (`_mask_1`, `_mask_2`, ...) into one binary mask
5. Augmentation for training only
6. Metric functions: Dice, IoU, HD95
7. Rule for empty masks (normal images): decide how HD95 is handled and document it

### 5.2 Notebook checklist (U-Net and DeepLabV3+)

- [ ] Markdown cell before every code cell explaining its purpose
- [ ] Comments in the code
- [ ] Load data and build DataLoaders
- [ ] Define the model (`segmentation_models_pytorch`)
- [ ] Define the loss (e.g. Dice + BCE) and optimizer
- [ ] Training loop with the fixed settings
- [ ] Save the trained weights to `weights/`
- [ ] Predict on the test set and save binary masks to `pred_masks/`
- [ ] Compute mean Dice, IoU, HD95 on the test set
- [ ] Visualize each test image: image, ground truth, prediction, overlay

### 5.3 nnU-Net notebook

- [ ] Convert BUSI to nnU-Net format (`Dataset001_BUSI`: `imagesTr`, `labelsTr`, `imagesTs`, `dataset.json`)
- [ ] Respect the fixed train/test split from the Excel files
- [ ] Set environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`)
- [ ] Run planning and preprocessing: `nnUNetv2_plan_and_preprocess`
- [ ] Train: `nnUNetv2_train`
- [ ] Predict the test set: `nnUNetv2_predict`
- [ ] Copy predicted masks and weights into the `nnU_Net/` folders
- [ ] Compute Dice, IoU, HD95 with the same metric code
- [ ] Visualization at the end of the notebook

## 6. Task 7: Understand the Metrics and Compare Results

- [ ] **Dice coefficient:** 2|A∩B| / (|A| + |B|), the overlap between prediction and ground truth
- [ ] **IoU:** |A∩B| / |A∪B|
- [ ] **HD95:** the 95th percentile of boundary distances between prediction and ground truth (lower is better), less sensitive to outliers than the plain Hausdorff distance
- [ ] Compare the three models visually and note examples of:
  - under-segmentation
  - over-segmentation
  - missed tumor regions
  - inaccurate boundaries

## 7. Task 8: Documentation

- [ ] Learn basic Markdown syntax
- [ ] Create `segmentation_comparison.md` with:
  1. Introduction to image segmentation
  2. Introduction to U-Net, nnU-Net, DeepLabV3+
  3. Experimental settings
  4. Result table (Dice, IoU, HD95)
  5. Example visualization results
  6. Findings
  7. Conclusion
- [ ] Carefully read and revise the file

## 8. Suggested Order of Work

1. Finish environment setup and GPU check
2. Inspect the dataset structure and the Excel files
3. Write the shared data loader and metrics, and test them
4. Complete the U-Net notebook end to end
5. Reuse it for DeepLabV3+ (only the model changes)
6. Build the nnU-Net notebook
7. Compare the visual results (Task 7)
8. Write and revise `segmentation_comparison.md`

## 9. Notes and Open Questions

- Handling of normal images with empty masks for HD95: _to decide_
- nnU-Net training epochs and settings that differ from the others: _to document_
- Understand every line of code with the help of AI (Task 6, item 10)
