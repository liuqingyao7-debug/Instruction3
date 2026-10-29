"""Train nnU-Net (2D configuration) on the converted BUSI dataset.

The dataset must already be converted (notebook Section 4) and preprocessed
(notebook Sections 5-6). nnU-Net is trained on ALL training cases (fold "all"),
so no validation split is created, and the test images are never used here.
The trainer "nnUNetTrainer_50epochs" runs 50 epochs (1 epoch = 250 iterations in nnU-Net),
the same epoch number as the U-Net and DeepLabV3+ experiments.

Usage (from the Instruction3 folder):
    mkdir -p logs
    sbatch --job-name=nnunet --time=04:00:00 submit.sh train_nnunet.py

To continue an interrupted run from its last checkpoint:
    sbatch --job-name=nnunet --time=04:00:00 --export=ALL,NNUNET_RESUME=1 submit.sh train_nnunet.py

Outputs (weights, logs, plans) are written to:
    outputs/nnunet/nnUNet_results/Dataset501_BUSI/nnUNetTrainer_50epochs__nnUNetPlans__2d/fold_all/
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/cv_data/users/seven/Instruction3")
NNUNET_BASE = PROJECT_ROOT / "outputs" / "nnunet"
NNUNET_ENV_BIN = Path("/mnt/cv_data/users/seven/envs/nnunet/bin")   # separate environment for nnU-Net

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_BUSI"
CONFIGURATION = "2d"
FOLD = "all"                          # use all training cases, no validation split
TRAINER = "nnUNetTrainer_50epochs"    # 50 epochs, like the other two models


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Submit this script with Slurm on a GPU node.")
    print("Device:", torch.cuda.get_device_name(0), flush=True)

    # nnU-Net finds its folders through these environment variables.
    os.environ["nnUNet_raw"] = str(NNUNET_BASE / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(NNUNET_BASE / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(NNUNET_BASE / "nnUNet_results")

    # Use the 4 CPUs allocated by Slurm for data augmentation workers, and avoid
    # torch.compile, which needs a compiler that may be missing on the cluster.
    os.environ.setdefault("nnUNet_n_proc_DA", "4")
    os.environ.setdefault("nnUNet_compile", "false")

    # Find the nnU-Net executable: first in the dedicated nnunet environment,
    # then next to the Python that runs this script.
    candidates = [NNUNET_ENV_BIN, Path(sys.executable).parent]
    bin_dir = next((c for c in candidates if (c / "nnUNetv2_train").is_file()), None)
    if bin_dir is None:
        raise FileNotFoundError(
            "nnUNetv2_train not found in: " + ", ".join(str(c) for c in candidates)
        )
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"

    # The preprocessing step must have been run first.
    plans = NNUNET_BASE / "nnUNet_preprocessed" / DATASET_NAME / "nnUNetPlans.json"
    if not plans.is_file():
        raise FileNotFoundError(f"Run the planning/preprocessing cells first: {plans}")

    command = [
        str(bin_dir / "nnUNetv2_train"),
        str(DATASET_ID), CONFIGURATION, FOLD,
        "-tr", TRAINER,
        "-device", "cuda",
    ]
    if os.environ.get("NNUNET_RESUME") == "1":
        command.append("--c")          # continue from the last checkpoint

    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()