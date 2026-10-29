#!/bin/bash
#SBATCH --job-name=busi_unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cv_data/users/seven/Instruction3/logs/%x_%j.out

set -euo pipefail

cd /mnt/cv_data/users/seven/Instruction3

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

/mnt/cv_data/users/seven/envs/ml/bin/python -u "${1:?请指定训练文件}"