#!/bin/bash
#SBATCH --job-name=careamics_training
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=igor.zubarev@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/igor.zubarev/data/splits/split_training_%j.log
#SBATCH --mem=64GB

ROOT_ARG="--rootpath /home/igor.zubarev/data/splits"
WANDB_ARG="--wandb_project test_dao3_ch01"
python /home/igor.zubarev/projects/microSplit-reproducibility/scripts/train_dao_ch01.py $ROOT_ARG $WANDB_ARG