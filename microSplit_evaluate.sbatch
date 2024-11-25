#!/bin/bash
#SBATCH --job-name=careamics_training
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --output=/group/jug/federico/careamics_training/microSplit_exp/hpc_outputs/split_eval_%j.log
#SBATCH --mem=64GB



CKPT_ARG="--ckpt_path /group/jug/federico/careamics_training/microSplit_exp/2411/denoisplit_HTIba1Ki67/6/"
DATA_ARG="--data_path /group/jug/ashesh/data/Stefania/20230327_Ki67_and_Iba1_trainingdata"
MMSE_ARG="--mmse_count 5"
SUBDSET_ARG="--subdset_type Iba1NucPercent50"
python /home/federico.carrara/Documents/projects/microSplit-reproducibility/scripts/evaluate_ht_iba1_ki64_2023.py $CKPT_ARG $DATA_ARG $MMSE_ARG $SUBDSET_ARG