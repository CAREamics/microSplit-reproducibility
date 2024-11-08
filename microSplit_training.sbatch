#!/bin/bash
#SBATCH --job-name=careamics_training
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --output=/group/jug/federico/careamics_training/microSplit_exp/hpc_outputs/split_training_%j.log
#SBATCH --mem=128GB

# Check if mamba is installed
MAMBA_PATH=$(whereis mamba | awk '{print $2}')

# If mamba is not installed, MAMBA_PATH will be empty
if [ -z "$MAMBA_PATH" ]
then
    echo "Mamba is not installed. Running initialization..."
    ~/miniforge3/bin/conda init
    ~/miniforge3/bin/mamba init
else
    echo "Using mamba at: $MAMBA_PATH"
fi

source $HOME/.bashrc
mamba activate microsplit_env


ALG_ARG="--algorithm denoiSplit"
ROOT_ARG="--root_path /group/jug/federico/careamics_training/microSplit_exp"
DATA_ARG="--data_path /group/jug/ashesh/data/Stefania/20230327_Ki67_and_Iba1_trainingdata"
WANDB_ARG="--wandb_project ht_iba1_ki64_2023_test"
python /home/federico.carrara/Documents/projects/microSplit-reproducibility/scripts/train_ht_iba1_ki64_2023.py $ALG_ARG $ROOT_ARG $DATA_ARG $WANDB_ARG