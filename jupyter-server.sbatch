#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=60G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=jupyter

source ~/myconda.sh
source ~/.bash_profile

env_name=$1

# your job execution follows:
whoami
hostname
conda activate catDB
module load gcc/10.5.0
module load cuda/12.2.2 

time jupyter notebook --no-browser --port=8900
