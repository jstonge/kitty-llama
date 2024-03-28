#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=60G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=jupyter
source ~/myconda.sh
env_name=$1

conda activate llama_env
module load gcc-9.3.0-gcc-7.3.0-fjzqkyt
# your job execution follows:
whoami
hostname


time jupyter notebook --no-browser --port=8900
