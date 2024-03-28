#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=kitty-llama
#SBATCH --output=slurms/%x_%j.out
source ~/myconda.sh
conda activate llama_env
module load gcc-9.3.0-gcc-7.3.0-fjzqkyt

python train_llama.py