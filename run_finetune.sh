#!/bin/bash

#SBATCH --job-name=CrysLDM        # Job name
#SBATCH --output=crysldm_mp.out
#SBATCH --error=crysldm_mp.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

#export CUDA_VISIBLE_DEVICES='0'
#export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
#export NUMEXPR_MAX_THREADS=64

source /home/rs1/21CS92R01/anaconda3/bin/activate CrysLDM


#python3 -W ignore  train_ae.py --dataset mp_20 --lat_dim 64 --epochs 1000 --batch_size 256


