#!/bin/bash

#SBATCH --job-name=crysllmgen        # Job name
#SBATCH --output=crysllmgen_mp.out
#SBATCH --error=crysllmgen_mp.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --partition=gpu_l40
#SBATCH --gpus=1

#export CUDA_VISIBLE_DEVICES='0'
#export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
#export NUMEXPR_MAX_THREADS=64

source /home/rs1/21CS92R01/miniconda3/bin/activate crystal-llm


python -W ignore llama_finetune.py --run-name 7b-mpts --model 7b --num-epochs 1 --data-path data/mpts_52


