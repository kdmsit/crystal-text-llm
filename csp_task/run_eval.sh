#!/bin/bash
#SBATCH --job-name=run_test # Job name
#SBATCH --output=/home/niloycs/workspace_kd/DiffCSP_clone/out/output_%j.txt # Standard output
#SBATCH --error=/home/niloycs/workspace_kd/DiffCSP_clone/error/error_%j.txt # Standard error
#SBATCH --partition=dgx2 # Partition name (modify if using a different node)
#SBATCH --qos=dgx2 # QoS (must match the partition)
#SBATCH --nodes=1 # Use 1 node
#SBATCH --ntasks-per-node=1 # Number of tasks per node (max 8)
#SBATCH --gres=gpu:1 # Number of GPUs (max 8)
#SBATCH --time=24:00:00 # Max runtime 72hrs(HH:MM:SS)
# Navigate to the working directory
cd /home/niloycs/workspace_kd/DiffCSP_clone/csp_task/
# Activate your Python environment
source /home/niloycs/miniconda3/bin/activate TGDMat
# Execute the training script
echo "Hello"
python -W ignore evaluate.py --model_path 'gen/' --chkpt_name out/carbon_24/02022025/162929/model_final.pt  --tasks recon --num_evals 1 --dataset carbon_24 --batch_size 1024 --timesteps 100
python -W ignore compute_metrics.py --root_path gen/carbon_24/ --tasks recon
