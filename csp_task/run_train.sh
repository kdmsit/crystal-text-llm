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
# Load necessary modules
echo "Hello"
source /scratch/apps/modules/init/bash
module load cuda/12.4
module load python/3.10
# Activate your Python environment
source /home/niloycs/miniconda3/bin/activate TGDMat
# Execute the training script
echo "Hello"
python -W ignore train.py --dataset carbon_24 --batch_size 512 --epochs 500 --timesteps 100