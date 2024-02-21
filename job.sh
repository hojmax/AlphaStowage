#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=res.txt
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

module load anaconda3/2023.03-py3.10
pip install -r requirements.txt --quiet
python3 Train.py --load_run hojmax/bachelor/xz3c50bb --model_path model3399.pt