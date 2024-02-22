#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=res.txt
#SBATCH -p gpu --gres=gpu:2
#SBATCH --time=10:00:00

module load anaconda3/2023.03-py3.10
pip install -r requirements.txt --quiet
python3 multi_thread.py