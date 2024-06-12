#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=%j.txt  # Use %j to denote the job ID
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4G
#SBATCH --exclude=hendrixgpu16fl # Exclude the node with H100s

echo "*** Loading modules ***"

module load anaconda3/2023.03-py3.10
module load python/3.9.16

echo "*** Loading environment ***"

# Define your conda environment name
ENV_NAME="alphastowage"

# Check if the environment exists, and create it if it doesn't
conda info --envs | grep $ENV_NAME &> /dev/null || conda create --name $ENV_NAME --yes

# Activate the environment
source activate $ENV_NAME

echo "*** Installing requirements ***"

# Install the requirements
pip install -r requirements.txt --quiet

echo "*** Running script: ${1:-src/benchmarking.py} ***"

# Run the specified Python script, defaulting to benchmarking.py if none is provided
python3 ${1:-src/benchmarking.py} 