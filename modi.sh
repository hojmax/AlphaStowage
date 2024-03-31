#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=%j.txt  # Use %j to denote the job ID
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

echo "*** Installing requirements ***"

# Install the requirements
pip install -r requirements.txt --quiet

echo "*** Running script: ${1:-src/main.py} ***"

# Run the specified Python script, defaulting to main.py if none is provided
python3 ${1:-src/main.py}