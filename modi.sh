#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=%j.txt  # Use %j to denote the job ID
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

export PATH="/opt/conda/condabin:$PATH"
conda activate myenv

conda install -c conda-forge -y --file requirements.txt

echo "*** Installing requirements ***"

# Install the requirements
pip install -r requirements.txt --quiet

echo "*** Running script: ${1:-src/main.py} ***"

# Run the specified Python script, defaulting to main.py if none is provided
python3 ${1:-src/main.py}