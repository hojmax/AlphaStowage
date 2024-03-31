#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Define the directory where Conda packages will be stored
export CONDA_PKGS_DIRS=~/modi_mount/conda_dir

# Add Conda's condabin to PATH
export PATH="/opt/conda/condabin:$PATH"

# Attempt to activate the environment, or create it if it doesn't exist
conda activate myenv
if [ $? -ne 0 ]; then
    conda create -n myenv -y python=3.8
    conda activate myenv
fi

echo "*** Installing requirements ***"

# Install the requirements
pip install -r requirements.txt --quiet

echo "*** Running script: ${1:-src/main.py} ***"

# Run the specified Python script, defaulting to main.py if none is provided
python3 ${1:-src/main.py}