#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=%j.txt  # Use %j to denote the job ID
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6


# Defines where the package should be installed.
# Since the modi_mount directory content is
# available on each node, we define the package(s) to be installed
# here so that the node can find it once the job is being executed.
export CONDA_PKGS_DIRS=~/modi_mount/conda_dir
# Activate conda in your PATH
# This ensures that we discover every conda environment
# before we try to activate it.
source $CONDA_DIR/etc/profile.d/conda.sh
# Either activate the existing environment
# or create a new one
conda activate DLC
if [ $? != 0 ]; then
    conda create -n DLC -y python=3.8
    conda activate DLC
fi

echo "*** Installing requirements ***"

# Install the requirements
pip install -r requirements.txt --quiet

echo "*** Running script: ${1:-src/main.py} ***"

# Run the specified Python script, defaulting to main.py if none is provided
python3 ${1:-src/main.py}