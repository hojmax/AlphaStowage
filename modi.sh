#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

echo "*** Loading modules ***"

# Create a virtual environment in your user space
python3 -m venv ~/torch_env

# Activate the virtual environment
source ~/torch_env/bin/activate

# Upgrade pip and install wheel for better package handling
pip install --upgrade pip wheel

echo "*** Installing requirements ***"

# Install the requirements
pip install -r requirements.txt --quiet

echo "*** Running script: ${1:-src/main.py} ***"

# Run the specified Python script, defaulting to main.py if none is provided
python3 ${1:-src/main.py}