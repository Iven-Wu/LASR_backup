#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --job-name=lasr_raw3
#SBATCH --partition=daf
#SBATCH --output=lasr_raw3.o%j
##SBATCH --error=lasr_cus.e%j
##SBATCH --mail-user=NetID@illinois.edu
##SBATCH --mail-type=BEGIN,END
#SBATCH --mem=32G
#
#
# End of embedded SBATCH options
#

# Run the hello world executable (a.out)

module load gcc
module load cmake

python run_script.py --index 3
