#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --job-name=aaf_cam
#SBATCH --partition=daf
#SBATCH --output=aaf_cam.o%j
##SBATCH --error=aaf_cam.e%j
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

bash scripts/template.sh aardvark_female  aardvark_female

bash scripts/extract_cam.sh aardvark_female-5 10 3 36 aardvark_female no no
