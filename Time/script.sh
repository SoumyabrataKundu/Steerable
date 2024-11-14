#!/bin/bash


#SBATCH --job-name=Time
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --account=pi-risi
#SBATCH --partition=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


module load python
source activate /home/soumyabratakundu/.conda/envs/conda_env


python time.py --channel=CHANNEL --kernel=KERNEL --n_radius=RADIUS --max_m=MAXM --restricted=RESTRICTED --conv_first=CONVFIRST
