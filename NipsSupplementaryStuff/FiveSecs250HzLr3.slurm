#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=18:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2000
#SBATCH --job-name="fs250lr3"
#SBATCH --output=fs250lr3.out
#SBATCH --mail-user=emilamy@stud.ntnu.no
#SBATCH --mail-type=ALL

module spider load GCCcore/.9.3.0
module spider load Python/3.8.2

source ../../dataci/bin/activate

python3 FiveSec250HzLr3.py
