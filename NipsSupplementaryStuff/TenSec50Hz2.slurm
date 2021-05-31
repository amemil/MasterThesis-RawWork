#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=1-10:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2000
#SBATCH --job-name="ts50_2"
#SBATCH --output=ts50_2.out
#SBATCH --mail-user=emilamy@stud.ntnu.no
#SBATCH --mail-type=ALL

module spider load GCCcore/.9.3.0
module spider load Python/3.8.2

source ../../dataci/bin/activate

python3 TenSec50Hz2.py
