#!/bin/bash

#SBATCH --job-name=InteractionAnalysis
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu=10000
#SBATCH -p bigmem
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err

echo "Did you run 'module load vmd'?"
python simulation_interaction_analysis.py -t 300 -s1 PROR -s2 PROA -s1n B1AR -s2n Ga -s1x "(resid 392 to 402) or (resid 255 to 318)"