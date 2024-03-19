#!/bin/bash
#SBATCH --account desi
#SBATCH --constraint cpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH --nodes 1
#SBATCH --cpus-per-task 256
#SBATCH --array=0-0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export NUMEXPR_MAX_THREADS=4

N_HOD=50000
START_HOD=$((SLURM_ARRAY_TASK_ID * N_HOD))

python /global/u1/e/epaillas/code/emc/abacushod_run.py --start_hod $START_HOD --n_hod $N_HOD