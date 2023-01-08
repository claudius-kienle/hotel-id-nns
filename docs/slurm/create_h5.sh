#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=privat@claudiuskienle.de
#SBATCH --partition=single,dev_single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output="data/logs/h5-%j.out"
#SBATCH -J H5

export EXECUTABLE="python hotel_id_nns/tools/csv_to_h5_converter.py"

source ~/.bashrc
conda activate hotel-id-nns

startexe=${EXECUTABLE}
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
echo $startexe
exec $startexe
