#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=privat@claudiuskienle.de
#SBATCH --partition=gpu_4_a100,gpu_4,gpu_8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output="data/logs/triplet-%j.out"
#SBATCH -J Triplet

export EXECUTABLE="python hotel_id_nns/scripts/train_triplet_hotel_id.py $1 -m $2 --data-path ${TMP}"

source ~/.bashrc
mkdir -p $TMP/data/dataset
cp data/dataset/*.h5 $TMP/data/dataset
cp data/dataset/*.csv $TMP/data/dataset
conda activate hotel-id-nns

startexe=${EXECUTABLE}
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
echo $startexe
exec $startexe
