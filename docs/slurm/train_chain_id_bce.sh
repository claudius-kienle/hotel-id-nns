#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=privat@claudiuskienle.de
#SBATCH --partition=gpu_4,gpu_8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output="data/logs/train_chain_id-%j.out"
#SBATCH -J TrainChainID

export EXECUTABLE="python hotel_id_nns/scripts/train_chain_id.py data/configs/train_chain_id_bce.json --data-path ${TMP}"

mkdir $TMP/data
cp -r data/dataset $TMP/data
ls $TMP/data

startexe=${EXECUTABLE}
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
echo $startexe
exec $startexe
