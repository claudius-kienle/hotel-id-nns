#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=privat@claudiuskienle.de
#SBATCH --partition=dev_gpu_4,gpu_4,gpu_8,gpu_4_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output="data/logs/train_chain_id-%j.out"
#SBATCH -J TrainChainID

export EXECUTABLE="python hotel_id_nns/scripts/train_chain_id.py data/configs/train_chain_id_ce.json --data-path ${TMP}"

mkdir -p $TMP/data/dataset
cp data/dataset/*.h5 $TMP/data/dataset
cp data/dataset/*.csv $TMP/data/dataset

startexe=${EXECUTABLE}
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
echo $startexe
exec $startexe
