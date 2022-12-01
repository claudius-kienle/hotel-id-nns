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

#Usually you should set
export KMP_AFFINITY=compact,1,0
#export KMP_AFFINITY=verbose,compact,1,0 prints messages concerning the supported affinity
#KMP_AFFINITY Description: https://software.intel.com/en-us/node/524790#KMP_AFFINITY_ENVIRONMENT_VARIABLE

export EXECUTABLE="python hotel_id_nns/scripts/train_chain_id.py data/configs/train_chain_id.json --data-path ${TMP}"

mkdir $TMP/data
cp -r data/dataset $TMP/data
ls $TMP/data

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))

startexe=${EXECUTABLE}
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
exec $startexe
echo $startexe
