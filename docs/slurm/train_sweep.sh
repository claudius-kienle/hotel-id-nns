#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=privat@claudiuskienle.de
#SBATCH --partition=gpu_4,gpu_8,gpu_4_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output="data/logs/train_chain_id-%j.out"
#SBATCH -J TrainChainID

export EXECUTABLE="wandb agent hotel-id-nns/ClassNet/tiigj472"

source ~/.bashrc
mkdir -p $TMP/data/dataset
cp data/dataset/*.h5 $TMP/data/dataset
cp data/dataset/*.csv $TMP/data/dataset
conda activate hotel-id-nns

startexe=${EXECUTABLE}
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
echo $startexe
exec $startexe
