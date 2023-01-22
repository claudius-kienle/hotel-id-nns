salloc -p gpu_4_a100,dev_gpu_4,gpu_4,gpu_8 -t 0:30:00 --gres=gpu:1 --cpus-per-task=40 --nodes=1
# salloc -p single -t 2:00:00 --cpus-per-task=40 --nodes=1 

mkdir -p $TMP/data/dataset
cp data/dataset/*.h5 $TMP/data/dataset
cp data/dataset/*.csv $TMP/data/dataset
