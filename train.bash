#1/bin/bash

source ~/.bashrc
module load conda
module load cuda

export DATA_DIR="/scratch/gautschi/joshi157/datasets/mpx/gen4/"
export MDL_CFG="tiny"
export BATCH_SIZE_PER_GPU=24
export TRAIN_WORKERS_PER_GPU=12
export EVAL_WORKERS_PER_GPU=2
export USE_TEST=1
export GPU_IDS=0

conda activate rvt

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 srun python train.py model=rnndet dataset=toffe dataset.name=toffe dataset.path=${DATA_DIR} wandb.project_name=RVT \
    wandb.group_name=1mpx +experiment/gen4="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
    batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
    hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
