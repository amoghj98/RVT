#1/bin/bash

source ~/.bashrc
module load conda
module load cuda

export DATA_DIR="/scratch/gautschi/joshi157/datasets/mpx/gen4/"
export CKPT_PATH="/home/joshi157/RVT/rvt-t.ckpt"
export MDL_CFG="tiny"
export USE_TEST=1
export GPU_ID=0

conda activate rvt

HYDRA_FULL_ERROR=1 python validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
    use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" +num_workers=14 \
    batch_size.eval=8 model.postprocess.confidence_threshold=0.001