#!/bin/bash

# Exit on any error
set -e

# === Configuration Variables ===
ACTIVATION="relu"
OPTIMIZER="Adam"
EPOCHS=100
NF=6
LEARNING_RATE=5e-4
BATCH_SIZE=20
IMAGE_WIDTH=101
IMAGE_HEIGHT=101
DECAY_RATE=0.3
PATIENCE=20

SCALE_FL=1e5
SCALE_OP0=10
SCALE_OP1=1
SCALE_DF=1
SCALE_QF=1
SCALE_RE=1

FILTERS_3D=128
KERNEL_3D="3 3 3"
STRIDE_3D="1 1 1"

FILTERS_2D=128
KERNEL_2D="3 3"
STRIDE_2D="1 1"

DATA_PATH="../data/20241118_data_splited.mat"
# === Run training ===
echo "Launching training with the following parameters:"

python train_tf.py \
    --activation $ACTIVATION \
    --optimizer $OPTIMIZER \
    --epochs $EPOCHS \
    --nF $NF \
    --learningRate $LEARNING_RATE \
    --batch $BATCH_SIZE \
    --xX $IMAGE_WIDTH \
    --yY $IMAGE_HEIGHT \
    --decayRate $DECAY_RATE \
    --scaleFL $SCALE_FL \
    --scaleOP0 $SCALE_OP0 \
    --scaleOP1 $SCALE_OP1 \
    --scaleDF $SCALE_DF \
    --scaleQF $SCALE_QF \
    --scaleRE $SCALE_RE \
    --nFilters3D $FILTERS_3D \
    --kernelConv3D $KERNEL_3D \
    --strideConv3D $STRIDE_3D \
    --nFilters2D $FILTERS_2D \
    --kernelConv2D $KERNEL_2D \
    --strideConv2D $STRIDE_2D \
    --data_path $DATA_PATH

echo "Training complete."