#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes an InceptionV3 model on the Breast Cancer (Malignant) training set.
# 2. Evaluates the model on the Breast Cancer (Malignant) validation set.
#
# Usage:
# ./scripts/finetune_inceptionv3_on_cancer.sh

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/Desktop/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/Desktop/BC_malignant_data/inception_v3

# Where the dataset is saved to.
DATASET_DIR=/Desktop/BC_malignant_data



# Fine-tune only the new layers for 3000 steps.
cd ~/Desktop/TFslim_fine_tune
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cancers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=3000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
cd ~/Desktop/TFslim_fine_tune
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cancers
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3

# Fine-tune all the new layers for 3000 steps.
cd ~/Desktop/TFslim_fine_tune
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=cancers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=3000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
cd ~/Desktop/TFslim_fine_tune
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=cancers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3
