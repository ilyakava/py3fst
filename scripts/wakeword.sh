#!/bin/bash

rm -rf /scratch0/ilya/locDoc/audio_st;
mkdir /scratch0/ilya/locDoc/audio_st;

CUDA_VISIBLE_DEVICES=1 python spectrogram_NN.py \
--model_root=/scratch0/ilya/locDoc/audio_st \
--batch_size 64 \
--train_data_root /scratch1/ilya/locDoc/data/alexa/v4b/train/ \
--val_data_root /scratch1/ilya/locDoc/data/alexa/v4b/val/
