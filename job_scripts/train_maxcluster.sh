#!/bin/bash

#$ -l gpu=2
#$ -l h=maxg07
#$ -l m_mem_free=64G
#$ -l cuda_name=Tesla-V100-SXM2-32GB
#$ -l h_rt=24:00:00

cd /fast/AG_Sanders/suharto/robots_in_disguise/train/

source ~/.bashrc
conda activate pytorch
"Conda env activated at : %s" $CONDA_DEFAULT_ENV > log.txt


python chr21_mock_train.py >> log.txt 2>&1
