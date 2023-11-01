#!/bin/bash

#$ -l gpu=2
#$ -l h=maxg07
#$ -l m_mem_free=64G
#$ -l cuda_name=Tesla-V100-SXM2-32GB
#$ -l h_rt=24:00:00

source ~/.bashrc
conda activate pytorch
printf "Conda env activated at : %s" $CONDA_DEFAULT_ENV

cd /fast/AG_Sanders/suharto/lm-trial/train/

python chr21_mock_train.py 2>&1 logs_chr1.txt
