#!/bin/bash

#$ -l gpu=2
#$ -l h=maxg07
#$ -l m_mem_free=64G
#$ -l cuda_name=Tesla-V100-SXM2-32GB
#$ -l h_rt=24:00:00
#$ -o /fast/AG_Sanders/suharto/robots_in_disguise/logs/log_$JOB_ID.txt
#$ -e /fast/AG_Sanders/suharto/robots_in_disguise/logs/log_$JOB_ID.txt

cd /fast/AG_Sanders/suharto/robots_in_disguise/train/

source ~/.bashrc
conda activate pytorch
printf "Conda env activated at : %s" $CONDA_DEFAULT_ENV


python chr21_mock_train.py
