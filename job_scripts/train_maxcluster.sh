#!/bin/bash

#$ -l gpu=3
#$ -l h=maxg14
#$ -l m_mem_free=64G
#$ -l h_rt=02:00:00
#$ -o /fast/AG_Sanders/suharto/robots_in_disguise/logs/log_$JOB_ID.txt
#$ -e /fast/AG_Sanders/suharto/robots_in_disguise/logs/log_$JOB_ID.txt

ROOT_DIR=/fast/AG_Sanders/suharto/robots_in_disguise

# moving the previous log to archive and getting back to train folder
cd $ROOT_DIR/logs/

curr_log="*"$JOB_ID"*"
find . -maxdepth 1 -type f -not -name $curr_log -exec mv {} archive/. \;
cd $ROOT_DIR/train

source ~/.bashrc
conda activate pytorch
printf "Conda env activated at : %s\n" $CONDA_DEFAULT_ENV

export GPUS=$(echo $SGE_HGR_gpu | sed 's/gpu//g' | sed 's/ /,/g')
#printf "CUDA_VISIBLE_DEVICES=%s\n" $(echo $CUDA_VISIBLE_DEVICES)

time python chr21_mock_train.py
# -l cuda_name=Tesla-V100-SXM2-32GB
