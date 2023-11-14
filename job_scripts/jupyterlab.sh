#!/bin/bash

#$ -N jupyterlab
#$ -l h_rt=10:00:00
#$ -l m_mem_free=32G
#$ -o /fast/AG_Sanders/suharto/robots_in_disguise/.jupyter_logs/jupyterlab_$JOB_ID
#$ -e /fast/AG_Sanders/suharto/robots_in_disguise/.jupyter_logs/jupyterlab_$JOB_ID
#$ -wd /fast/AG_Sanders/suharto/robots_in_disguise/

ROOT_DIR=/fast/AG_Sanders/suharto/robots_in_disguise

source ~/.bashrc
conda activate pytorch
printf "Conda env activated : %s" $CONDA_DEFAULT_ENV

# moving the previous log to archive and getting back to train folder
cd $ROOT_DIR/.jupyter_logs/

curr_log="*"$JOB_ID"*"
find . -maxdepth 1 -type f -not -name $curr_log -exec mv {} archive/. \;

cd $ROOT_DIR

jupyter lab --ip ${HOSTNAME} --port ${SGE_INTERACT_PORT} --no-browser
