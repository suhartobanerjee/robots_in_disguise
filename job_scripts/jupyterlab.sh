#!/bin/bash

#$ -N jupyterlab
#$ -l h_rt=48:00:00
#$ -l m_mem_free=16G
#$ -o /fast/AG_Sanders/suharto/robots_in_disguise/.jupyter_logs/jupyterlab_$JOB_ID
#$ -e /fast/AG_Sanders/suharto/robots_in_disguise/.jupyter_logs/jupyterlab_$JOB_ID
#$ -cwd

source ~/.bashrc
conda activate pytorch
printf "Conda env activated : %s" $CONDA_DEFAULT_ENV


jupyter lab --ip ${HOSTNAME} --port ${SGE_INTERACT_PORT} --no-browser
