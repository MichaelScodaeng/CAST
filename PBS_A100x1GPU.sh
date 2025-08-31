#!/bin/csh

#################################################################
# A100 1GPU Job Script for HPC System "KAGAYAKI" 
#                                       2022.3.3 k-miya
#################################################################

#PBS -N gpu
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp -m be

source /etc/profile.d/modules.csh
module purge
module load cuda
# Activate conda env (update with your actual environment name)
source activate kan_mammote
cd ${PBS_O_WORKDIR}

nvidia-smi -a > nvidia-smi.log
