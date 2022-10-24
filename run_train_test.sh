#!/bin/bash
#PBS -q gpu_long@meta-pbs.metacentrum.cz
#PBS -l walltime=240:0:0
#PBS -l select=1:ncpus=2:mem=60gb:scratch_local=100gb:ngpus=1:cl_galdor=True
#PBS -N xlmr

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
DATADIR=/storage/plzen1/home/barticka/distillation
PYTHON_SCRIPT=ner_train.py

singularity shell $CONTAINER
module add conda-modules
conda activate torch

cd $DATADIR

wandb login --relogin bb99cb2b7077af36587e0415e9ea6b87a1f0013b

python $PYTHON_SCRIPT --lr 1e-5 --model xlm-roberta-large --batch_size 16 --epochs 2 --cache_file "${SCRATCHDIR}/cachefile"

rm -rf $SCRATCHDIR/*
