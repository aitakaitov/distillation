#!/bin/bash

declare -a MODELS=( "xlm-roberta-large" )
declare -a LRS=( "1e-5" "5e-5" "5e-6" "1e-6")
declare -a COUNT=(1 2 3)

for LR in ${LRs[@]}; do
  for MODEL in ${MODELS[@]}; do
    for C in ${COUNT[@]}; do
      qsub -v MODEL=$MODEL,LR=$LR run_hello_on_meta.sh
    done
  done
done
