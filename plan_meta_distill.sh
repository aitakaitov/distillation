#!/bin/bash

declare -a TEACHER_MODELS=( "xlm-roberta-large-trained" )
declare -a STUDENT_MODELS=( "microsoft/Multilingual-MiniLM-L12-H384" )
declare -a TEMPERATURES=( "1.0" "5.0" "1.0")
declare -a LRS=( "1e-5" "5e-5" "5e-6" "1e-6")
declare -a COUNT=(1 2 3)

for LR in ${LRs[@]}; do
  for STUDENT_MODEL in ${STUDENT_MODELS[@]}; do
    for TEACHER_MODEL in ${TEACHER_MODELS[@]}; do
      for TEMPERATURE in ${TEMPERATURES[@]}; do
        for C in ${COUNT[@]}; do
          qsub -v TEACHER_MODEL=$TEACHER_MODEL,STUDENT_MODEL=$STUDENT_MODEL,LR=$LR,TEMPERATURE=$TEMPERATURE run_distill.sh
        dome
      done
    done
  done
done
