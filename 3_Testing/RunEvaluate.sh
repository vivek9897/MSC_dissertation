#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -o ./evaluate_output.out
#$ -pe smp 5

PAIR=$1
THETA=$2
MODE=$3

source activate RLEnv
python3 ./Evaluate.py $PAIR $THETA $MODE
