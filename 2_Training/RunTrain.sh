#!/bin/bash

PAIR=$1
WINDOW=$2
THETA=$3

#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -pe smp 5

source activate RLEnv
python3 ./Train.py $PAIR $WINDOW $THETA
