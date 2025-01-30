#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -o ./preparation_output.out
#$ -pe smp 3

PAIR=$1

source activate DataPrepEnv
python3 ./Run.py $PAIR