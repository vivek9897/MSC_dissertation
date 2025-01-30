#!/bin/bash

MODE='Val'  # make sure this is capitalised
THETA=0.00015

for PAIR in AUDJPY AUDUSD CADJPY EURCHF EURGBP EURJPY EURUSD GBPUSD NZDUSD USDCAD USDCHF USDJPY
# for PAIR in EURUSD USDCAD USDJPY
do
  mkdir -p ./EvalLogs/$THETA/
  qsub -o ./EvalLogs/$THETA/$PAIR.txt RunEvaluate.sh $PAIR $THETA $MODE
  echo $PAIR
done