#!/bin/bash

MIN_WINDOW=0
MAX_WINDOW=49

PAIR='USDJPY'
THETA=0.00029

# loop over each maximum window value
for WINDOW in $( eval echo {$MIN_WINDOW..$MAX_WINDOW} )
    do
        # create logging folder if needed and submit
        mkdir -p ./TrainLogs/$THETA/$PAIR/
        qsub -o ./TrainLogs/$THETA/$PAIR/output_window_$WINDOW.txt RunTrain.sh $PAIR $WINDOW $THETA
        echo $WINDOW
        
    done