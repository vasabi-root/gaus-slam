#!/bin/bash
for exp in 0
do 
    EXP=${exp}
    export EXP
    for seed in 0
    do
        SEED=${seed}
        export SEED
        for scene in 4 3 0 1 2 3 4
        do
            SCENE_NUM=${scene}
            export SCENE_NUM
            echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
            python3 -u -W ignore scripts/gaus_mp.py configs/tum/config.py
        done
    done
done
