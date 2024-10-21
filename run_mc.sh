#!/bin/zsh

threads=10
startSeed=56054190

seeds=(56854448 56954271 56054251 56754333 56454227 56654259 56354501 56154509 56254262 56554214)

# for i in $(seq 1 $threads)
for seed in $seeds
do
    # echo $seed
    # seed=$(($startSeed + 100001 * ($i-1)))
    i=$seed
    echo "Running thread $i with seed $seed"

    
    # pathPlanner='lawnmower'
    # nohup python3 -u main.py $seed 1 $pathPlanner> outputs/lawnmower/$seed.log 2>&1 & 
    pathPlanner='optimization'
    nohup python3 -u main.py $seed 1 $pathPlanner> outputs/optimization/$i.log 2>&1 & 
done
