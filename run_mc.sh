#!/bin/zsh

threads=1
startSeed=143202793

for i in $(seq 1 $threads)
do
    seed=$(($startSeed + 100000 * ($i-1)))
    echo "Running thread $i with seed $seed"
    
    # pathPlanner='lawnmower'
    # nohup python3 -u main.py $seed 1 $pathPlanner> outputs/lawnmower/$i.log 2>&1 & 
    pathPlanner='optimization'
    nohup python3 -u main.py $seed 1 $pathPlanner> outputs/optimization/$i.log 2>&1 & 
done