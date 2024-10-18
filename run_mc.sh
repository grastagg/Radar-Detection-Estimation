#!/bin/zsh

threads=10
startSeed=56054190


for i in $(seq 1 $threads)
# for seed in $seeds
do
    # echo $seed
    seed=$(($startSeed + 100001 * ($i-1)))
    # i=$seed
    echo "Running thread $i with seed $seed"

    
    pathPlanner='lawnmower'
    nohup python3 -u main.py $seed 1 $pathPlanner> outputs/lawnmower/$seed.log 2>&1 & 
    # pathPlanner='optimization'
    # nohup python3 -u main.py $seed 1 $pathPlanner> outputs/optimization/$i.log 2>&1 & 
done
