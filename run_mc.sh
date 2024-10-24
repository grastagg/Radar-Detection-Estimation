#!/bin/zsh

# threads=10
# startSeed=66054190

# seeds=(66654209 66954242 66254449 66854281 66454389 66754324 66054256 66354230 66154399 66554406 56854448 56954271 56054251 56754333 56454227 56654259 56354501 56154509 56254262 56554214)
# seeds=(66654257 56654547 66954242 56854448 56954271 66254449 66854281 66454389 66754324 56054251 66054256 66354230 66154399 56754333 56454227 56354501 56154509 66554406 56254262 56554214)
seeds=(66654257)

# # for i in $(seq 1 $threads)
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


wait

echo "All threads finished"
