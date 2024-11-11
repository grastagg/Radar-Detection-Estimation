#!/bin/zsh

# threads=10
# startSeed=66054190

seeds=(66654257 56654547 66954242 56854448 56954271 66254449 66854281 66454389 66754324 56054251 66054256 66354230 66154399 56754333 56454227 56354501 56154509 66554406 56254262 56554214)
# seeds=(66654257 66954242 66254449 66854281 66454389 66754324 66354230 66154399 66554406)
expCovRatio=9
expDistRatioList=(1 2 3 4 5 6 7 8 9 10)
# expDistRatioList=(10)


for expDistRatio in $expDistRatioList
do
    mkdir -p saved_data/ratioData/expCov$expCovRatio/expDist$expDistRatio
    echo "Running experiment with expDistRatio $expDistRatio"
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
        nohup python3 -u main.py $seed 1 $pathPlanner $expCovRatio $expDistRatio> outputs/optimization/$i.log 2>&1 & 
    done
    wait

    # cp -r saved_data/mc_runs/* saved_data/ratioData/expCov$expCovRatio/expDist$expDistRatio
    # rm -rf saved_data/mc_runs/*
    echo "Finished experiment with expDistRatio $expDistRatio"
    
done


