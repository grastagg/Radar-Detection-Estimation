#!/bin/zsh

threads=2
startSeed=1213123

for i in $(seq 1 $threads)
do
    seed=$(($startSeed + 1000 * $i))
    echo "Running thread $i with seed $seed"
    
    # Replace the following line with the actual command or script you want to run
    # Example: Run a Python script in the background
    nohup python3 -u main.py $seed 1 > outputs/$i.log 2>&1 & 
done