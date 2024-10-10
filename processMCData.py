import numpy as np

import os



def processMCData(dataFile):
    averageLawnmowerMaxPD = 0
    averageOptimizedMaxPD = 0
    averageLawnmowerMaxProbUndiscovered = 0
    averageOptimizedMaxProbUndiscovered = 0
    averageLawnmowerTimeToFindPath = 0
    averageOptimizedTimeToFindPath = 0
    averageLawnmowerDiffDeterministic = 0
    averageOptimizedDiffDeterministic = 0
    count = 0
    for file in os.listdir(dataFile):
        print(file)


        lawnmowerMaxPD = np.genfromtxt(dataFile + "/" + file + "/lawnmower/high_priority_path/groundTruthPdAlongSplines.txt", delimiter=",")
        optimizedMaxPD = np.genfromtxt(dataFile + "/" + file + "/optimization/high_priority_path//groundTruthPdAlongSplines.txt", delimiter=",")
        averageLawnmowerMaxPD += np.max(lawnmowerMaxPD)
        averageOptimizedMaxPD += np.max(optimizedMaxPD)
        # print("Lawnmower Max PD: ", np.max(lawnmowerMaxPD))
        # print("Optimized Max PD: ", np.max(optimizedMaxPD))

        lawnmowerMaxProbUndiscovered = np.genfromtxt(dataFile + "/" + file + "/lawnmower/high_priority_path/maxProbUndiscoveredRadar.txt", delimiter=",")
        optimizedMaxProbUndiscovered = np.genfromtxt(dataFile + "/" + file + "/optimization/high_priority_path/maxProbUndiscoveredRadar.txt", delimiter=",")
        # print("Lawnmower Max Prob Undiscovered: ", np.max(lawnmowerMaxProbUndiscovered))
        # print("Optimized Max Prob Undiscovered: ", np.max(optimizedMaxProbUndiscovered))
        averageOptimizedMaxProbUndiscovered += np.max(optimizedMaxProbUndiscovered)
        averageLawnmowerMaxProbUndiscovered += np.max(lawnmowerMaxProbUndiscovered)

        lawnmowerTimeToFindPath = np.genfromtxt(dataFile + "/" + file + "/lawnmower/high_priority_path/lpFindPathTime.txt", delimiter=",")
        optimizedTimeToFindPath = np.genfromtxt(dataFile + "/" + file + "/optimization/high_priority_path/lpFindPathTime.txt", delimiter=",")
        # print("Lawnmower Time To Find Path: ", np.sum(lawnmowerTimeToFindPath))
        # print("Optimized Time To Find Path: ", np.sum(optimizedTimeToFindPath))
        averageLawnmowerTimeToFindPath += np.sum(lawnmowerTimeToFindPath)
        averageOptimizedTimeToFindPath += np.sum(optimizedTimeToFindPath)
        
        optimalPathTime = np.genfromtxt(dataFile + "/" + file + "/lawnmower/high_priority_path/deterministic_path_time.txt", delimiter=",")
        lawnmowerOptimalPathTime = np.genfromtxt(dataFile + "/" + file + "/lawnmower/high_priority_path/optimalTime.txt", delimiter=",")
        optimizedOptimalPathTime = np.genfromtxt(dataFile + "/" + file + "/optimization/high_priority_path/optimalTime.txt", delimiter=",")
        
        averageLawnmowerDiffDeterministic += np.abs(optimalPathTime - lawnmowerOptimalPathTime)/optimalPathTime
        averageOptimizedDiffDeterministic += np.abs(optimalPathTime - optimizedOptimalPathTime)/optimalPathTime
        
        count += 1
    
    averageLawnmowerMaxPD /= count
    averageOptimizedMaxPD /= count
    averageOptimizedMaxProbUndiscovered /= count
    averageLawnmowerMaxProbUndiscovered /= count
    averageOptimizedTimeToFindPath /= count
    averageLawnmowerTimeToFindPath /= count
    averageLawnmowerDiffDeterministic /= count
    averageOptimizedDiffDeterministic /= count
    print("Average Lawnmower Max PD: ", averageLawnmowerMaxPD)
    print("Average Optimized Max PD: ", averageOptimizedMaxPD)
    print("Average Lawnmower Max Prob Undiscovered: ", averageLawnmowerMaxProbUndiscovered)
    print("Average Optimized Max Prob Undiscovered: ", averageOptimizedMaxProbUndiscovered)
    print("Average Lawnmower Time To Find Path: ", averageLawnmowerTimeToFindPath)
    print("Average Optimized Time To Find Path: ", averageOptimizedTimeToFindPath)
    print("Average Lawnmower Diff Deterministic: ", averageLawnmowerDiffDeterministic)
    print("Average Optimized Diff Deterministic: ", averageOptimizedDiffDeterministic)
    print("Count: ", count)



if __name__ == "__main__":
    processMCData("./saved_data/mc_runs/processed4")
    
    

    