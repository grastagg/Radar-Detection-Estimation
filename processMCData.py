import numpy as np
import importlib

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
    countOpt = 0
    countLawn = 1
    totalCount = 0
    countDistToGoal = 0
    for file in os.listdir(dataFile):
        totalCount += 1
        print(file)
        # importDir = (
        #     "saved_data.mc_runs.processed."
        #     + str(file)
        #     + "."
        #     + "optimization"
        #     + ".params"
        # )
        # params = importlib.import_module(importDir)
        # if params.useDistToGoal:
        #     countDistToGoal += 1

        if os.path.isfile(
            dataFile
            + "/"
            + file
            + "/lawnmower/high_priority_path/groundTruthPdAlongSplines.txt"
        ):
            lawnmowerMaxPD = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/lawnmower/high_priority_path/groundTruthPdAlongSplines.txt",
                delimiter=",",
            )
            averageLawnmowerMaxPD += np.max(lawnmowerMaxPD)
            lawnmowerMaxProbUndiscovered = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/lawnmower/high_priority_path/maxProbUndiscoveredRadar.txt",
                delimiter=",",
            )
            averageLawnmowerMaxProbUndiscovered += np.max(lawnmowerMaxProbUndiscovered)
            lawnmowerTimeToFindPath = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/lawnmower/high_priority_path/lpFindPathTime.txt",
                delimiter=",",
            )
            averageLawnmowerTimeToFindPath += np.sum(lawnmowerTimeToFindPath)
            optimalPathTime = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/lawnmower/high_priority_path/deterministic_path_time.txt",
                delimiter=",",
            )
            lawnmowerOptimalPathTime = np.genfromtxt(
                dataFile + "/" + file + "/lawnmower/high_priority_path/optimalTime.txt",
                delimiter=",",
            )
            averageLawnmowerDiffDeterministic += (
                np.abs(optimalPathTime - lawnmowerOptimalPathTime) / optimalPathTime
            )
            countLawn += 1
        if os.path.isfile(
            dataFile
            + "/"
            + file
            + "/optimization/high_priority_path/groundTruthPdAlongSplines.txt"
        ):
            optimizedMaxPD = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/optimization/high_priority_path//groundTruthPdAlongSplines.txt",
                delimiter=",",
            )
            averageOptimizedMaxPD += np.max(optimizedMaxPD)

            optimizedMaxProbUndiscovered = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/optimization/high_priority_path/maxProbUndiscoveredRadar.txt",
                delimiter=",",
            )
            averageOptimizedMaxProbUndiscovered += np.max(optimizedMaxProbUndiscovered)

            optimizedTimeToFindPath = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/optimization/high_priority_path/lpFindPathTime.txt",
                delimiter=",",
            )
            averageOptimizedTimeToFindPath += np.sum(optimizedTimeToFindPath)

            optimalPathTime = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/optimization/high_priority_path/deterministic_path_time.txt",
                delimiter=",",
            )
            optimizedOptimalPathTime = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/optimization/high_priority_path/optimalTime.txt",
                delimiter=",",
            )

            averageOptimizedDiffDeterministic += (
                np.abs(optimalPathTime - optimizedOptimalPathTime) / optimalPathTime
            )

            countOpt += 1
            print("lpFindPathTime: ", optimizedTimeToFindPath)
        else:
            print("Optimized path not found for: ", file)

    averageLawnmowerMaxPD /= countLawn
    averageOptimizedMaxPD /= countOpt
    averageOptimizedMaxProbUndiscovered /= countOpt
    averageLawnmowerMaxProbUndiscovered /= countLawn
    averageOptimizedTimeToFindPath /= countOpt
    averageLawnmowerTimeToFindPath /= countLawn
    averageLawnmowerDiffDeterministic /= countLawn
    averageOptimizedDiffDeterministic /= countOpt
    print("Average Lawnmower Max PD: ", averageLawnmowerMaxPD)
    print("Average Optimized Max PD: ", averageOptimizedMaxPD)
    print(
        "Average Lawnmower Max Prob Undiscovered: ", averageLawnmowerMaxProbUndiscovered
    )
    print(
        "Average Optimized Max Prob Undiscovered: ", averageOptimizedMaxProbUndiscovered
    )
    print("Average Lawnmower Time To Find Path: ", averageLawnmowerTimeToFindPath)
    print("Average Optimized Time To Find Path: ", averageOptimizedTimeToFindPath)
    print("Average Lawnmower Diff Deterministic: ", averageLawnmowerDiffDeterministic)
    print("Average Optimized Diff Deterministic: ", averageOptimizedDiffDeterministic)

    print("Count Lawn: ", countLawn)
    print("Count Opt: ", countOpt)
    print("Total Count: ", totalCount)
    print("Count DistToGoal: ", countDistToGoal)


if __name__ == "__main__":
    # processMCData("./saved_data/mc_runs/processed/")
    processMCData("./saved_data/mc_runs/")
