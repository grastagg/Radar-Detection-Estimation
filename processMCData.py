import numpy as np
import importlib
import matplotlib.pyplot as plt

import os


def processMCData(dataFile):
    print()
    print("Processing data from: ", dataFile)
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
    percentFoundOptimized = countOpt / totalCount
    print("countOpt: ", countOpt)
    # print("Average Lawnmower Max PD: ", averageLawnmowerMaxPD)
    # print("Average Optimized Max PD: ", averageOptimizedMaxPD)
    # print(
    #     "Average Lawnmower Max Prob Undiscovered: ", averageLawnmowerMaxProbUndiscovered
    # )
    # print(
    #     "Average Optimized Max Prob Undiscovered: ", averageOptimizedMaxProbUndiscovered
    # )
    # print("Average Lawnmower Time To Find Path: ", averageLawnmowerTimeToFindPath)
    # print("Average Optimized Time To Find Path: ", averageOptimizedTimeToFindPath)
    # print("Average Lawnmower Diff Deterministic: ", averageLawnmowerDiffDeterministic)
    # print("Average Optimized Diff Deterministic: ", averageOptimizedDiffDeterministic)
    #
    # print("Count Lawn: ", countLawn)
    # print("Count Opt: ", countOpt)
    # print("Total Count: ", totalCount)
    # print("Count DistToGoal: ", countDistToGoal)

    return np.array(
        [
            averageOptimizedTimeToFindPath,
            averageOptimizedMaxPD,
            averageOptimizedMaxProbUndiscovered,
            averageOptimizedDiffDeterministic,
            percentFoundOptimized,
        ]
    )


def get_sorted_expCovRatios(dataFile):
    expCovRatios = []
    for expCovFolder in os.listdir(dataFile):
        if expCovFolder[-2] == "v":
            expCovRatio = float(expCovFolder[-1])
        else:
            expCovRatio = float(expCovFolder[-2:])

        expCovRatios.append(expCovRatio)

    expCovRatios = np.array(expCovRatios)
    return np.sort(expCovRatios)


def get_sorted_expDistRatios(dataFile):
    expDistRatios = []
    for expCovFolder in os.listdir(dataFile):
        if expCovFolder[-2] == "t":
            expDistRatio = float(expCovFolder[-1])
        else:
            expDistRatio = float(expCovFolder[-2:])
        expDistRatios.append(expDistRatio)
    expDistRatios = np.array(expDistRatios)
    return np.sort(expDistRatios)


def create_heatmap(dataFile, parameterIndex, title):
    expCoverageRatios = get_sorted_expCovRatios(dataFile)
    valOuterArray = []
    expCovRatiosOuterArray = []
    expDistRatiosOuterArray = []

    for expCovRatio in expCoverageRatios:
        valInnerArray = []
        expCovRatiosInnerArray = []
        expDistRatiosInnerArray = []
        expCovFolder = dataFile + "expCov" + str(int(expCovRatio))
        expDistRatios = get_sorted_expDistRatios(expCovFolder)
        for expDistRatio in expDistRatios:
            expDistFolder = expCovFolder + "/expDist" + str(int(expDistRatio))
            parameter = processMCData(expDistFolder)[parameterIndex]
            valInnerArray.append(parameter)
            expCovRatiosInnerArray.append(expCovRatio)
            expDistRatiosInnerArray.append(expDistRatio)
        valOuterArray.append(valInnerArray)
        expCovRatiosOuterArray.append(expCovRatiosInnerArray)
        expDistRatiosOuterArray.append(expDistRatiosInnerArray)

    values = np.array(valOuterArray)
    expDistRatios = np.array(expDistRatiosOuterArray)
    expCovRatios = np.array(expCovRatiosOuterArray)
    print(values)

    fig, ax = plt.subplots()
    ax.pcolormesh(expDistRatios, expCovRatios, values, cmap="Blues")
    ax.set_xlabel("expDistRatio")
    ax.set_ylabel("expCovRatio")
    ax.set_xticks(expDistRatios[0])
    ax.set_yticks(expCovRatios[:, 0])
    ax.set_aspect("equal")
    ax.set_title(title)
    for i in range(len(expCovRatios)):
        for j in range(len(expDistRatios[0])):
            ax.text(
                expDistRatios[i, j],
                expCovRatios[i, j],
                round(values[i, j], 4),
                ha="center",
                va="center",
                color="black",
            )

    # for expCovFolder in os.listdir(dataFile):
    #     if expCovFolder[-2] == "v":
    #         expCovRatio = float(expCovFolder[-1])
    #     else:
    #         expCovRatio = float(expCovFolder[-2:])
    #
    #     if expCovRatio > maxExpCovRatio:
    #         maxExpCovRatio = expCovRatio
    # print("Max ExpCovRatio: ", maxExpCovRatio)
    #
    # for expCovFolder in os.listdir(dataFile):
    #     print(expCovFolder)
    #     if expCovFolder[-2] == "v":
    #         expCovRatio = float(expCovFolder[-1])
    #     else:
    #         expCovRatio = float(expCovFolder[-2:])
    #     expCovRatio = float(expCovFolder[-1])
    #     print("expCovRatio: ", expCovRatio)
    #     for expDistFolder in os.listdir(dataFile + expCovFolder):
    #         print(expDistFolder)
    #
    #         if expDistFolder[-2] == "t":
    #             expDistRatio = float(expDistFolder[-1])
    #         else:
    #             expDistRatio = float(expDistFolder[-2:])
    #         print("expDistRatio: ", expDistRatio)
    #         (
    #             averageOptimizedTimeToFindPath,
    #             averageOptimizedMaxPD,
    #             averageOptimizedMaxProbUndiscovered,
    #             averageOptimizedDiffDeterministic,
    #             percentFoundOptimized,
    #         ) = processMCData(dataFile + expCovFolder + "/" + expDistFolder)
    #


if __name__ == "__main__":
    parameterNames = [
        "Average Time To Find Path",
        "Average Max PD",
        "Average Max Prob Undiscovered",
        "Average Diff Deterministic",
        "Percent Found Optimized",
    ]
    parameterIndex = 0
    for i, title in enumerate(parameterNames):
        create_heatmap("./saved_data/ratioData/", i, title)
    plt.show()
