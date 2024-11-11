import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib

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


def create_simplex(dataFile, parameterIndex, title):
    expCoverageRatios = get_sorted_expCovRatios(dataFile)
    valOuterArray = []
    distWeightOuterArray = []
    covWeightOuterArray = []
    expWeightOuterArray = []

    valTempArray = []
    distWeightTempArray = []
    covWeightTempArray = []
    expWeightTempArray = []

    for expCovRatio in expCoverageRatios:
        valInnerArray = []
        distWeightInnerArray = []
        covWeightInnerArray = []
        expWeightInnerArray = []
        expCovFolder = dataFile + "expCov" + str(int(expCovRatio))
        expDistRatios = get_sorted_expDistRatios(expCovFolder)
        for expDistRatio in expDistRatios:
            expDistFolder = expCovFolder + "/expDist" + str(int(expDistRatio))
            parameter = processMCData(expDistFolder)[parameterIndex]
            valInnerArray.append(parameter)
            print(expCovRatio, expDistRatios)
            importDir = (
                "saved_data.ratioData.expCov"
                + str(int(expCovRatio))
                + ".expDist"
                + str(int(expDistRatio))
                + "."
                + str(56054251)
                + "."
                + "optimization"
                + ".params"
            )
            params = importlib.import_module(importDir)
            distWeightInnerArray.append(params.distFromStraitWeight)
            covWeightInnerArray.append(params.nextCovarianceWeight)
            expWeightInnerArray.append(params.seperationWeight)

            valTempArray.append(parameter)
            distWeightTempArray.append(params.distFromStraitWeight)
            covWeightTempArray.append(params.nextCovarianceWeight)
            expWeightTempArray.append(params.seperationWeight)
        valOuterArray.append(valInnerArray)
        distWeightOuterArray.append(distWeightInnerArray)
        covWeightOuterArray.append(covWeightInnerArray)
        expWeightOuterArray.append(expWeightInnerArray)

    values = np.array(valOuterArray)
    print(values)
    distWeightOuterArray = np.array(distWeightOuterArray)
    covWeightOuterArray = np.array(covWeightOuterArray)
    expWeightOuterArray = np.array(expWeightOuterArray)
    # Create a figure and a 3D axis
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    #
    # # Define the range for x and y
    # x = np.linspace(0, 1, 100)
    # y = np.linspace(0, 1, 100)
    #
    # # Meshgrid for x and y
    # X, Y = np.meshgrid(x, y)
    #
    # # Calculate Z based on x + y + z = 1 => z = 1 - x - y
    # Z = 1 - X - Y
    #
    # # Mask values outside the range 0 < z < 1
    # Z = np.where((Z > 0) & (Z < 1), Z, np.nan)
    #
    # # Plot the surface
    # ax.plot_surface(X, Y, Z, color="cyan", edgecolor="gray", alpha=0.7)
    #
    # # Set labels and title
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Surface plot of x + y + z = 1")
    #
    # plt.show()
    #
    # Normalize V for color mapping
    valuesTemp = np.array(valTempArray)
    distWeightTemp = np.array(distWeightTempArray)
    covWeightTemp = np.array(covWeightTempArray)
    expWeightTemp = np.array(expWeightTempArray)
    norm = matplotlib.colors.Normalize(vmin=valuesTemp.min(), vmax=valuesTemp.max())
    colors = plt.cm.viridis(norm(valuesTemp))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(distWeightTemp, covWeightTemp, expWeightTemp, c=colors)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)

    print("test", values.shape)
    norm = matplotlib.colors.Normalize(vmin=values.min(), vmax=values.max())
    colors = plt.cm.viridis(norm(values))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        distWeightOuterArray,
        covWeightOuterArray,
        expWeightOuterArray,
        facecolors=colors,
        edgecolor="k",
        alpha=0.7,
        cmap="Blues",
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    mappable.set_array(values)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
    plt.show()
    #
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
        # create_heatmap("./saved_data/ratioData/", i, title)
        create_simplex("./saved_data/ratioData/", i, title)
    plt.show()
    # processMCData("./saved_data/ratioData/dist0")
