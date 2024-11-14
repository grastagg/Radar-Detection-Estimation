import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib

import os
from scipy.interpolate import griddata
import ternary


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

    indcludeFailedRunsInTime = False

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
            if indcludeFailedRunsInTime:
                averageOptimizedTimeToFindPath += 1200
            print("Optimized path not found for: ", file)

    averageLawnmowerMaxPD /= countLawn
    averageOptimizedMaxPD /= countOpt
    averageOptimizedMaxProbUndiscovered /= countOpt
    averageLawnmowerMaxProbUndiscovered /= countLawn
    averageOptimizedTimeToFindPath /= countOpt
    if indcludeFailedRunsInTime:
        averageOptimizedTimeToFindPath /= totalCount
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


def create_simplex_ternary_projection(
    dataFile, parameterIndex, title, module_suffix="56054251"
):
    distWeights = []
    covWeights = []
    expWeights = []
    values = []

    for folder in os.listdir(dataFile):
        try:
            import_dir = (dataFile + folder).replace(
                "/", "."
            ) + "." + module_suffix + ".optimization" ".params"
            print("Importing module:", import_dir)
            params = importlib.import_module(import_dir)

            # Collect weights and values
            distWeights.append(params.distFromStraitWeight)
            covWeights.append(params.nextCovarianceWeight)
            expWeights.append(params.seperationWeight)
            values.append(processMCData(os.path.join(dataFile, folder))[parameterIndex])

        except ModuleNotFoundError as e:
            print(f"Module not found: {import_dir}. Error: {e}")
            continue
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    # Convert lists to arrays
    dist_weights_arr = np.array(distWeights)
    cov_weights_arr = np.array(covWeights)
    exp_weights_arr = np.array(expWeights)
    values_arr = np.array(values)

    # Since the weights already sum to 1, no need for projection or normalization
    projected_x = dist_weights_arr
    projected_y = cov_weights_arr
    projected_z = exp_weights_arr  # Just use the third weight directly

    # Normalize the color values to the range [0, 1] for the colormap
    norm = plt.Normalize(min(values_arr), max(values_arr))
    cmap = matplotlib.cm.viridis  # Choose a colormap (e.g., viridis, plasma, etc.)

    # Set up the ternary plot
    scale = 1
    fig, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=0.08333333333333333)

    # Plotting points in the ternary plot
    for x, y, z, c in zip(projected_x, projected_y, projected_z, values_arr):
        print("point: ", [x, y, z])
        print("color: ", c)
        point = (x, y, z)
        tax.scatter([point], marker="o", color=cmap(norm(c)), s=1000)
        # tax.get_axes().text(x, y, f"{values_arr[i]:.2f}", color="black", fontsize=8)
        tax.annotate(
            f"{c:.2f}",
            (x, y, z),
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
        )
    # scatter = tax.scatter(
    #     np.array([projected_x, projected_y, projected_z]).T,
    #     c=values_arr,
    #     cmap="viridis",
    #     edgecolors="k",
    #     s=100,
    # )

    # Set the labels for the axes
    tax.left_axis_label(r"Exploration Weight ($\alpha_e$)", fontsize=12)
    tax.right_axis_label(r"Covariance Reduction Weight ($\alpha_u$)", fontsize=12)
    tax.bottom_axis_label(r"Strait-Line Distance Weight ($\alpha_d$)", fontsize=12)

    # Adjust ticks to display correctly
    # 0.0 0.08333333333333333 0.16666666666666666 0.25 0.3333333333333333 0.4166666666666667 0.5 0.5833333333333334 0.6666666666666666 0.75 0.8333333333333334 0.9166666666666666 1.0
    tax.ticks(
        axis="lbr", multiple=0.08333333333333333, linewidth=1, tick_formats="%.3f"
    )

    # Add a colorbar
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=values_arr.min(), vmax=values_arr.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation="vertical")
    if parameterIndex == 0:
        cbar.set_label("Seconds", fontsize=12)
    elif parameterIndex == 4:
        cbar.set_label("%", fontsize=12)
    tax.clear_matplotlib_ticks()
    plt.box(False)

    # Add a title
    plt.title(f"{title}", fontsize=14)


#


if __name__ == "__main__":
    parameterNames = [
        "Average Time To Find Path (Only Successful Runs)",
        "Average Max PD",
        "Average Max Prob Undiscovered",
        "Average Diff Deterministic",
        "Percent Found Optimized",
    ]
    parameterIndecies = [0, 4]
    # create_simplex(
    #     "saved_data/ratioData/", parameterIndex, parameterNames[parameterIndex]
    # )
    for i in parameterIndecies:
        # create_heatmap("./saved_data/ratioData/", i, title)
        create_simplex_ternary_projection("saved_data/new_data/", i, parameterNames[i])
    plt.show()
    # processMCData("./saved_data/ratioData/dist0")
