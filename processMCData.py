import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib

import os
from scipy.interpolate import griddata
from scipy.sparse import data
import ternary

failedRuns = {}


boxPlotPosition = 0


def processMCData(dataFile, pathPlanner="optimization", boxPlot=False, ax=None):
    global boxPlotPosition
    print("Processing data from: ", dataFile)
    averageOptimizedMaxPD = 0
    averageOptimizedMaxProbUndiscovered = 0
    averageOptimizedTimeToFindPath = 0
    averageOptimizedDiffDeterministic = 0
    countOpt = 0
    totalCount = 0
    testCount = 0

    indcludeFailedRunsInTime = False

    timesToFindPath = []

    countTest = 0
    for file in os.listdir(dataFile):
        totalCount += 1
        if os.path.isfile(
            dataFile
            + "/"
            + file
            + "/"
            + pathPlanner
            + "/high_priority_path/groundTruthPdAlongSplines.txt"
        ):
            plot = False
            if plot:
                fig, ax = plt.subplots()
                img = plt.imread(
                    dataFile
                    + "/"
                    + file
                    + "/"
                    + pathPlanner
                    + "/high_priority_path/spline.png"
                )
                plt.imshow(img)
                plt.show()
            optimizedMaxPD = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/"
                + pathPlanner
                + "/high_priority_path//groundTruthPdAlongSplines.txt",
                delimiter=",",
            )
            averageOptimizedMaxPD += np.max(optimizedMaxPD)

            optimizedMaxProbUndiscovered = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/"
                + pathPlanner
                + "/high_priority_path/maxProbUndiscoveredRadar.txt",
                delimiter=",",
            )
            averageOptimizedMaxProbUndiscovered += np.max(optimizedMaxProbUndiscovered)

            optimizedTimeToFindPath = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/"
                + pathPlanner
                + "/high_priority_path/lpFindPathTime.txt",
                delimiter=",",
            )

            timesToFindPath.append(optimizedTimeToFindPath)
            averageOptimizedTimeToFindPath += np.sum(optimizedTimeToFindPath)

            optimalPathTime = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/"
                + pathPlanner
                + "/high_priority_path/deterministic_path_time.txt",
                delimiter=",",
            )
            optimizedOptimalPathTime = np.genfromtxt(
                dataFile
                + "/"
                + file
                + "/"
                + pathPlanner
                + "/high_priority_path/optimalTime.txt",
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
            plot = True
            if not os.path.isfile(
                dataFile
                + "/"
                + file
                + "/"
                + pathPlanner
                + "/high_priority_path/noPathFound.png"
            ):
                testCount += 1
            if plot:
                fig, ax = plt.subplots()
                # show high_priority_path/noPathFound.png
                if os.path.isfile(
                    dataFile
                    + "/"
                    + file
                    + "/"
                    + pathPlanner
                    + "/high_priority_path/noPathFound.png"
                ):
                    img = plt.imread(
                        dataFile
                        + "/"
                        + file
                        + "/"
                        + pathPlanner
                        + "/high_priority_path/noPathFound.png"
                    )
                    plt.imshow(img)
                    plt.show()

            # if file == "86454584":
            #     countTest += 1
    timesToFindPath = np.array(timesToFindPath)
    print("mean time to find path: ", np.mean(timesToFindPath))
    print("std time to find path: ", np.std(timesToFindPath))
    if boxPlot:
        ax.boxplot(timesToFindPath, positions=[boxPlotPosition])
        boxPlotPosition += 1

    averageOptimizedMaxPD /= countOpt
    averageOptimizedMaxProbUndiscovered /= countOpt
    averageOptimizedTimeToFindPath /= countOpt
    if indcludeFailedRunsInTime:
        averageOptimizedTimeToFindPath /= totalCount
    averageOptimizedDiffDeterministic /= countOpt
    percentFoundOptimized = countOpt / totalCount
    print("countOpt: ", countOpt)
    print("totalCount: ", totalCount)
    print("average lp time: ", averageOptimizedTimeToFindPath)
    print("percent found: ", percentFoundOptimized)

    print("testCount: ", testCount)

    return np.array(
        [
            averageOptimizedTimeToFindPath,
            averageOptimizedMaxPD,
            averageOptimizedMaxProbUndiscovered,
            averageOptimizedDiffDeterministic,
            percentFoundOptimized,
        ]
    ), countTest


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
    dataFile, parameterIndex, title, module_suffix="76854237"
):
    distWeights = []
    covWeights = []
    expWeights = []
    values = []

    countTest = 0
    for folder in os.listdir(dataFile):
        try:
            import_dir = (dataFile + folder).replace(
                "/", "."
            ) + "." + module_suffix + ".optimization" ".params"
            params = importlib.import_module(import_dir)

            # Collect weights and values
            distWeights.append(params.distFromStraitWeight)
            covWeights.append(params.nextCovarianceWeight)
            expWeights.append(params.seperationWeight)
            vals, cTest = processMCData(os.path.join(dataFile, folder))
            val = vals[parameterIndex]
            countTest += cTest

            values.append(val)

        except ModuleNotFoundError as e:
            print(f"Module not found: {import_dir}. Error: {e}")
            continue
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    print("countTest: ", countTest)

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
    tax.gridlines(color="black", multiple=0.08333333333333333)

    # Plotting points in the ternary plot
    for x, y, z, c in zip(projected_x, projected_y, projected_z, values_arr):
        point = (x, y, z)
        tax.scatter([point], marker="o", color=cmap(norm(c)), s=1000)
        # tax.get_axes().text(x, y, f"{values_arr[i]:.2f}", color="black", fontsize=8)
        fontsize = 14
        if parameterIndex == 0:
            fontsize = 10
        tax.annotate(
            f"{c:.0f}",
            (x, y, z),
            fontsize=fontsize,
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
    tax.left_axis_label(r"Exploration Weight ($\alpha_e$)", fontsize=20, offset=0.15)
    tax.right_axis_label(
        r"Covariance Reduction Weight ($\alpha_u$)", fontsize=20, offset=0.15
    )
    tax.bottom_axis_label(r"Distance to Goal Weight ($\alpha_d$)", fontsize=20)

    # Adjust ticks to display correctly
    # 0.0 0.08333333333333333 0.16666666666666666 0.25 0.3333333333333333 0.4166666666666667 0.5 0.5833333333333334 0.6666666666666666 0.75 0.8333333333333334 0.9166666666666666 1.0
    tax.ticks(
        axis="lbr",
        multiple=0.08333333333333333,
        linewidth=1,
        tick_formats="%.3f",
        offset=0.019,
        fontsize=16,
    )

    # Add a colorbar
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=values_arr.min(), vmax=values_arr.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation="vertical")
    cbar.ax.tick_params(labelsize=18)
    if parameterIndex == 0:
        cbar.set_label("Seconds", fontsize=18)
    elif parameterIndex == 4:
        cbar.set_label("%", fontsize=18)
    tax.clear_matplotlib_ticks()
    plt.box(False)

    # Add a title
    plt.title(f"{title}", fontsize=24)


def get_radar_uncertainty_over_time(folder):
    print("TESTING: ", folder)
    numRuns = 13
    detListList = []

    for i in range(numRuns):
        # saved_data/new_data/run37/86154247/optimization/radar_0/estimated_params_cov.npz
        radarData = np.load(folder + "/radar_" + str(i) + "/estimated_params_cov.npz")
        detList = []
        firstCovDet = 0
        for j in radarData.files:
            if radarData[j].shape[0] > 1:
                if firstCovDet == 0:
                    firstCovDet = np.linalg.det(radarData[j])
                detList.append(np.linalg.det(radarData[j]))
            else:
                detList.append(0)

        time = np.genfromtxt(folder + "/radarEstimateTimestamps.txt")
        newTime = np.linspace(time[0], time[-1], 1000)
        detList = np.array(detList)
        print("len time: ", len(time))
        print("len detList: ", len(detList))
        detList[detList == 0] = firstCovDet
        detListResampled = np.interp(newTime, time, detList)
        detListList.append(detListResampled)
    detListList = np.array(detListList)
    detTotal = np.sum(detListList, axis=0)
    # fig, ax = plt.subplots()
    # ax.plot(newTime, np.log10(detTotal))
    # plt.show()
    return detTotal


def plot_radar_uncertainty_over_time(dataFile, pathPlanner="optimization"):
    detList = []
    for folder in os.listdir(dataFile):
        print("Processing data from: ", folder)
        detList.append(
            get_radar_uncertainty_over_time(dataFile + folder + "/" + pathPlanner)
        )
    detList = np.array(detList)
    detTotal = np.mean(detList, axis=0)

    fig, ax = plt.subplots()
    ax.plot(np.log10(detTotal))
    plt.show()


#


if __name__ == "__main__":
    # plot_radar_uncertainty_over_time("saved_data/new_data/run37/")
    fig, ax = plt.subplots()
    # processMCData("saved_data/new_data/run437/", "optimization", boxPlot=True, ax=ax)
    processMCData("saved_data/new_data/run37/", "optimization", boxPlot=True, ax=ax)
    processMCData("saved_data/new_data/run100/", "lawnmower", boxPlot=True, ax=ax)
    processMCData("saved_data/new_data/run101/", "lawnmower", boxPlot=True, ax=ax)
    # processMCData("saved_data/new_data/run537/", "optimization", boxPlot=True, ax=ax)
    plt.show()
    # parameterNames = [
    #     "Average Time To Find Path (Only Successful Runs)",
    #     "Average Max PD",
    #     "Average Max Prob Undiscovered",
    #     "Average Diff Deterministic",
    #     "Percent Successful Runs",
    # ]
    # parameterIndecies = [0, 4]
    # # create_simplex(
    # #     "saved_data/ratioData/", parameterIndex, parameterNames[parameterIndex]
    # # )
    # for i in parameterIndecies:
    #     # create_heatmap("./saved_data/ratioData/", i, title)
    #     create_simplex_ternary_projection("saved_data/new_data/", i, parameterNames[i])
    # plt.show()
    # for key in failedRuns:
    #     print(key)
    #     for val in failedRuns[key]:
    #         print(val)
