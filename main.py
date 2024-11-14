import numpy as np
import matplotlib.pyplot as plt
import cProfile
from pstats import Stats
import matplotlib.style as mplstyle
import matplotlib

mplstyle.use("fast")
# matplotlib.use('TkAgg')
import time
import getpass
import importlib
import os
import sys


# from radar import RadarCircularPattern
# from agent import Agent
from multipleEmitterOnlineLocationAndPowerEstimator import (
    MultipleEmitterOnlineLocationAndPowerEstimator,
)
from probabilityOfDetectionMap import ProbabilityOfDetectionMap

# import params
from pathPlanning import SplinePathPlanningLowPriority


from main_helper import create_agent_list, create_radar_list

from highPriorityPathPlanner import (
    HighPriorityPathPlanner,
    test_high_priority_path_planner,
)

from utils import (
    change_random_seed,
    copy_params,
    create_data_file,
    set_path_planner,
    change_parameter_ratios,
    change_weights,
)

from lawn_mower_control import LawnMowerControlAllAgents

# np.random.seed(1234)


def plot_scene(
    radarList,
    agentList,
    bounds,
    plotIndex,
    multipleEmitterOnlineLocationAndPowerEstimator,
    probabilityOfDetectionMap,
    lowPriorityPathPlanner,
    numMeasurements,
    fig,
    ax,
    params,
):
    c = None

    # numMeasurements = 0
    for i, agent in enumerate(agentList):
        agent.plot_agent(ax)
        # numMeasurements += len(agent.measurementPowerValues)

    for radar in radarList:
        radar.plot_view_area(ax)

    if multipleEmitterOnlineLocationAndPowerEstimator is not None:
        measurement_lines, estimator_locs = (
            multipleEmitterOnlineLocationAndPowerEstimator.plot(ax)
        )

    if params.plotBestMeasurement:
        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_best_measurement_map(ax)
        cb = None
        if c is not None:
            cb = plt.colorbar(c)

        plt.title(numMeasurements - 1)
        fig.savefig("images/best_measurement/" + str(plotIndex) + ".png")

        if c is not None:
            cb.remove()
            c.remove()
            cb = None
            c = None

    if params.plotPdCov:
        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_cov(ax)
        if c is not None:
            cb = plt.colorbar(c)

        plt.title(numMeasurements - 1)
        plt.savefig("images/pd_cov/" + str(plotIndex) + ".png")

        if c is not None:
            cb.remove()
            c.remove()
            cb = None
            c = None

    if params.plotPd:
        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_mean(ax)
        if c is not None:
            cb = plt.colorbar(c)

        plt.title(numMeasurements - 1)
        scatterPlotList = None

        # if lowPriorityPathPlanner.bestMeasurementLocList is not None:
        #     scatterPlotList = []
        #     for i,point in enumerate(lowPriorityPathPlanner.bestMeasurementLocList):
        #         s = ax.scatter(point[0],point[1],c=params.agentColors[i])
        # scatterPlotList.append(s)

        fig.savefig("images/pd_mean/" + str(plotIndex) + ".png")
        if scatterPlotList is not None:
            for s in scatterPlotList:
                s.remove()
        if c is not None:
            cb.remove()
            c.remove()
            cb = None
            c = None

    # for i,agent in enumerate(agentList):
    if params.plotObjectiveFunction:
        if (
            len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params)
            > 0
        ):
            # c = lowPriorityPathPlanner.plot_objective_and_constraint(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements,agentList,i)
            c = lowPriorityPathPlanner.plot_objective_and_constraint(
                ax,
                probabilityOfDetectionMap.X_test,
                multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params,
                multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances,
                probabilityOfDetectionMap,
                agentList[0].position,
                numMeasurements,
                agentList,
                0,
            )
            # c = lowPriorityPathPlanner.plot_chance_constraints(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements)
            # proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections]
            # ax.legend(proxy, ['safe', 'unsafe'])
            cb = plt.colorbar(c)
        # plt.title(numMeasurements-1)
        ax.set_title(str(numMeasurements - 1))
        # plt.savefig('images/objective_function/'+str(i)+'/'+str(plotIndex)+'.png')
        fig.savefig("images/objective_function/" + str(plotIndex) + ".png")
        if c is not None:
            if cb is not None:
                cb.remove()
            c.remove()
            cb = None
            c = None

    if params.plotChanceConstraints:
        if (
            len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params)
            > 0
        ):
            c = lowPriorityPathPlanner.plot_chance_constraints(
                ax,
                probabilityOfDetectionMap.X_test,
                multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params,
                multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances,
                probabilityOfDetectionMap,
                agentList[0].position,
                numMeasurements,
            )
            # c = lowPriorityPathPlanner.plot_chance_constraints(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements)
            # proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections]
            # ax.legend(proxy, ['safe', 'unsafe'])
            cb = plt.colorbar(c[-1])
        plt.title(numMeasurements - 1)
        fig.savefig("images/chance_constraints/" + str(plotIndex) + ".png")
        if c is not None:
            if cb is not None:
                cb.remove()
            for temp in c:
                temp.remove()
            cb = None
            c = None

    for line in measurement_lines:
        line.remove()
    for loc in estimator_locs:
        loc.remove()


# def convert_lists_to_arrays(paramsList):
def remove_empty_lists(paramsList):
    newParamsList = []
    for i in range(len(paramsList)):
        if len(paramsList[i]) > 0:
            newParamsList.append(paramsList[i].copy())
    return newParamsList


def main(params, dataFile):
    plot = False
    if plot:
        fig, ax = plt.subplots()

        ax.set_xlim((0, params.bounds[0]))
        ax.set_ylim((0, params.bounds[1]))
        ax.set_aspect("equal")
    bounds = params.bounds
    numTestPoints = params.numTestPoints
    X_test = params.create_test_points(numTestPoints, bounds)

    radarList = create_radar_list(
        params.radarPositions,
        params.radarPhases,
        params.radarAngularRates,
        params.radarOutputPowerList,
        params.radarTransmitGainList,
        params.radarRecieveGainList,
        params.radarWavelength,
        params.radarPulseWidth,
        params.radarSystemTemperature,
        params.radarProbabilityOfFalseAlarm,
    )
    agentList = create_agent_list(
        params.agentInitialStates,
        len(radarList),
        params.agentSensingRange,
        params.agentPowerMeasurementStdDev,
        params.agentAngleMeasurementStdDev,
        params.agentELINTAnteneaGain,
        params.agentELINTSystemLoss,
        params.radarWavelength,
        params.agentRadarCrossSection,
        params.agentColors,
        params.agentPathHistorydt,
        dataFile,
        params,
    )

    # radarList = params.radarList
    # agentList = params.agentList

    multipleEmitterOnlineLocationAndPowerEstimator = (
        MultipleEmitterOnlineLocationAndPowerEstimator(
            sensing_range=params.agentSensingRange,
            angle_measurement_std_dev=params.agentAngleMeasurementStdDev,
            measurement_cov=params.measurementCov,
            X_test=X_test,
            radar_measurement_coeff=params.radarMeasurementCoeff,
            params=params,
            radarList=radarList,
            dataFile=dataFile,
        )
    )
    probabilityOfDetectionMap = ProbabilityOfDetectionMap(
        X_test, tuple(radarList), params=params
    )

    if params.lowPriorityPathPlanner == "lawnmower":
        lowPriorityPathPlanner = LawnMowerControlAllAgents(
            numAgents=len(agentList), boundary=params.bounds[0], params=params
        )

    else:
        lowPriorityPathPlanner = SplinePathPlanningLowPriority(params=params)

    tEnd = params.simulationEndTime
    if params.lowPriorityPathPlanner == "lawnmower":
        tEnd *= 2
    else:
        tEnd *= 2
    dt = params.simulationTimestep
    tCurrent = 0
    plotIndex = 0

    currentNumberOfMeasurements = 0
    currentNumberOfMeasurementsArray = np.zeros(len(agentList), dtype=int)

    timeSinceLastPlot = 0

    while tCurrent < tEnd:
        print("current time", tCurrent)
        start = time.time()
        if timeSinceLastPlot >= params.plotTimeStep:
            # if len(remove_empty_lists(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params)) > 0:
            #     probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, remove_empty_lists(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params), remove_empty_lists(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances))

            if plot:
                plot_scene(
                    radarList,
                    agentList,
                    bounds,
                    plotIndex,
                    multipleEmitterOnlineLocationAndPowerEstimator,
                    probabilityOfDetectionMap,
                    lowPriorityPathPlanner,
                    currentNumberOfMeasurements,
                    fig,
                    ax,
                    params,
                )
            timeSinceLastPlot = 0
            plotIndex += 1

        # radar_time = time.time()
        for radar in radarList:
            radar.update(dt)

        # estimator_total_time = 0
        # update_agent_time = 0
        # control_time = 0
        for i, agent in enumerate(agentList):
            # agent.update(134,.0,dt,radarList,currentNumberOfMeasurements)
            turnRate, velocity = lowPriorityPathPlanner.get_control(
                dt, agent.position, i
            )
            agent.update(velocity, turnRate, dt, radarList, currentNumberOfMeasurements)
            if (
                len(agent.measurementAngleOfArrivalValues)
                != currentNumberOfMeasurementsArray[i]
            ):
                # print(
                #     "adding measurement:",
                #     currentNumberOfMeasurementsArray[i],
                #     "from agent",
                #     i,
                # )
                # print("measurement:", currentNumberOfMeasurements)
                # print("truth group lists", agent.truthEmitterCorrespondence)
                currentNumberOfMeasurementsArray[i] += 1
                # start_e = time.time()
                # multipleEmitterOnlineLocationAndPowerEstimator.add_measurement(agent.measurementLocations[-1], [agent.measurementAngleOfArrivalValues[-1], agent.measurementPowerValues[-1]])
                multipleEmitterOnlineLocationAndPowerEstimator.add_measurement_known_association(
                    agent.measurementLocations[-1],
                    [
                        agent.measurementAngleOfArrivalValues[-1],
                        agent.measurementPowerValues[-1],
                    ],
                    agent.radarMeasurementIdx[-1],
                    tCurrent,
                )
                # estimator_total_time += time.time()-start_e
                currentNumberOfMeasurements += 1
                # if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
                # start_pd = time.time()
                # probabilityOfDetEctionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances)
                # print("pd map time", time.time()-start_pd)

        allAgentCurrentPositions = np.array(
            [agent.position[0:2] for agent in agentList]
        )

        # lowPriorityPathPlanner.update_path(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList, currentNumberOfMeasurements,multipleEmitterOnlineLocationAndPowerEstimator.measurement_locations ,dt,allAgentCurrentPositions)

        if params.lowPriorityPathPlanner == "optimization":
            lowPriorityPathPlanner.update_path(
                remove_empty_lists(
                    multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params
                ),
                remove_empty_lists(
                    multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances
                ),
                probabilityOfDetectionMap,
                agentList,
                currentNumberOfMeasurements,
                multipleEmitterOnlineLocationAndPowerEstimator.measurement_locations,
                dt,
                allAgentCurrentPositions,
            )

        timeSinceLastPlot += dt
        tCurrent += dt
    multipleEmitterOnlineLocationAndPowerEstimator.save_radar_estimates_to_file()


def run_mc_simulation(
    startSeed,
    numSeeds,
    lowPriorityPathPlanner,
    explorationWeight,
    covarianceWeight,
    distWeight,
    dataFile,
):
    currentNumSeeds = 0
    numRadar = 13
    username = getpass.getuser()
    # for seed in seeds:
    seed = startSeed
    while currentNumSeeds < numSeeds:
        # lowPriorityPathPlanner = "lawnmower"
        print("running seed:", seed)
        print("dataFile:", dataFile)
        # dataFile = (
        #     "/home/"
        #     + username
        #     + "/repos/magiccvs/radar_detection_estimation/saved_data/mc_runs/"
        #     + str(seed)
        #     + "/"
        # )
        create_data_file(dataFile, numRadar, lowPriorityPathPlanner)
        copy_params(dataFile + lowPriorityPathPlanner + "/")
        paramsFile = dataFile + "/" + lowPriorityPathPlanner + "/" + "params.py"
        change_random_seed(
            dataFile + "/" + lowPriorityPathPlanner + "/" + "params.py", seed
        )
        set_path_planner(
            dataFile + "/" + lowPriorityPathPlanner + "/" + "params.py",
            lowPriorityPathPlanner,
        )
        change_weights(paramsFile, explorationWeight, covarianceWeight, distWeight)

        # set_path_planner("params.py", lowPriorityPathPlanner)
        importDir = dataFile.replace("/", ".") + pathPlanner + ".params"
        print(importDir)
        #
        # importDir = (
        #     "saved_data.mc_runs." + str(seed) + "." + lowPriorityPathPlanner + ".params"
        # )
        # importDir = "saved_data.mc_runs."+str(seed)+".params"
        params = importlib.import_module(importDir)
        if not params.radarPositionsFound:
            print("skipping seed:", seed)
            os.system("rm -r " + dataFile)
            seed += 1
            continue

        radarList = create_radar_list(
            params.radarPositions,
            params.radarPhases,
            params.radarAngularRates,
            params.radarOutputPowerList,
            params.radarTransmitGainList,
            params.radarRecieveGainList,
            params.radarWavelength,
            params.radarPulseWidth,
            params.radarSystemTemperature,
            params.radarProbabilityOfFalseAlarm,
        )
        loadHPPDataFromFile = True
        loadHPPDataFile = (
            "saved_data/new_data/run0/"
            + str(seed)
            + "/optimization/high_priority_path/deterministic_path_time.txt"
        )
        if loadHPPDataFromFile:
            optPathTime = np.genfromtxt(loadHPPDataFile)
        else:
            hpp = HighPriorityPathPlanner(radarList=radarList, params=params)
            try:
                optPathTime = hpp.plan_deterministic_path(tuple(radarList))
                print("Optimal path time:", optPathTime)
            except:
                print("Optimal path not found, skipping seed:", seed)
                os.system("rm -r " + dataFile)
                seed += 1
                continue
        np.savetxt(
            dataFile
            + lowPriorityPathPlanner
            + "/high_priority_path/deterministic_path_time.txt",
            np.array([optPathTime]),
        )
        main(params, dataFile + lowPriorityPathPlanner + "/")
        #
        # ##uncomment to run optimization path planner
        # # lowPriorityPathPlanner = "optimization"
        # # create_data_file(dataFile,numRadar,lowPriorityPathPlanner)
        # # copy_params(dataFile+lowPriorityPathPlanner+"/")
        # # change_random_seed(dataFile + "/"+lowPriorityPathPlanner + "/" + "params.py", seed)
        # # set_path_planner(dataFile + "/"+lowPriorityPathPlanner + "/" + "params.py", lowPriorityPathPlanner)
        # # importDir = "saved_data.mc_runs."+str(seed)+"."+lowPriorityPathPlanner+".params"
        # # params = importlib.import_module(importDir)
        # # main(params)
        seed += 1
        currentNumSeeds += 1


if __name__ == "__main__":
    randomSeed = int(sys.argv[1])
    print("randomSeed", randomSeed)
    numSeeds = int(sys.argv[2])
    print("numSeeds", numSeeds)
    pathPlanner = sys.argv[3]
    print("pathPlanner", pathPlanner)
    # expCovRatio = sys.argv[4]
    # expDist = sys.argv[5]
    explorationWeight = float(sys.argv[4])
    print("explorationWeight", explorationWeight)
    covarianceWeight = float(sys.argv[5])
    print("covarianceWeight", covarianceWeight)
    distWeight = float(sys.argv[6])
    print("distWeight", distWeight)
    dataFile = sys.argv[7]
    print("dataFile", dataFile)
    print("here")

    print("Test", explorationWeight, covarianceWeight, distWeight)

    run_mc_simulation(
        randomSeed,
        numSeeds,
        pathPlanner,
        explorationWeight,
        covarianceWeight,
        distWeight,
        dataFile,
    )
    test_high_priority_path_planner(
        [randomSeed], pathPlanner, dataFile + pathPlanner + "/"
    )
