import numpy as np
import matplotlib.pyplot as plt




from radar import RadarCircularPattern
from agent import Agent
from emitterLocationEstimator import EmitterLocationEstimator 
from batchEmitterLocationEstimator import BatchEmitterLocationEstimator 
from multipleEmitterBatchLocationEstimator import MultipleEmitterBatchLocationEstimator
from gp import GaussianProcess
from multipleEmitterPowerParametricEstimator import MultipleEmittorPowerParametricEstimator
from batchEmitterLocationAndPowerEstimator import BatchEmitterLocationAndPowerEstimator
from multipleEmitterOnlineLocationAndPowerEstimator import MultipleEmitterOnlineLocationAndPowerEstimator
from probabilityOfDetectionMap import ProbabilityOfDetectionMap
import params

# np.random.seed(12342)

def plot_scene(radarList, agentList,bounds,plotIndex,emmitterLocationEstimator, gp, batchEmmiterLocationEstimator, multipleEmmiterLocationEstimator, multipleEmitterPowerParametricEstimator, batchEmitterLocationAndPowerEstimator, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap):
        fig,ax = plt.subplots()
        ax.set_xlim((0,bounds[0]))
        ax.set_ylim((0,bounds[1]))
        ax.set_aspect('equal')
        # c = gp.plot(ax)
        # if c is not None:
            # plt.colorbar(c)
        numMeasurements = 0
        for agent in agentList:
            agent.plot_agent(ax)
            numMeasurements += len(agent.measurementPowerValues)
        if emmitterLocationEstimator is not None:
            c = emmitterLocationEstimator.plot(ax, False)
        # if c is not None:
        #     plt.colorbar(c)
        if batchEmmiterLocationEstimator is not None:
            batchEmmiterLocationEstimator.plot(ax)
        if multipleEmmiterLocationEstimator is not None:
            multipleEmmiterLocationEstimator.plot(ax)
        # c = multipleEmitterPowerParametricEstimator.plot(ax)
        # if c is not None:
        #     plt.colorbar(c)
        if batchEmitterLocationAndPowerEstimator is not None:
            batchEmitterLocationAndPowerEstimator.plot(ax, plot_var = True)

        if multipleEmitterOnlineLocationAndPowerEstimator is not None:
            multipleEmitterOnlineLocationAndPowerEstimator.plot(ax)
            # c = multipleEmitterOnlineLocationAndPowerEstimator.plot(ax)
        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot(ax)
        if c is not None:
            cb = plt.colorbar(c)
            
        for radar in radarList:
            radar.plot_view_area(ax)
        plt.title(numMeasurements-1)
        plt.savefig('images/pd_mean/'+str(plotIndex)+'.png')

        if c is not None:
            cb.remove()
        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_cov(ax)
        if c is not None:
            plt.colorbar(c)
            
        for radar in radarList:
            radar.plot_view_area(ax)
        plt.title(numMeasurements-1)
        plt.savefig('images/pd_cov/'+str(plotIndex)+'.png')
        plt.close()


def main():
    bounds = params.bounds 
    numTestPoints = params.numTestPoints
    X_test = params.create_test_points(numTestPoints, bounds)
    


    radarList = params.radarList
    agentList = params.agentList




    # emmitterLocationEstimator = EmmitterLocationEstimator(groundTruth=np.array([[600,600]]))
    # batchEmmiterLocationEstimator = BatchEmmiterLocationEstimator(np.array([[600,600]]))
    # batchEmiterAndPowerEstimator = BatchEmitterLocationAndPowerEstimator(measurement_cov=np.array([[agent.angle_measurement_std_dev**2,0],[0,agent.power_measurement_std_dev**2]]))
    batchEmiterAndPowerEstimator = None
    emmitterLocationEstimator = None 
    batchEmmiterLocationEstimator = None
    # multipleRadarLocationEstimator = MultipleEmitterBatchLocationEstimator(sensing_range=agent.sensing_range, angle_measurement_std_dev=agent.angle_measurement_std_dev)
    multipleRadarLocationEstimator = None
    # multipleEmitterPowerParametricEstimator = MultipleEmittorPowerParametricEstimator(X_test)
    multipleEmitterPowerParametricEstimator = None

    multipleEmitterOnlineLocationAndPowerEstimator = MultipleEmitterOnlineLocationAndPowerEstimator(sensing_range=params.agentSensingRange, angle_measurement_std_dev=params.agentAngleMeasurementStdDev, measurement_cov=np.array([[params.agentAngleMeasurementStdDev**2,0],[0,params.agentPowerMeasurementStdDev**2]]), X_test=X_test, radar_measurement_coeff=params.radarMeasurementCoeff)
    probabilityOfDetectionMap = ProbabilityOfDetectionMap(X_test)



    
    # gp = GaussianProcess(X_test)
    gp = None 


    
    tEnd = params.simulationEndTime
    dt = params.simulationTimestep
    tCurrent = 0
    plotIndex = 0

    currentNumberOfMeasurements = 0

    
    while tCurrent < tEnd:
        plot_scene(radarList, agentList, bounds, plotIndex,emmitterLocationEstimator, gp, batchEmmiterLocationEstimator, multipleRadarLocationEstimator, multipleEmitterPowerParametricEstimator, batchEmiterAndPowerEstimator, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap)
        for radar in radarList:
            radar.update(dt)
        for agent in agentList:
            agent.update(134,.025,dt,radarList)
        if len(agent.measurementAngleOfArrivalValues) != currentNumberOfMeasurements:
            print("adding measurement:", currentNumberOfMeasurements)
            print("truth group lists",agent.truthEmitterCorrespondence)
            currentNumberOfMeasurements += 1
            multipleEmitterOnlineLocationAndPowerEstimator.add_measurement(agent.measurementLocations[-1], [agent.measurementAngleOfArrivalValues[-1], agent.measurementPowerValues[-1]])
            if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
                # probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, radarList[0], multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, agentList[0])
                probabilityOfDetectionMap.compute_probability_of_detection_at_points(X_test, radarList[0], multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params[0],multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances[0], agentList[0])



        plotIndex += 1
        tCurrent += dt

    
    # plt.figure()
    # plt.plot(np.linspace(0, len(emmitterLocationEstimator.errorHistory), len(emmitterLocationEstimator.errorHistory)), emmitterLocationEstimator.errorHistory, label = "ekf")
    # plt.plot(np.linspace(0, len(batchEmmiterLocationEstimator.errorHistory), len(batchEmmiterLocationEstimator.errorHistory)), batchEmmiterLocationEstimator.errorHistory, label = "batch")
    # plt.legend()
    # plt.show()
    



if __name__ == '__main__':
    main()