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

def plot_scene(radarList, agentList,bounds,plotIndex, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap):
        fig,ax = plt.subplots()
        ax.set_xlim((0,bounds[0]))
        ax.set_ylim((0,bounds[1]))
        ax.set_aspect('equal')

        numMeasurements = 0
        for agent in agentList:
            agent.plot_agent(ax)
            numMeasurements += len(agent.measurementPowerValues)

        if multipleEmitterOnlineLocationAndPowerEstimator is not None:
            c = multipleEmitterOnlineLocationAndPowerEstimator.plot(ax)

        if c is not None:
            cb = plt.colorbar(c)
            
        for radar in radarList:
            radar.plot_view_area(ax)
        plt.title(numMeasurements-1)
        plt.savefig('images/best_measurement/'+str(plotIndex)+'.png')

        if c is not None:
            cb.remove()

        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_cov(ax)
        if c is not None:
            cb = plt.colorbar(c)
            
        for radar in radarList:
            radar.plot_view_area(ax)
        plt.title(numMeasurements-1)
        plt.savefig('images/pd_cov/'+str(plotIndex)+'.png')

        if c is not None:
            cb.remove()

        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_mean(ax)
        if c is not None:
            plt.colorbar(c)
            
        for radar in radarList:
            radar.plot_view_area(ax)
        plt.title(numMeasurements-1)
        plt.savefig('images/pd_mean/'+str(plotIndex)+'.png')
        plt.close()


def main():
    bounds = params.bounds 
    numTestPoints = params.numTestPoints
    X_test = params.create_test_points(numTestPoints, bounds)
    


    radarList = params.radarList
    agentList = params.agentList





    multipleEmitterOnlineLocationAndPowerEstimator = MultipleEmitterOnlineLocationAndPowerEstimator(sensing_range=params.agentSensingRange, angle_measurement_std_dev=params.agentAngleMeasurementStdDev, measurement_cov=np.array([[params.agentAngleMeasurementStdDev**2,0],[0,params.agentPowerMeasurementStdDev**2]]), X_test=X_test, radar_measurement_coeff=params.radarMeasurementCoeff)
    probabilityOfDetectionMap = ProbabilityOfDetectionMap(X_test)



    


    
    tEnd = params.simulationEndTime
    dt = params.simulationTimestep
    tCurrent = 0
    plotIndex = 0

    currentNumberOfMeasurements = 0
    currentNumberOfMeasurementsArray = np.zeros(len(agentList),dtype=int)


    
    while tCurrent < tEnd:
        plot_scene(radarList, agentList, bounds, plotIndex, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap)
        for radar in radarList:
            radar.update(dt)
        for i,agent in enumerate(agentList):
            agent.update(134,.025,dt,radarList)
            if len(agent.measurementAngleOfArrivalValues) != currentNumberOfMeasurementsArray[i]:
                print("adding measurement:", currentNumberOfMeasurementsArray[i], "from agent",i)
                print("measurement:", currentNumberOfMeasurements)
                print("truth group lists",agent.truthEmitterCorrespondence)
                currentNumberOfMeasurementsArray[i] += 1
                multipleEmitterOnlineLocationAndPowerEstimator.add_measurement(agent.measurementLocations[-1], [agent.measurementAngleOfArrivalValues[-1], agent.measurementPowerValues[-1]])
                currentNumberOfMeasurements += 1
                if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
                    probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, radarList[0], multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances)



        plotIndex += 1
        tCurrent += dt

    
    



if __name__ == '__main__':
    main()