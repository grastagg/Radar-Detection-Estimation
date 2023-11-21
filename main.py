import numpy as np
import matplotlib.pyplot as plt
import cProfile
from pstats import Stats
import matplotlib.style as mplstyle
mplstyle.use('fast')
# matplotlib.use('Agg')
import time


from radar import RadarCircularPattern
from agent import Agent
from multipleEmitterOnlineLocationAndPowerEstimator import MultipleEmitterOnlineLocationAndPowerEstimator
from probabilityOfDetectionMap import ProbabilityOfDetectionMap
import params

# np.random.seed(12342)

fig,ax = plt.subplots()
ax.set_xlim((0,params.bounds[0]))
ax.set_ylim((0,params.bounds[1]))
ax.set_aspect('equal')
def plot_scene(radarList, agentList,bounds,plotIndex, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap):
    c = None


    numMeasurements = 0
    for agent in agentList:
        agent.plot_agent(ax)
        numMeasurements += len(agent.measurementPowerValues)

    for radar in radarList:
        radar.plot_view_area(ax)

    if multipleEmitterOnlineLocationAndPowerEstimator is not None:
        measurement_lines = multipleEmitterOnlineLocationAndPowerEstimator.plot(ax)


    if probabilityOfDetectionMap is not None:
        c = probabilityOfDetectionMap.plot_best_measurement_map(ax)
    cb = None
    if c is not None:
        cb = plt.colorbar(c)
    

    plt.title(numMeasurements-1)
    plt.savefig('images/best_measurement/'+str(plotIndex)+'.png')

    if c is not None:
        cb.remove()
        c.remove()
        cb = None
        c = None

    # start3 = time.time()
    if probabilityOfDetectionMap is not None:
        c = probabilityOfDetectionMap.plot_cov(ax)
    # print("cov:", time.time() - start3)
    if c is not None:
        cb = plt.colorbar(c)
        
    # for radar in radarList:
    #     radar.plot_view_area(ax)
    plt.title(numMeasurements-1)
    plt.savefig('images/pd_cov/'+str(plotIndex)+'.png')

    if c is not None:
        cb.remove()
        c.remove()
        cb = None
        c = None

    # start3 = time.time()
    if probabilityOfDetectionMap is not None:
        c = probabilityOfDetectionMap.plot_mean(ax)
    # print("mean:", time.time() - start3)
    if c is not None:
        cb = plt.colorbar(c)
        
    # for radar in radarList:
    #     radar.plot_view_area(ax)
    plt.title(numMeasurements-1)
    plt.savefig('images/pd_mean/'+str(plotIndex)+'.png')
    if c is not None:
        cb.remove()
        c.remove()
        cb = None
        c = None

    for line in measurement_lines:
        line.remove()


def main():
    bounds = params.bounds 
    numTestPoints = params.numTestPoints
    X_test = params.create_test_points(numTestPoints, bounds)
    


    radarList = params.radarList
    agentList = params.agentList





    multipleEmitterOnlineLocationAndPowerEstimator = MultipleEmitterOnlineLocationAndPowerEstimator(sensing_range=params.agentSensingRange, angle_measurement_std_dev=params.agentAngleMeasurementStdDev, measurement_cov=params.measurementCov, X_test=X_test, radar_measurement_coeff=params.radarMeasurementCoeff)
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
            # agent.update(134,.025,dt,radarList)
            agent.update(134,.0,dt,radarList)
            if len(agent.measurementAngleOfArrivalValues) != currentNumberOfMeasurementsArray[i]:
                print("adding measurement:", currentNumberOfMeasurementsArray[i], "from agent",i)
                print("measurement:", currentNumberOfMeasurements)
                print("truth group lists",agent.truthEmitterCorrespondence)
                currentNumberOfMeasurementsArray[i] += 1
                multipleEmitterOnlineLocationAndPowerEstimator.add_measurement(agent.measurementLocations[-1], [agent.measurementAngleOfArrivalValues[-1], agent.measurementPowerValues[-1]])
                currentNumberOfMeasurements += 1
                if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
                    probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances)



        plotIndex += 1
        tCurrent += dt

    
    



if __name__ == '__main__':
    do_profile = False 
    if do_profile:
        with cProfile.Profile() as pr:
            main()
        with open('profile_state.txt','w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
            
    main()