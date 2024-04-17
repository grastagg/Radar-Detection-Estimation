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
from pathPlanning import SplinePathPlanningLowPriority 


from main_helper import create_agent_list, create_radar_list

# np.random.seed(1234)

fig,ax = plt.subplots()
ax.set_xlim((0,params.bounds[0]))
ax.set_ylim((0,params.bounds[1]))
ax.set_aspect('equal')

def plot_scene(radarList, agentList,bounds,plotIndex, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap, lowPriorityPathPlanner):
    c = None


    numMeasurements = 0
    for agent in agentList:
        agent.plot_agent(ax)
        numMeasurements += len(agent.measurementPowerValues)

    for radar in radarList:
        radar.plot_view_area(ax)

    if multipleEmitterOnlineLocationAndPowerEstimator is not None:
        measurement_lines,estimator_locs = multipleEmitterOnlineLocationAndPowerEstimator.plot(ax)
        plt.savefig('images/temp/'+str(plotIndex)+'.png')
    

    if params.plotBestMeasurement:
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

    if params.plotPdCov:
        if probabilityOfDetectionMap is not None:
            c = probabilityOfDetectionMap.plot_cov(ax)
        if c is not None:
            cb = plt.colorbar(c)
            
        plt.title(numMeasurements-1)
        plt.savefig('images/pd_cov/'+str(plotIndex)+'.png')

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
            
        plt.title(numMeasurements-1)
        plt.savefig('images/pd_mean/'+str(plotIndex)+'.png')
        if c is not None:
            cb.remove()
            c.remove()
            cb = None
            c = None
    
    for i,agent in enumerate(agentList):
        if params.plotObjectiveFunction:
            if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
                c = lowPriorityPathPlanner.plot_objective_and_constraint(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements,agentList,i)
                # c = lowPriorityPathPlanner.plot_chance_constraints(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements)
                # proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections]
                # ax.legend(proxy, ['safe', 'unsafe'])
                cb = plt.colorbar(c)
            plt.title(numMeasurements-1)
            plt.savefig('images/objective_function/'+str(i)+'/'+str(plotIndex)+'.png')
            if c is not None:
                if cb is not None:
                    cb.remove()
                c.remove()
                cb = None
                c = None

    if params.plotChanceConstraints:
        if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
            c = lowPriorityPathPlanner.plot_chance_constraints(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements)
            # c = lowPriorityPathPlanner.plot_chance_constraints(ax, probabilityOfDetectionMap.X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList[0].position, numMeasurements)
            # proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in c.collections]
            # ax.legend(proxy, ['safe', 'unsafe'])
            cb = plt.colorbar(c[-1])
        plt.title(numMeasurements-1)
        plt.savefig('images/chance_constraints/'+str(plotIndex)+'.png')
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


def main():
    bounds = params.bounds 
    numTestPoints = params.numTestPoints
    X_test = params.create_test_points(numTestPoints, bounds)
    

    radarList = create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm)
    agentList = create_agent_list(params.agentInitialStates, len(radarList), params.agentSensingRange, params.agentPowerMeasurementStdDev, params.agentAngleMeasurementStdDev, params.agentELINTAnteneaGain, params.agentELINTSystemLoss, params.radarWavelength, params.agentRadarCrossSection)

    # radarList = params.radarList
    # agentList = params.agentList





    multipleEmitterOnlineLocationAndPowerEstimator = MultipleEmitterOnlineLocationAndPowerEstimator(sensing_range=params.agentSensingRange, angle_measurement_std_dev=params.agentAngleMeasurementStdDev, measurement_cov=params.measurementCov, X_test=X_test, radar_measurement_coeff=params.radarMeasurementCoeff)
    probabilityOfDetectionMap = ProbabilityOfDetectionMap(X_test, radarList)
    lowPriorityPathPlanner = SplinePathPlanningLowPriority()



    


    
    tEnd = params.simulationEndTime
    dt = params.simulationTimestep
    tCurrent = 0
    plotIndex = 0

    currentNumberOfMeasurements = 0
    currentNumberOfMeasurementsArray = np.zeros(len(agentList),dtype=int)

    timeSinceLastPlot = 0

    

    while tCurrent < tEnd:
        start = time.time()
        if timeSinceLastPlot >= params.plotTimeStep:
            plot_scene(radarList, agentList, bounds, plotIndex, multipleEmitterOnlineLocationAndPowerEstimator, probabilityOfDetectionMap, lowPriorityPathPlanner)
            timeSinceLastPlot = 0
            plotIndex += 1
            # print("plot time", time.time()-start)
        for radar in radarList:
            radar.update(dt)
        
        for i,agent in enumerate(agentList):
            # agent.update(134,.0,dt,radarList)
            turnRate, velocity = lowPriorityPathPlanner.get_control(dt,agent.position,i)
            agent.update(velocity,turnRate,dt,radarList,currentNumberOfMeasurements)
            if len(agent.measurementAngleOfArrivalValues) != currentNumberOfMeasurementsArray[i]:
                print("adding measurement:", currentNumberOfMeasurementsArray[i], "from agent",i)
                print("measurement:", currentNumberOfMeasurements)
                print("truth group lists",agent.truthEmitterCorrespondence)
                currentNumberOfMeasurementsArray[i] += 1
                start_e = time.time()
                multipleEmitterOnlineLocationAndPowerEstimator.add_measurement(agent.measurementLocations[-1], [agent.measurementAngleOfArrivalValues[-1], agent.measurementPowerValues[-1]])
                # print("estimator time", time.time()-start_e)
                currentNumberOfMeasurements += 1
                if len(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params) > 0:
                    start_pd = time.time()
                    probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances)
                    # print("pd map time", time.time()-start_pd)
        start_p = time.time()
        lowPriorityPathPlanner.update_path(multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances, probabilityOfDetectionMap, agentList, len(agent.measurementAngleOfArrivalValues),multipleEmitterOnlineLocationAndPowerEstimator.measurement_locations )
            # print("path planning time", time.time()-start_p)
        # print("time for step:", time.time()-start)



        timeSinceLastPlot+=dt
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
