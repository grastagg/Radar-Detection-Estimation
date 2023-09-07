import numpy as np
import matplotlib.pyplot as plt




from radar import RadarCircularPattern
from agent import Agent
from emmiterLocationEstimator import EmmitterLocationEstimator
from batchEmmiterLocationEstimator import BatchEmmiterLocationEstimator
from mulitple_radar_location_tracking import MultipleRadarLocationEstimator
from gp import GaussianProcess

# np.random.seed(12342)

def plot_scene(radar_list, agent_list,bounds,plt_index,emmitterLocationEstimator, gp, batchEmmiterLocationEstimator, multipleEmmiterLocationEstimator):
        fig,ax = plt.subplots()
        ax.set_xlim((0,bounds[0]))
        ax.set_ylim((0,bounds[1]))
        ax.set_aspect('equal')
        # c = gp.plot(ax)
        # if c is not None:
            # plt.colorbar(c)
        for agent in agent_list:
            agent.plot_agent(ax, multipleEmmiterLocationEstimator.inlier_mask)
        c = emmitterLocationEstimator.plot(ax, False)
        # if c is not None:
        #     plt.colorbar(c)
        batchEmmiterLocationEstimator.plot(ax)
        multipleEmmiterLocationEstimator.plot(ax)
        for radar in radar_list:
            radar.plot_view_area(ax)
        plt.savefig('images/'+str(plt_index)+'.png')
        plt.close()

def main():
    radar_list = []
    agent_list = []
    radar = RadarCircularPattern()
    radar2 = RadarCircularPattern(position=[200,200], phase=np.pi, angular_rate=2.5)
    radar3 = RadarCircularPattern(position=[1000,1000], phase=np.pi/2, angular_rate=3.5)
    radar4 = RadarCircularPattern(position=[200,1000], phase=np.pi/3, angular_rate=4)
    agent = Agent([600,10,0])
    emmitterLocationEstimator = EmmitterLocationEstimator(groundTruth=np.array([[600,600]]))
    batchEmmiterLocationEstimator = BatchEmmiterLocationEstimator(np.array([[600,600]]))
    multipleRadarLocationEstimator = MultipleRadarLocationEstimator(sensing_range=agent.sensing_range, angle_measurement_std_dev=agent.angle_measurement_std_dev)
    
    
    
    radar_list.append(radar)
    radar_list.append(radar2)
    radar_list.append(radar3)
    radar_list.append(radar4)
    agent_list.append(agent)
    bounds = (1200,1200)

    numTestPoints = 60
    
    x_test = np.linspace(0,bounds[0],numTestPoints)
    y_test = np.linspace(0,bounds[1],numTestPoints)

    X_test = []
    
    for i in range(numTestPoints):
        for j in range(numTestPoints):
            X_test.append(np.array([x_test[i],y_test[j]]))
    
    gp = GaussianProcess(X_test)
    
    t_end = 20
    dt = .1
    t_current = 0
    plt_index = 0

    current_number_of_aoa_measurements = 0

    inlier_mask = None
    
    while t_current < t_end:
        plot_scene(radar_list, agent_list, bounds, plt_index,emmitterLocationEstimator, gp, batchEmmiterLocationEstimator, multipleRadarLocationEstimator)
        for radar in radar_list:
            radar.update(dt)
        for agent in agent_list:
            agent.update(120,.25,dt,radar_list)
        if len(agent.measurement_angle_of_arrival_values) != current_number_of_aoa_measurements:
            print("adding measurement")
            current_number_of_aoa_measurements += 1
            emmitterLocationEstimator.add_measurement(agent.measurement_locations[-1], agent.measurement_angle_of_arrival_values[-1], agent.angle_measurement_std_dev**2)
            batchEmmiterLocationEstimator.add_measurement(agent.measurement_locations[-1], agent.measurement_angle_of_arrival_values[-1], agent.angle_measurement_std_dev**2)
            multipleRadarLocationEstimator.add_measurement(agent.measurement_locations[-1], agent.measurement_angle_of_arrival_values[-1], agent.angle_measurement_std_dev**2)
            
            if emmitterLocationEstimator.alreadyComputedInitialEmitterLocation:
                predictiveMean, predictiveCov = gp.gp_prediction(X_test, agent.measurement_locations, agent.measurement_power_values, emmitterLocationEstimator.xHat)
            # if len(agent.measurement_angle_of_arrival_values) == 2:
            #     xhat, cov = emmitterLocationEstimator.initial_emmitter_location_estimations(agent.measurement_locations[0], agent.measurement_angle_of_arrival_values[0], None, agent.angle_measurement_std_dev**2, agent.measurement_locations[1], agent.measurement_angle_of_arrival_values[1], None, agent.angle_measurement_std_dev**2)
            # if len(agent.measurement_angle_of_arrival_values) > 2:
            #     emmitterLocationEstimator.ekf_update(agent.measurement_locations[-1], agent.measurement_angle_of_arrival_values[-1], agent.angle_measurement_std_dev**2)

        plt_index += 1
        t_current += dt

    
    plt.figure()
    plt.plot(np.linspace(0, len(emmitterLocationEstimator.errorHistory), len(emmitterLocationEstimator.errorHistory)), emmitterLocationEstimator.errorHistory, label = "ekf")
    plt.plot(np.linspace(0, len(batchEmmiterLocationEstimator.errorHistory), len(batchEmmiterLocationEstimator.errorHistory)), batchEmmiterLocationEstimator.errorHistory, label = "batch")
    plt.legend()
    plt.show()
    



if __name__ == '__main__':
    main()