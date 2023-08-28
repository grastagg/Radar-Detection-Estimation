import numpy as np
import matplotlib.pyplot as plt




from radar import RadarCircularPattern
from agent import Agent
from emmiterLocationEstimator import EmmitterLocationEstimator


def plot_scene(radar_list, agent_list,bounds,plt_index,emmitterLocationEstimator):
        fig,ax = plt.subplots()
        ax.set_xlim((0,bounds[0]))
        ax.set_ylim((0,bounds[1]))
        ax.set_aspect('equal')
        for radar in radar_list:
            radar.plot_view_area(ax)
        for agent in agent_list:
            agent.plot_agent(ax)
        c = emmitterLocationEstimator.plot(ax)
        if c is not None:
            plt.colorbar(c)
        plt.savefig('images/'+str(plt_index)+'.png')
        plt.close()

def main():
    radar_list = []
    agent_list = []
    radar = RadarCircularPattern()
    agent = Agent([600,10,0])
    emmitterLocationEstimator = EmmitterLocationEstimator(groundTruth=np.array([[600,600]]))
    radar_list.append(radar)
    agent_list.append(agent)
    bounds = (1200,1200)
    
    t_end = 20
    dt = .1
    t_current = 0
    plt_index = 0

    current_number_of_aoa_measurements = 0
    while t_current < t_end:
        plot_scene(radar_list, agent_list, bounds, plt_index,emmitterLocationEstimator)
        for radar in radar_list:
            radar.update(dt)
        for agent in agent_list:
            agent.update(120,.2,dt,radar_list)
        if len(agent.measurement_angle_of_arrival_values) != current_number_of_aoa_measurements:
            print("adding measurement")
            current_number_of_aoa_measurements += 1
            emmitterLocationEstimator.add_measurement(agent.measurement_locations[-1], agent.measurement_angle_of_arrival_values[-1], agent.angle_measurement_std_dev**2)
            # if len(agent.measurement_angle_of_arrival_values) == 2:
            #     xhat, cov = emmitterLocationEstimator.initial_emmitter_location_estimations(agent.measurement_locations[0], agent.measurement_angle_of_arrival_values[0], None, agent.angle_measurement_std_dev**2, agent.measurement_locations[1], agent.measurement_angle_of_arrival_values[1], None, agent.angle_measurement_std_dev**2)
            # if len(agent.measurement_angle_of_arrival_values) > 2:
            #     emmitterLocationEstimator.ekf_update(agent.measurement_locations[-1], agent.measurement_angle_of_arrival_values[-1], agent.angle_measurement_std_dev**2)

        plt_index += 1
        t_current += dt

    emmitterLocationEstimator.plot_xHatHistory()
    
    



if __name__ == '__main__':
    main()