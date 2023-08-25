import numpy as np
import matplotlib.pyplot as plt




from radar import RadarCircularPattern
from agent import Agent
from emmiterLocationEstimator import EmmitterLocationEstimator


def plot_scene(radar_list, agent_list,bounds,plt_index):
        fig,ax = plt.subplots()
        ax.set_xlim((0,bounds[0]))
        ax.set_ylim((0,bounds[1]))
        ax.set_aspect('equal')
        for radar in radar_list:
            radar.plot_view_area(ax)
        for agent in agent_list:
            agent.plot_agent(ax)
        plt.savefig('images/'+str(plt_index)+'.png')
        plt.close()

def main():
    radar_list = []
    agent_list = []
    radar = RadarCircularPattern()
    agent = Agent([10,300,0])
    emmitterLocationEstimator = EmmitterLocationEstimator()
    radar_list.append(radar)
    agent_list.append(agent)
    bounds = (1200,1200)
    
    t_end = 6
    dt = .1
    t_current = 0
    plt_index = 0

    
    while t_current < t_end:
        plot_scene(radar_list, agent_list, bounds, plt_index)
        for radar in radar_list:
            radar.update(dt)
        for agent in agent_list:
            agent.update(20,0,dt,radar_list)
        if len(agent.measurement_angle_of_arrival_values) > 1:
            emmitterLocationEstimator.compute_location_from_two_aoa_measurements(agent.measurement_locations[0],agent.measurement_angle_of_arrival_values[0], agent.measurement_locations[1], agent.measurement_angle_of_arrival_values[1])
        plt_index += 1
        t_current += dt

    
    
    



if __name__ == '__main__':
    main()