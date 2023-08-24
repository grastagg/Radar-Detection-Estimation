import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, initial_position, sensing_range = 900):
        self.position = initial_position #x,y,heading
        self.sensing_range = sensing_range
        self.measurement_power_values = []
        self.measurement_angle_of_arrival_values = []
        self.measurement_locations = []
        
        
        
        
        
        
    def update(self,v,u,dt,radar_list):
        self.measure_radar(radar_list)
        self.position[0] +=  v * np.cos(self.position[2]) * dt
        self.position[1] +=  v * np.sin(self.position[2]) * dt
        self.position[2] += u*dt

    def get_distance(self,p1,p2):
        return np.linalg.norm(np.array(p1)-np.array(p2))
    
    def map_angle_minus_pi_to_pi(self, angle):
        while angle < -np.pi:
            angle+= 2*np.pi
        while angle > np.pi:
            angle -= 2*np.pi
        return angle

    def measure_radar(self, radar_list):
        angle_of_arrival = None
        for radar in radar_list:
            dist = self.get_distance(radar.position, self.position[0:1])
            if dist < self.sensing_range:
                angle_between_radar_and_agent = np.arctan2(self.position[1]-radar.position[1], self.position[0]-self.position[0])
                # print("angle",angle_between_radar_and_agent)
                # print("radar angle", radar.current_angle)
                radar_angle = self.map_angle_minus_pi_to_pi(radar.current_angle)
                if angle_between_radar_and_agent < radar_angle + radar.beamwidth/2 and angle_between_radar_and_agent > radar_angle - radar.beamwidth/2:
                    print("angle",angle_between_radar_and_agent)
                    z = radar.output_power / dist**2
                    #assume angle of arrival is measured in the global frame
                    angle_of_arrival = self.map_angle_minus_pi_to_pi(radar_angle - np.pi)

                else:
                    z = None
            else:
                z = None
        if z is not None:
            self.measurement_power_values.append(z)
            self.measurement_locations.append(self.position[0:2])
            self.measurement_angle_of_arrival_values.append(angle_of_arrival)
        return z

    def plot_power_measurements(self, ax):
        if len(self.measurement_locations) > 1:
            data = np.array(self.measurement_locations)
            ax.scatter(data[:,0],data[:,1],c=self.measurement_power_values)
        
    def plot_angle_of_arrival_measurements(self, ax):
        for i,angle in enumerate(self.measurement_angle_of_arrival_values):
            start_x = self.measurement_locations
        
    def plot_agent(self,ax):
        x = self.position[0] 
        y = self.position[1] 
        circle = Circle((x,y),radius = 1,fill = False)
        ax.add_patch(circle)

        h = self.position[2]
        x_end = x + 10 * np.cos(h)
        y_end = y + 10 * np.sin(h)
        ax.plot([x, x_end], [y, y_end])
        self.plot_power_measurements(ax)

            

