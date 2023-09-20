import numpy as np
import matplotlib.pyplot as plt



class RadarCircularPattern:
    def __init__(self, position = (600,600), range=600, beamwidth = 15 * np.pi/180, angular_rate = 3, phase = 0, output_power = 1000):
        '''
        position: location of radar transmitter and reciever 
        range: distance at which radar can detect agent
        beamwidth: angular width of radar beam where agent can be detected (radians)
        angular_rate: rate at which radar spins (rad/sec)
        '''
        self.position = position
        self.range = range
        # self.power_output = power_output
        self.beamwidth = beamwidth
        self.angular_rate = angular_rate
        self.output_power = output_power

        
        self.current_angle = phase
        
        
    def update(self, dt):
        self.current_angle += self.angular_rate * dt
    
    
    def plot_arc_length(self, ax):
        start_theta = self.current_angle - self.beamwidth/2
        end_theta = self.current_angle + self.beamwidth/2

        theta = np.linspace(start_theta, end_theta, 100)
        x = self.position[0] + self.range * np.cos(theta)
        y = self.position[1] + self.range * np.sin(theta)

        ax.plot(x,y)

    def plot_end_bounds(self,ax):
        x_left = self.position[0] + self.range * np.cos(self.current_angle - self.beamwidth/2)
        y_left = self.position[1] + self.range * np.sin(self.current_angle - self.beamwidth/2)

        x_right = self.position[0] + self.range * np.cos(self.current_angle + self.beamwidth/2)
        y_right = self.position[1] + self.range * np.sin(self.current_angle + self.beamwidth/2)

        x_middle = self.position[0]
        y_middle = self.position[1]

        ax.plot([x_left,x_middle,x_right],[y_left,y_middle,y_right])

    def plot_radar_location(self, ax):
        plt.scatter(self.position[0],self.position[1], marker='*')

    def plot_view_area(self, ax):
        self.plot_arc_length(ax)
        self.plot_end_bounds(ax)
        self.plot_radar_location(ax)
        
        
