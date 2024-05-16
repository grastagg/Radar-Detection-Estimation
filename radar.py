import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
from matplotlib.lines import Line2D



class RadarCircularPattern:
    def __init__(self, position = (600,600), range=4000, beamwidth = 15 * np.pi/180, angular_rate = 3, phase = 0, outputPower = 1000, transmitGain = 1, recieveGain = 1, wavelength = 0.001, pulseWidth = 1.1e-5, systemTemperature = 740, probabilityOfFalseAlarm = 1e-6):
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
        self.outputPower = outputPower
        self.transmitGain = transmitGain
        self.recieveGain = recieveGain
        self.wavelength = wavelength
        self.pulseWidth = pulseWidth
        self.systemTemperature = systemTemperature
        self.probabilityOfFalseAlarm = probabilityOfFalseAlarm

        
        self.current_angle = phase

        self.firstPlot = True
        self.plotArc = Arc((self.position[0], self.position[1]), 2*self.range, 2*self.range,angle = self.current_angle, theta1=-self.beamwidth/2*180/np.pi, theta2=self.beamwidth/2*180/np.pi,zorder = 100,color='r')

        x_left = self.position[0] + self.range * np.cos(self.current_angle - self.beamwidth/2)
        y_left = self.position[1] + self.range * np.sin(self.current_angle - self.beamwidth/2)

        x_right = self.position[0] + self.range * np.cos(self.current_angle + self.beamwidth/2)
        y_right = self.position[1] + self.range * np.sin(self.current_angle + self.beamwidth/2)

        x_middle = self.position[0]
        y_middle = self.position[1]
        
        # self.plotLine = Line2D(xdata=[x_left, x_middle, x_right], ydata=[y_left, y_middle, y_right])
        self.plotPoly = Polygon(xy = np.array([[x_left,y_left],[x_middle, y_middle],[x_right,y_right]]), closed=False, fill = False,zorder = 101, color='r')
        
        
    def update(self, dt):
        self.current_angle += self.angular_rate * dt
    
    
    def plot_arc_length(self, ax):
        if self.firstPlot:
            ax.add_patch(self.plotArc)
        else:
            self.plotArc.set_angle(self.current_angle*180/np.pi)
        # start_theta = self.current_angle - self.beamwidth/2
        # end_theta = self.current_angle + self.beamwidth/2

        # theta = np.linspace(start_theta, end_theta, 100)
        # x = self.position[0] + self.range * np.cos(theta)
        # y = self.position[1] + self.range * np.sin(theta)

        # ax.plot(x,y,zorder = 102)

    def plot_end_bounds(self,ax):
        if self.firstPlot:
            ax.add_patch(self.plotPoly)
        else:
            x_left = self.position[0] + self.range * np.cos(self.current_angle - self.beamwidth/2)
            y_left = self.position[1] + self.range * np.sin(self.current_angle - self.beamwidth/2)

            x_right = self.position[0] + self.range * np.cos(self.current_angle + self.beamwidth/2)
            y_right = self.position[1] + self.range * np.sin(self.current_angle + self.beamwidth/2)

            x_middle = self.position[0]
            y_middle = self.position[1]

            
            
            self.plotPoly.set_xy(np.array([[x_left,y_left],[x_middle, y_middle],[x_right,y_right]]))


        # ax.plot([x_left,x_middle,x_right],[y_left,y_middle,y_right],zorder = 101)

    def plot_radar_location(self, ax):
        plt.scatter(self.position[0],self.position[1], marker='*',zorder = 100, color = 'r')

    def plot_view_area(self, ax):
        self.plot_arc_length(ax)
        self.plot_end_bounds(ax)
        if self.firstPlot:
            self.plot_radar_location(ax)
            self.firstPlot = False
        
        
