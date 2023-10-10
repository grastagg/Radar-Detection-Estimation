import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, initialPosition, numRadar, sensingRange, powerMeasurementStdDev, angleMeasurementStdDev, elintAntenneaGain, elintSystemLoss, emittorWavelength):
        self.position = initialPosition #x,y,heading
        self.sensingRange = sensingRange
        self.measurementPowerValues = []
        self.measurementAngleOfArrivalValues = []
        self.measurementLocations = []
        self.powerMeasurementStdDev = powerMeasurementStdDev
        self.angelMeasurementStdDev = angleMeasurementStdDev

        self.truthEmitterCorrespondence = [[] for i in range(numRadar)]
        
        
        self.elintAntenneaGain = elintAntenneaGain
        self.elsintSystemLoss = elintSystemLoss
        self.emittor_signal_wavelength = emittorWavelength
        self.radarMeasurementCoeff = (self.elintAntenneaGain * self.emittor_signal_wavelength**2)/((4*np.pi)**2 * self.elsintSystemLoss)
        
        
        
        
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
        angle_of_arrivals = []
        power_measurements = []
        radar_distances = []
        radar_indecies = []
        for i,radar in enumerate(radar_list):
            dist = self.get_distance(radar.position, self.position[0:2])
            if dist < self.sensingRange:
                angle_between_radar_and_agent = np.arctan2(self.position[1]-radar.position[1], self.position[0]-radar.position[0])
                # print("angle",angle_between_radar_and_agent)
                # print("radar angle", radar.current_angle)
                radar_angle = self.map_angle_minus_pi_to_pi(radar.current_angle)
                if angle_between_radar_and_agent < radar_angle + radar.beamwidth/2 and angle_between_radar_and_agent > radar_angle - radar.beamwidth/2:
                    radar_distances.append(dist)
                    radar_indecies.append(i)
                    # print("radar angle", radar.current_angle)
                    # print("angle",angle_between_radar_and_agent)
                    power_measurements.append((self.radarMeasurementCoeff * radar.outputPower * radar.transmitGain) / dist**2) #+ np.random.normal(0,self.powerMeasurementStdDev)**2
                    #aoa measured in global frame
                    angle_of_arrivals.append(self.map_angle_minus_pi_to_pi(angle_between_radar_and_agent + np.pi + np.random.normal(0,self.angelMeasurementStdDev)))
                    # print("AOA", angle_of_arrival)

                else:
                    z = None
            else:
                z = None
        if len(angle_of_arrivals) == 1:
            self.measurementPowerValues.append(power_measurements[0])
            self.measurementLocations.append(self.position[0:2])
            self.measurementAngleOfArrivalValues.append(angle_of_arrivals[0])
            self.truthEmitterCorrespondence[radar_indecies[0]].append(len(self.measurementLocations)-1)
            # print("truth group lists",self.truthEmitterCorrespondence)
        elif len(angle_of_arrivals) > 1:
            closest_radar_index = radar_distances.index(min(radar_distances))
            self.measurementPowerValues.append(power_measurements[closest_radar_index])
            self.measurementLocations.append(self.position[0:2])
            self.measurementAngleOfArrivalValues.append(angle_of_arrivals[closest_radar_index])
            self.truthEmitterCorrespondence[radar_indecies[closest_radar_index]].append(len(self.measurementLocations)-1)
        

    def plot_power_measurements(self, ax):
        if len(self.measurementLocations) > 0:
            data = np.array(self.measurementLocations)
            ax.scatter(data[:,0],data[:,1],c=np.log10(np.array(self.measurementPowerValues)),vmin = -5, vmax =.5)
            # ax.scatter(data[:,0],data[:,1],c=np.log10(np.array(self.measurementPowerValues)))
        
    def plot_angle_of_arrival_measurements(self, ax, inlier_mask):
        color = 'g'
        for i,angle in enumerate(self.measurementAngleOfArrivalValues):
            if inlier_mask is not None:
                if not inlier_mask[i]:
                    color = 'r'
                else:
                    color = 'g'
            start_x = self.measurementLocations[i][0]
            start_y = self.measurementLocations[i][1]
            end_x = start_x + self.sensingRange * np.cos(angle)
            end_y = start_y + self.sensingRange * np.sin(angle)
            ax.plot([start_x,end_x],[start_y,end_y], c=color)
            end_x = start_x + self.sensingRange * np.cos(angle+self.angelMeasurementStdDev)
            end_y = start_y + self.sensingRange * np.sin(angle+self.angelMeasurementStdDev)
            ax.plot([start_x,end_x],[start_y,end_y],linestyle = '--',c = color)
            end_x = start_x + self.sensingRange * np.cos(angle-self.angelMeasurementStdDev)
            end_y = start_y + self.sensingRange * np.sin(angle-self.angelMeasurementStdDev)
            ax.plot([start_x,end_x],[start_y,end_y],linestyle = '--',c=color)
        
    def plot_agent(self,ax, inlier_mask = None):
        x = self.position[0] 
        y = self.position[1] 
        circle = Circle((x,y),radius = 1,fill = False)
        ax.add_patch(circle)

        h = self.position[2]
        x_end = x + 10 * np.cos(h)
        y_end = y + 10 * np.sin(h)
        ax.plot([x, x_end], [y, y_end])
        self.plot_power_measurements(ax)
        # self.plot_angle_of_arrival_measurements(ax, inlier_mask)

            

