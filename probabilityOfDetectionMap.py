import numpy as np
from scipy.constants import Boltzmann
import params



class ProbabilityOfDetectionMap():
    def __init__(self, X_test):
        self.X_test = np.array(X_test)
        self.pdMap = None

    
    
    def probability_of_detection(self, probabilityOfFalseAlarm, snr):
        return np.exp((np.log(probabilityOfFalseAlarm))/(snr + 1))
    
    def signal_to_noise_ration(self, effectiveRadarPower, radarRecieverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
        #antennea gain not i decibels
        return (effectiveRadarPower*radarRecieverGain*wavelength**2*radarCrossSection*radarPulseWidth)/((4*np.pi)**3*distance**4*Boltzmann*params.radarSystemTemperature)
        
    def compute_probability_of_detection_at_xy(self, radar, position, estimatedRadarParams):
        radarXY = estimatedRadarParams[0:1]
        distance = np.linalg.norm(radarXY-position)
        snr = self.signal_to_noise_ration(estimatedRadarParams[2], radar.recieveGain, radar.wavelength, params.radarCrossSection, radar.pulseWidth, distance, radar.sytemTemperature)
        return self.probability_of_detection(radar.probabilityOfFalseAlarm, snr)
    
    def compute_probability_of_detection_at_points(self, X_test, radar, estimatedRadarParams, agent):
        print("TEST", estimatedRadarParams)
        pdMap = np.zeros(len(X_test))
        
        radarXY = estimatedRadarParams[0:2]
        for i,position in enumerate(X_test):
            distance = np.linalg.norm(radarXY-position)
            snr = self.signal_to_noise_ration(estimatedRadarParams[2], radar.recieveGain, radar.wavelength, agent.radarCrossSection, radar.pulseWidth, distance, radar.systemTemperature)
            pdMap[i] = self.probability_of_detection(radar.probabilityOfFalseAlarm, snr)
    
        self.pdMap = pdMap
        return pdMap

    def compute_probability_of_detection_at_points_multiple_radar(self, X_test, radar, estimatedRadarParamsList, agent):
        probabilityOfNoDetection = np.ones(len(X_test))
        
        for j, estimatedRadarParams in enumerate(estimatedRadarParamsList):
            radarXY = estimatedRadarParams[0:2]
            for i,position in enumerate(X_test):
                distance = np.linalg.norm(radarXY-position)
                snr = self.signal_to_noise_ration(estimatedRadarParams[2], radar.recieveGain, radar.wavelength, agent.radarCrossSection, radar.pulseWidth, distance, radar.systemTemperature)
                probabilityOfNoDetection[i] *= (1-self.probability_of_detection(radar.probabilityOfFalseAlarm, snr))
    
        self.pdMap = 1 - probabilityOfNoDetection 
        return self.pdMap

        
    
    def plot(self, ax):
        if self.pdMap is not None:
            x_test = self.X_test[:,0].reshape((params.numTestPoints,params.numTestPoints))
            y_test = self.X_test[:,1].reshape((params.numTestPoints,params.numTestPoints))

            c = ax.pcolormesh(x_test, y_test, self.pdMap.reshape((params.numTestPoints, params.numTestPoints)))
            return c

            

    