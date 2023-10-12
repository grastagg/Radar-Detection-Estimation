import numpy as np
from scipy.constants import Boltzmann
import params



class ProbabilityOfDetectionMap():
    def __init__(self, X_test):
        self.X_test = np.array(X_test)
        self.pdMap = None
        self.pdCovMap = None

    
    
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
    
    def compute_probability_of_detection_at_points(self, X_test, radar, estimatedRadarParams, estimatedRadarParamsCov, agent):
        pdMap = np.zeros(len(X_test))
        pdCovMap = np.zeros(len(X_test))
        
        radarXY = estimatedRadarParams[0:2]
        for i,position in enumerate(X_test):
            distance = np.linalg.norm(radarXY-position)
            snr = self.signal_to_noise_ration(estimatedRadarParams[2], radar.recieveGain, radar.wavelength, agent.radarCrossSection, radar.pulseWidth, distance, radar.systemTemperature)
            pdMap[i] = self.probability_of_detection(radar.probabilityOfFalseAlarm, snr)
            pdCovMap[i] = self.probability_of_detection_uncertainty_single_radar_at_xy(position, radar, estimatedRadarParams, estimatedRadarParamsCov, agent)
    
        self.pdMap = pdMap
        self.pdCovMap = pdCovMap
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
    
    
    def probability_of_detection_uncertainty_single_radar_at_xy(self, position, radar, estimatedRadarParams, estimatedRadarParamsCov, agent):
        estimatedRadarParamsJacobian = self.pd_jacobian_emittor_params(position, estimatedRadarParams)

        
        return np.squeeze(estimatedRadarParamsJacobian[2:].T @ estimatedRadarParamsCov[2:,2:] @ estimatedRadarParamsJacobian[2:])
        



    def pd_jacobian_emittor_params(self,position, estimatedRadarParams):
        x = position[0]
        y = position[1]
        x_em = estimatedRadarParams[0]
        y_em = estimatedRadarParams[1]
        ERP = estimatedRadarParams[2]
        Gr = params.radarRecieveGainPriorMean
        Pfa = params.radarProbabilityOfFalseAlarmPriorMean
        rcs = params.agentRadarCrossSection
        tau_p = params.radarPulseWidthPriorMean
        wavelength = params.radarWavelengthPriorMean
        T_s = params.radarSystemTemperaturePriorMean
        k = Boltzmann
        

        d_pd_d_xem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(x-x_em))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
        d_pd_d_yem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(y-y_em))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
        d_pd_d_erp = -(Gr*np.log(Pfa)*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2*((Gr*rcs*tau_p*wavelength**2*ERP)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2)

        

        return np.array([[d_pd_d_xem],[d_pd_d_yem],[d_pd_d_erp]])
        

        
    
    def plot(self, ax):
        if self.pdMap is not None:
            x_test = self.X_test[:,0].reshape((params.numTestPoints,params.numTestPoints))
            y_test = self.X_test[:,1].reshape((params.numTestPoints,params.numTestPoints))

            c = ax.pcolormesh(x_test, y_test, self.pdMap.reshape((params.numTestPoints, params.numTestPoints)))
            return c

    def plot_cov(self, ax):
        if self.pdCovMap is not None:
            x_test = self.X_test[:,0].reshape((params.numTestPoints,params.numTestPoints))
            y_test = self.X_test[:,1].reshape((params.numTestPoints,params.numTestPoints))

            c = ax.pcolormesh(x_test, y_test, self.pdCovMap.reshape((params.numTestPoints, params.numTestPoints)))
            return c

            

    