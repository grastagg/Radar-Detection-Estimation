import numpy as np
from scipy.constants import Boltzmann
import params
import jax.numpy as jnp
from jax import jacfwd

class ProbabilityOfDetectionMap():
    def __init__(self, X_test):
        self.X_test = np.array(X_test)
        self.pdMap = None
        self.pdCovMap = None

    
    
    def probability_of_detection(self, probabilityOfFalseAlarm, snr):
        return np.exp((np.log(probabilityOfFalseAlarm))/(snr + 1))
    
    def signal_to_noise_ration(self, effectiveRadarPower, radarRecieverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
        #antennea gain not i decibels
        return (effectiveRadarPower*radarRecieverGain*wavelength**2*radarCrossSection*radarPulseWidth)/((4*np.pi)**3*distance**4*Boltzmann*radarSystemTemperature)
        
    def compute_probability_of_detection_at_xy(self, radar, position, estimatedRadarParams):
        radarXY = estimatedRadarParams[0:1]
        distance = np.linalg.norm(radarXY-position)
        snr = self.signal_to_noise_ration(estimatedRadarParams[2], radar.recieveGain, radar.wavelength, params.agentRadarCrossSection, radar.pulseWidth, distance, params.radarSystemTemperature)
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
        radarParametersJacobian = self.pd_jacobian_unkown_radar_parameters(position, estimatedRadarParams)

        
        return np.squeeze(estimatedRadarParamsJacobian.T @ estimatedRadarParamsCov@estimatedRadarParamsJacobian)
        



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
        

        # d_pd_d_xem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(x-x_em))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
        # d_pd_d_yem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(y-y_em))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
        # d_pd_d_erp = -(Gr*np.log(Pfa)*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2*((Gr*rcs*tau_p*wavelength**2*ERP)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2)
        d_pd_d_erp = -(Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*np.exp(np.log(Pfa)/((Gr*rcs*tau_p*wavelength**2*ERP)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)))/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2*((Gr*rcs*tau_p*wavelength**2*ERP)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2)
        d_pd_d_xem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(x-x_em)*np.exp(np.log(Pfa)/((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
        d_pd_d_yem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(y-y_em)*np.exp(np.log(Pfa)/((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2*((y-y_em)**2+(x-x_em)**2)**3)
        return np.array([[d_pd_d_xem],[d_pd_d_yem],[d_pd_d_erp]])
        # return np.array([[jac[0]],[jac[1]],[jac[2]]])
    
    def pd_jacobian_unkown_radar_parameters(self, position, estimatedRadarParams):
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

        parameters = [Gr, wavelength, rcs, tau_p, T_s, Pfa]
        # jac = jacfwd(self.jax_pd, 2)(estimatedRadarParams,position, parameters)
        SNR = self.signal_to_noise_ration(ERP, Gr, wavelength, rcs, tau_p, np.sqrt((x-x_em)**2+(y-y_em)**2),T_s)

        d_pd_d_Gr = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (ERP*wavelength**2*rcs*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_wavelength = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*2*wavelength*rcs*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_rcs = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_tau_p = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*rcs)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_T_s = np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*rcs*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s**2)
        d_pd_d_Pfa = 1/(SNR+1) * np.exp(np.log(Pfa)/(SNR+1)) * 1/Pfa


        
    
    def jax_pd(self, estimatedRadaraParameters, position, parameters):
        ERP = estimatedRadaraParameters[2]
        x_em = estimatedRadaraParameters[0]
        y_em = estimatedRadaraParameters[1]
        x = position[0]
        y = position[1]
        Gr = parameters[0]
        wavelength = parameters[1]
        rcs = parameters[2]
        tua_p = parameters[3]
        Ts = parameters[4]
        Pfa = parameters[5]
        
        distance = jnp.sqrt((x-x_em)**2+(y-y_em)**2)
        snr = self.signal_to_noise_ration(ERP, Gr, wavelength, rcs, tua_p, distance, Ts)
        return jnp.exp((jnp.log(Pfa))/(snr + 1))

    # def jax_pd(self, estimatedRadaraParameters, position):
    #     ERP = estimatedRadaraParameters[2]
    #     x_em = estimatedRadaraParameters[0]
    #     y_em = estimatedRadaraParameters[1]
    #     x = position[0]
    #     y = position[1]
    #     distance = jnp.sqrt((x-x_em)**2+(y-y_em)**2)
    #     # snr = (ERP*params.radarRecieveGain*params.radarWavelength**2*params.agentRadarCrossSection*params.radarPulseWidth)/((4*jnp.pi)**3*distance**4*Boltzmann*params.radarSystemTemperature)
    #     snr = self.signal_to_noise_ration(ERP, params.radarRecieveGain, params.radarWavelength, params.agentRadarCrossSection, params.radarPulseWidth, distance, params.radarSystemTemperature)
    #     print("snr",snr)
    #     return jnp.exp((jnp.log(params.radarProbabilityOfFalseAlarm))/(snr + 1))
    def f(self, x, radar, position):
        return np.array([self.compute_probability_of_detection_at_xy(radar, position, x)])

    def get_gradient_finite_diff(self,f,x,h,radar, position):
        #calculate gradient using finite differencing
        x = np.array(x)

        #store the function value at x
        fx = f(x,radar,position)

        #initialize the jacobian matrix (# functions by # variables
        grad = np.zeros((len(fx),len(x)))

        #this loops through each column of the Jacobian
        for i in range(len(x)):
            #the next three lines creates a step vector where all elements are zeros except for the current step direction
            epsilon = np.zeros(len(x))
            step = h * (1 + abs(x[i]))
            epsilon[i] = step

            #add the step to the x vector
            xi = x + epsilon

            grad[:,i] = (f(xi,radar, position) - fx)/step
            # print("grad[:,i]",grad[:,i])

        return grad
        
    
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

            

    