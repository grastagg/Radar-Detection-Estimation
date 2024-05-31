import numpy as np
from scipy.constants import Boltzmann
import params
import jax.numpy as jnp
from jax import jacfwd
from numpy.random import multivariate_normal
from params import measurement_jacobian

class ProbabilityOfDetectionMap():
    def __init__(self, X_test,radarList):
        self.X_test = np.array(X_test)
        self.pdMap = None
        self.pdCovMap = None
        self.estimatedRadarParamsList = None
        self.estimatedRadarParamsCovList = None

        # self.groundTruthpdMap = self.ground_truth_probability_of_detection(X_test, params.radarList) 
        self.groundTruthpdMap = self.ground_truth_probability_of_detection(X_test, radarList) 

    
    
    def probability_of_detection(self, probabilityOfFalseAlarm, snr):
        return np.exp((np.log(probabilityOfFalseAlarm))/(snr + 1))
    
    def signal_to_noise_ration(self, effectiveRadarPower, radarRecieverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
        #antennea gain not i decibels
        return (effectiveRadarPower*radarRecieverGain*wavelength**2*radarCrossSection*radarPulseWidth)/((4*np.pi)**3*distance**4*Boltzmann*radarSystemTemperature)
        
    def compute_probability_of_detection_at_xy(self, position, estimatedRadarParams):
        radarXY = estimatedRadarParams[0:2]
        distance = np.linalg.norm(radarXY-position)
        snr = self.signal_to_noise_ration(estimatedRadarParams[2], params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, distance, params.radarSystemTemperaturePriorMean)
        if snr<0:
            print("STOP")
        return np.array([self.probability_of_detection(params.radarProbabilityOfFalseAlarmPriorMean, snr)])
    
    def compute_probability_of_detection_at_points(self, X_test, estimatedRadarParams, estimatedRadarParamsCov):
        pdMap = np.zeros(len(X_test))
        pdCovMap = np.zeros(len(X_test))
        
        radarXY = estimatedRadarParams[0:2]
        for i,position in enumerate(X_test):
            distance = np.linalg.norm(radarXY-position)
            snr = self.signal_to_noise_ration(estimatedRadarParams[2], params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidthPriorMean, distance, params.radarSystemTemperaturePriorMean)
            pdMap[i] = self.probability_of_detection(params.radarProbabilityOfFalseAlarm, snr)
            pdCovMap[i] = self.probability_of_detection_uncertainty_single_radar_at_xy(position, estimatedRadarParams, estimatedRadarParamsCov)
    
        self.pdMap = pdMap
        self.pdCovMap = pdCovMap
        return pdMap, pdCovMap

    def compute_probability_of_detection_at_points_multiple_radar(self, X_test, estimatedRadarParamsList, estimatedRadarParamsCovList, updateMap = True):
        pdCovMap = np.zeros(len(X_test))
        probabilityOfNoDetection = np.ones(len(X_test))
        
        self.estimatedRadarParamsList = estimatedRadarParamsList
        self.estimatedRadarParamsCovList = estimatedRadarParamsCovList

        for i,position in enumerate(X_test):
            pd_list = []
            pd_cov_list = []
            for j, estimatedRadarParams in enumerate(estimatedRadarParamsList):
                # radarXY = estimatedRadarParams[0:2]
                # distance = np.linalg.norm(radarXY-position)
                # snr = self.signal_to_noise_ration(10**(estimatedRadarParams[2]/10), params.radarRecieveGain, params.radarWavelength, params.agentRadarCrossSection, params.radarPulseWidth, distance, params.radarSystemTemperature)
                # pdi = self.probability_of_detection(params.radarProbabilityOfFalseAlarm, snr)
                pdi = self.compute_probability_of_detection_at_xy(position, estimatedRadarParams)
                probabilityOfNoDetection[i] *= (1-pdi)
                # dpdi_dparams = self.probability_of_detection_uncertainty_single_radar_at_xy_bootstrap(position, estimatedRadarParams, estimatedRadarParamsCovList[j])
                dpdi_dparams = self.probability_of_detection_uncertainty_single_radar_at_xy(position, estimatedRadarParams, estimatedRadarParamsCovList[j])
                pd_cov_list.append(dpdi_dparams)
                pd_list.append(pdi)
                
            for ind1 in range(len(pd_list)):
                dpdt_dpdind1 = 1
                for ind2 in range(len(pd_list)):
                    if ind1 != ind2:
                        dpdt_dpdind1 *= (1-pd_list[ind2])
                pdCovMap[i] += dpdt_dpdind1**2*pd_cov_list[ind1]


        if updateMap:
            self.pdCovMap = pdCovMap
            self.pdMap = 1 - probabilityOfNoDetection
        
        
        
        return 1-probabilityOfNoDetection, pdCovMap 

    def get_gradient_finite_diff(self, f,x,h, position):
        #calculate gradient using finite differencing
        x = np.array(x)

        #store the function value at x
        fx = f(position, x)

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

            grad[:,i] = ((f(position, xi) - fx)/step).flatten()
            # print("grad[:,i]",grad[:,i])

        return grad
    
    # def probability_of_detection_uncertainty_multiple_radar_at_xy(position, estimatedRadarParamsList, estimatedRadarParamsCovList):
    def multi_radar_jacobian_test_x(self, pdList):
        out = 1
        for pd in pdList:
            out *= (1-pd)
        return 1-out
    
    
    def probability_of_detection_uncertainty_single_radar_at_xy(self, position, estimatedRadarParams, estimatedRadarParamsCov):
        estimatedRadarParamsJacobian = self.pd_jacobian_emittor_params(position, estimatedRadarParams)
        radarParametersJacobian = self.pd_jacobian_unkown_radar_parameters(position, estimatedRadarParams)
        # radarParametersCovariance = np.zeros((radarParametersJacobian.shape[0],radarParametersJacobian.shape[0]))
        radarParametersCovariance = np.diag([params.radarRecieveGainPriorVariance, params.radarWavelengthPriorVariance, 0, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorVariance])

        
        return np.squeeze(estimatedRadarParamsJacobian @ estimatedRadarParamsCov@estimatedRadarParamsJacobian.T + radarParametersJacobian @ radarParametersCovariance @ radarParametersJacobian.T)
        
    def probability_of_detection_uncertainty_single_radar_at_xy_bootstrap(self, position, estimatedRadarParams, estimatedRadarParamsCov):
        num_samples = 500
        samples = multivariate_normal(estimatedRadarParams, estimatedRadarParamsCov, num_samples)
        pd_samples = np.zeros(num_samples)
        for i in range(num_samples):
            pd_samples[i] = self.compute_probability_of_detection_at_xy(position, samples[i,:])
        
        # print(pd_samples.var())
        # print("TEST")
        return pd_samples.var()



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
        d_pd_d_erp = -(Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*np.exp(np.log(Pfa)/((Gr*rcs*tau_p*wavelength**2*ERP)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)))/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2*((Gr*rcs*tau_p*wavelength**2*ERP)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2)
        d_pd_d_xem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(x-x_em)*np.exp(np.log(Pfa)/((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
        d_pd_d_yem = -(ERP*Gr*np.log(Pfa)*rcs*tau_p*wavelength**2*(y-y_em)*np.exp(np.log(Pfa)/((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)))/(16*np.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*np.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2*((y-y_em)**2+(x-x_em)**2)**3)
        return np.array([d_pd_d_xem,d_pd_d_yem,d_pd_d_erp])

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
        SNR = self.signal_to_noise_ration(ERP, Gr, wavelength, rcs, tau_p, np.sqrt((x-x_em)**2+(y-y_em)**2),T_s)

        d_pd_d_Gr = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (ERP*wavelength**2*rcs*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_wavelength = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*2*wavelength*rcs*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_rcs = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_tau_p = -np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*rcs)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
        d_pd_d_T_s = np.exp((np.log(Pfa)/(SNR+1))) * (np.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*rcs*tau_p)/((4*np.pi)**3*np.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s**2)
        d_pd_d_Pfa = 1/(SNR+1) * np.exp(np.log(Pfa)/(SNR+1)) * 1/Pfa

        return np.array([d_pd_d_Gr, d_pd_d_wavelength, d_pd_d_rcs, d_pd_d_tau_p, d_pd_d_T_s, d_pd_d_Pfa])


    
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

    # def get_gradient_finite_diff(self,f,x,h,radar, position):
    #     #calculate gradient using finite differencing
    #     x = np.array(x)

    #     #store the function value at x
    #     fx = f(x,radar,position)

    #     #initialize the jacobian matrix (# functions by # variables
    #     grad = np.zeros((len(fx),len(x)))

    #     #this loops through each column of the Jacobian
    #     for i in range(len(x)):
    #         #the next three lines creates a step vector where all elements are zeros except for the current step direction
    #         epsilon = np.zeros(len(x))
    #         step = h * (1 + abs(x[i]))
    #         epsilon[i] = step

    #         #add the step to the x vector
    #         xi = x + epsilon

    #         grad[:,i] = (f(xi,radar, position) - fx)/step
    #         # print("grad[:,i]",grad[:,i])

    #     return grad

    def measurement_jacobian(self, xem, yem, erp, x, y):
        return measurement_jacobian(xem, yem, erp, x, y)

        # d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        # d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        # d_h1_d_p_emmitter = 0

        # d_h2_d_x_emmitter = -(2*erp*params.radarMeasurementCoeff*(xem-x))/((xem-x)**2+(yem-y)**2)**2 
        # d_h2_d_y_emmitter = -(2*erp*params.radarMeasurementCoeff*(yem-y))/((yem-y)**2+(xem-x)**2)**2 
        # d_h2_d_p_emmitter = params.radarMeasurementCoeff/((yem-y)**2+(xem-x)**2)

        # return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    def next_measurement_covariance_determinant(self, pos, estimatedRadarParams, estimatedRadarCovariance):
        x = pos[0]
        y = pos[1]
        x_em = estimatedRadarParams[0]
        y_em = estimatedRadarParams[1]
        erp = estimatedRadarParams[2]
        H = self.measurement_jacobian(x_em, y_em, erp, x, y)
        R = params.measurementCov
        K = estimatedRadarCovariance @ H.T @ np.linalg.inv(H@estimatedRadarCovariance@H.T + R)
        nextCovariance = (np.eye(3) - K@H)@estimatedRadarCovariance
        # nextCovariance = estimatedRadarCovariance - estimatedRadarCovariance@H.T@np.linalg.inv(H@estimatedRadarCovariance@H.T+R)@H@estimatedRadarCovariance
        # return 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(estimatedRadarCovariance)) - 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(nextCovariance))
        return np.linalg.det(estimatedRadarCovariance) - np.linalg.det(nextCovariance)
    
    def create_best_measurement_location_map(self, estimatedRadarParams_list, estimatedRadarCovariance_list):
        best_measurement_location_map = np.zeros(len(self.X_test))
        max_entropy = -1

        for j,estimatedRadarParams in enumerate(estimatedRadarParams_list):
            for i,pos in enumerate(self.X_test):
                entropy = self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarCovariance_list[j])
                best_measurement_location_map[i] += entropy
                if entropy > max_entropy:
                    max_entropy = entropy
            
        return best_measurement_location_map/max_entropy
        
    
    def plot_mean(self, ax, plotGroundTruth = False):
        if plotGroundTruth:
            x_test = self.X_test[:,0].reshape((params.numTestPoints,params.numTestPoints))
            y_test = self.X_test[:,1].reshape((params.numTestPoints,params.numTestPoints))


            c = ax.pcolormesh(x_test, y_test, self.groundTruthpdMap.reshape((params.numTestPoints, params.numTestPoints)))
            
        else:
            if self.pdMap is not None:
                x_test = self.X_test[:,0].reshape((params.numTestPoints,params.numTestPoints))
                y_test = self.X_test[:,1].reshape((params.numTestPoints,params.numTestPoints))

                # plotMap = np.square(np.subtract(self.pdMap, self.groundTruthpdMap))

                # c = ax.pcolormesh(x_test, y_test, self.groundTruthpdMap.reshape((params.numTestPoints, params.numTestPoints)))
                c = ax.pcolormesh(x_test, y_test, self.pdMap.reshape((params.numTestPoints, params.numTestPoints)))
                # c = ax.pcolormesh(x_test, y_test, plotMap.reshape((params.numTestPoints, params.numTestPoints)))
                return c

    def plot_cov(self, ax):
        if self.pdCovMap is not None:
            x_test = self.X_test[:,0].reshape((params.numTestPoints,params.numTestPoints))
            y_test = self.X_test[:,1].reshape((params.numTestPoints,params.numTestPoints))

            # c1 = ax.contour(x_test, y_test, self.pdMap.reshape((params.numTestPoints,params.numTestPoints)), levels = [params.probabilityOfDetectionThreshold])
            c = ax.pcolormesh(x_test, y_test, self.pdCovMap.reshape((params.numTestPoints, params.numTestPoints)))
            return c
    def plot_best_measurement_map(self, ax):
        
        c = None
        if self.estimatedRadarParamsList is not None:
            X_test = np.array(self.X_test)

            
            best_measurement_location_map = self.create_best_measurement_location_map(self.estimatedRadarParamsList, self.estimatedRadarParamsCovList)

            c = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), best_measurement_location_map.reshape((params.numTestPoints,params.numTestPoints)))

        return c

    def ground_truth_probability_of_detection(self, X_test, trueRadarParametersList):
        probabilityOfNoDetection = np.ones(len(X_test))
        

        for i,position in enumerate(X_test):
            pd_list = []
            pd_cov_list = []
            for j, radar in enumerate(trueRadarParametersList):
                pdi = self.compute_probability_of_detection_at_xy(position, (radar.position[0], radar.position[1], params.radarOutputPower*params.radarTransmitGain))
                probabilityOfNoDetection[i] *= (1-pdi)
        
        
        return 1-probabilityOfNoDetection   
            

if __name__ == '__main__':
    Pdmap = ProbabilityOfDetectionMap(params.X_test)
    val, cov = Pdmap.compute_probability_of_detection_at_points(np.array([[500,0]]), np.array([0,0,params.radarOutputPower]), np.array([[10,0,0],[0,10,0],[0,0,1]]), )
    print(val)
    print(cov)
    # Pdmap.compute_probability_of_detection_at_points_multiple_radar(X_test, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params, multipleEmitterOnlineLocationAndPowerEstimator.estimated_emmiter_params_covariances)
    