import jax.numpy as jnp
import jax

#get ride of this after testing (jax defaults to 32 bits, numpy to 64 bits)
# jax.config.update("jax_enable_x64", True)
from functools import partial
from jax import jit

import numpy as np
from scipy.constants import k as boltzman
from time import time



from probabilityOfDetectionMap import ProbabilityOfDetectionMap
from main_helper import create_radar_list
import params

@jit
def signal_to_noise_ration(effectiveRadarPower, radarRecieverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
    #antennea gain not i decibels
    return (effectiveRadarPower*radarRecieverGain*wavelength**2*radarCrossSection*radarPulseWidth)/((4*np.pi)**3*distance**4*boltzman*radarSystemTemperature)

# def compute_probability_of_detection_at_xy(position, radarParams, radarRecieveGainPriorMean, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, radarSystemTemperaturePriorMean, radarProbabilityOfFalseAlarmPriorMean):
#     radarXY = radarParams[0:2]
#     distance = np.linalg.norm(radarXY-position)
#     snr = signal_to_noise_ration(radarParams[2], radarRecieveGainPriorMean, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, distance, radarSystemTemperaturePriorMean)
#     if snr<0:
#         print("STOP")
#     return np.array([probability_of_detection(radarProbabilityOfFalseAlarmPriorMean, snr)])

@jit
def compute_probability_of_detection_vectorized(position, radarParams, radarRecieveGainPriorMean, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, radarSystemTemperaturePriorMean, radarProbabilityOfFalseAlarmPriorMean):
    radarXY = radarParams[0:2]
    distance = jnp.linalg.norm(radarXY-position,axis=1)
    snr = signal_to_noise_ration(radarParams[2], radarRecieveGainPriorMean, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, distance, radarSystemTemperaturePriorMean)
    # if snr<0:
    #     print("STOP")
    return jnp.array([probability_of_detection(radarProbabilityOfFalseAlarmPriorMean, snr)])

@jit
def probability_of_detection(probabilityOfFalseAlarm, snr):
    return jnp.exp((jnp.log(probabilityOfFalseAlarm))/(snr + 1))

@partial(jit, static_argnums=(1,))
def ground_truth_probability_of_detection(X_test, trueRadarParametersList,radarOutputPower,radarTransmitGain, radarRecieveGainPriorMean, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, radarSystemTemperaturePriorMean, radarProbabilityOfFalseAlarmPriorMean):
    probabilityOfNoDetection = jnp.ones(len(X_test))
    

    pdi = jnp.ones(len(X_test))
    for j, radar in enumerate(trueRadarParametersList):
        pdi = compute_probability_of_detection_vectorized(X_test, jnp.array([radar.position[0], radar.position[1], radarOutputPower*radarTransmitGain]), radarRecieveGainPriorMean, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, radarSystemTemperaturePriorMean, radarProbabilityOfFalseAlarmPriorMean)
        probabilityOfNoDetection *= (1-pdi.squeeze())
    
    
    return 1-probabilityOfNoDetection


def compute_probability_of_detection_at_points_multiple_radar(self, X_test, estimatedRadarParamsList, estimatedRadarParamsCovList):
    pdCovMap = np.zeros(len(X_test))
    probabilityOfNoDetection = jnp.ones(len(X_test))
    
    
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



if __name__=="__main__":

    testDeterministicCase = False

    #test deterministic case
    if testDeterministicCase:
        radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
        # x_test = np.array([[100,100],[200,200],[150,150]],dtype=np.float32)
        x_test = np.random.normal(0,params.bounds[1],(100,2))

        pd = ground_truth_probability_of_detection(x_test, radarList, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)

        start = time()
        pd = ground_truth_probability_of_detection(x_test, radarList, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
        pd_time = time()-start
        print("time for pd computation", pd_time)
        
        start = time()
        pdmap = ProbabilityOfDetectionMap(x_test, radarList)
        original_pdmap_time = time()-start
        print("time for pdmap computation", original_pdmap_time)
        print("Speedup", original_pdmap_time/pd_time)
        print(pd == pdmap.ground_truth_probability_of_detection(x_test, radarList))
    else:
        paramNum = 837
        radarParams = np.load("saved_data/estimated_params/"+str(837) + ".npy")
        radarParamsCov = np.load("saved_data/estimated_params_cov/"+str(837) + ".npy")
        print(radarParams.shape)
        print(radarParamsCov[0])
        