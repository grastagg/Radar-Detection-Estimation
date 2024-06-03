import jax.numpy as jnp
import jax
import jax.scipy as jsp

#get ride of this after testing (jax defaults to 32 bits, numpy to 64 bits)
jax.config.update("jax_enable_x64", True)
from functools import partial
from jax import jit

import numpy as np
from scipy.constants import k as boltzman
from time import time



# from probabilityOfDetectionMap import ProbabilityOfDetectionMap
from main_helper import create_radar_list
import params

@jit
def signal_to_noise_ration(effectiveRadarPower, radarRecieverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
    #antennea gain not i decibels
    return (effectiveRadarPower*radarRecieverGain*wavelength**2*radarCrossSection*radarPulseWidth)/((4*jnp.pi)**3*distance**4*boltzman*radarSystemTemperature)

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
def ground_truth_probability_of_detection(X_test, trueRadarParametersList, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, radarSystemTemperaturePriorMean, radarProbabilityOfFalseAlarmPriorMean):
    probabilityOfNoDetection = jnp.ones(len(X_test))
    

    pdi = jnp.ones(len(X_test))
    for j, radar in enumerate(trueRadarParametersList):
        pdi = compute_probability_of_detection_vectorized(X_test, jnp.array([radar.position[0], radar.position[1], radar.outputPower*radar.transmitGain]), radar.recieveGain, radarWavelengthPriorMean, agentRadarCrossSection, radarPulseWidth, radarSystemTemperaturePriorMean, radarProbabilityOfFalseAlarmPriorMean)
        probabilityOfNoDetection *= (1-pdi.squeeze())
    
    
    return 1-probabilityOfNoDetection

@jit
def pd_jacobian_emittor_params(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    # x = position[0]
    if len(position.shape) == 1:
        x = position[0]
        y = position[1]
        
    else:
        x = position[:,0]
        y = position[:,1]
    x_em = estimatedRadarParams[0]
    y_em = estimatedRadarParams[1]
    ERP = estimatedRadarParams[2]
    Gr = radarRecieveGain
    Pfa = radarProbabilityOfFalseAlarm
    rcs = agentRadarCrossSection
    tau_p = radarPulseWidth
    wavelength = radarWavelength
    T_s = radarSystemTemperature
    k = boltzman
    d_pd_d_erp = -(Gr*jnp.log(Pfa)*rcs*tau_p*wavelength**2*jnp.exp(jnp.log(Pfa)/((Gr*rcs*tau_p*wavelength**2*ERP)/(64*jnp.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)))/(64*jnp.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2*((Gr*rcs*tau_p*wavelength**2*ERP)/(64*jnp.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2)
    d_pd_d_xem = -(ERP*Gr*jnp.log(Pfa)*rcs*tau_p*wavelength**2*(x-x_em)*jnp.exp(jnp.log(Pfa)/((ERP*Gr*rcs*tau_p*wavelength**2)/(64*jnp.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)))/(16*jnp.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*jnp.pi**3*T_s*k*((x-x_em)**2+(y-y_em)**2)**2)+1)**2*((x-x_em)**2+(y-y_em)**2)**3)
    d_pd_d_yem = -(ERP*Gr*jnp.log(Pfa)*rcs*tau_p*wavelength**2*(y-y_em)*jnp.exp(jnp.log(Pfa)/((ERP*Gr*rcs*tau_p*wavelength**2)/(64*jnp.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)))/(16*jnp.pi**3*T_s*k*((ERP*Gr*rcs*tau_p*wavelength**2)/(64*jnp.pi**3*T_s*k*((y-y_em)**2+(x-x_em)**2)**2)+1)**2*((y-y_em)**2+(x-x_em)**2)**3)
    return jnp.array([d_pd_d_xem,d_pd_d_yem,d_pd_d_erp])

@jit
def pd_jacobian_unkown_radar_parameters(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    x = position[:,0]
    y = position[:,1]
    x_em = estimatedRadarParams[0]
    y_em = estimatedRadarParams[1]
    ERP = estimatedRadarParams[2]
    Gr = radarRecieveGain
    Pfa = radarProbabilityOfFalseAlarm
    rcs = agentRadarCrossSection
    tau_p = radarPulseWidth
    wavelength = radarWavelength
    T_s = radarSystemTemperature
    k = boltzman
    SNR = signal_to_noise_ration(ERP, Gr, wavelength, rcs, tau_p, jnp.sqrt((x-x_em)**2+(y-y_em)**2),T_s)

    d_pd_d_Gr = -jnp.exp((jnp.log(Pfa)/(SNR+1))) * (jnp.log(Pfa))/(SNR+1)**2 * (ERP*wavelength**2*rcs*tau_p)/((4*jnp.pi)**3*jnp.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
    d_pd_d_wavelength = -jnp.exp((jnp.log(Pfa)/(SNR+1))) * (jnp.log(Pfa))/(SNR+1)**2 * (Gr*ERP*2*wavelength*rcs*tau_p)/((4*jnp.pi)**3*jnp.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
    d_pd_d_rcs = -jnp.exp((jnp.log(Pfa)/(SNR+1))) * (jnp.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*tau_p)/((4*jnp.pi)**3*jnp.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
    d_pd_d_tau_p = -jnp.exp((jnp.log(Pfa)/(SNR+1))) * (jnp.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*rcs)/((4*jnp.pi)**3*jnp.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s)
    d_pd_d_T_s = jnp.exp((jnp.log(Pfa)/(SNR+1))) * (jnp.log(Pfa))/(SNR+1)**2 * (Gr*ERP*wavelength**2*rcs*tau_p)/((4*jnp.pi)**3*jnp.sqrt((x-x_em)**2+(y-y_em)**2)**4*k*T_s**2)
    d_pd_d_Pfa = 1/(SNR+1) * jnp.exp(jnp.log(Pfa)/(SNR+1)) * 1/Pfa

    return jnp.array([d_pd_d_Gr, d_pd_d_wavelength, d_pd_d_rcs, d_pd_d_tau_p, d_pd_d_T_s, d_pd_d_Pfa])

@jit
def probability_of_detection_uncertainty_single_radar_at_xy(position, estimatedRadarParams, estimatedRadarParamsCov,radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
    estimatedRadarParamsJacobian = pd_jacobian_emittor_params(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm).T

    unkownRadarParametersJacobian = pd_jacobian_unkown_radar_parameters(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm).T
    radarParametersCovariance = jnp.diag(jnp.array([radarRecieveGainVar,radarWavelengthVar, 0, radarPulseWidthVar, radarSystemTemperatureVar, radarProbabilityOfFalseAlarmVar]))


    # estimatedRadarParamsCov = jsp.linalg.block_diag(*[estimatedRadarParamsCov for i in range(len(position))])
    # estimatedRadarParamsCov = j.linalg.block_diag(*[estimatedRadarParamsCov for i in range(len(position))])

    
    # return np.squeeze(estimatedRadarParamsJacobian @ estimatedRadarParamsCov@estimatedRadarParamsJacobian.T + radarParametersJacobian @ radarParametersCovariance @ radarParametersJacobian.T)
    return jnp.array([estimatedRadarParamsJacobian[i] @ estimatedRadarParamsCov@estimatedRadarParamsJacobian[i].T + unkownRadarParametersJacobian[i] @ radarParametersCovariance@unkownRadarParametersJacobian[i].T for i in range(len(position))])

# @jit
def compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParamsList, estimatedRadarParamsCovList, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
    probabilityOfNoDetection = jnp.ones(len(X_test))
    
    

    pd_list = []
    pd_cov_list = []
    for j, estimatedRadarParams in enumerate(estimatedRadarParamsList):
        pdi = compute_probability_of_detection_vectorized(X_test, jnp.array([estimatedRadarParams[0], estimatedRadarParams[1], (estimatedRadarParams[2])]), radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)
        pd_list.append(pdi)
        probabilityOfNoDetection *= (1-pdi.squeeze())
        pdCov = probability_of_detection_uncertainty_single_radar_at_xy(X_test, estimatedRadarParams, estimatedRadarParamsCovList[j], radarRecieveGain, radarRecieveGainVar, radarWavelength, radarWavelengthVar, agentRadarCrossSection, radarPulseWidth, radarPulseWidthVar, radarSystemTemperature, radarSystemTemperatureVar, radarProbabilityOfFalseAlarm, radarProbabilityOfFalseAlarmVar)
        pd_cov_list.append(pdCov)

            
    pdCov = np.zeros(len(X_test))
    for ind1 in range(len(pd_list)):
        dpdt_dpdind1 = 1
        for ind2 in range(len(pd_list)):
            if ind1 != ind2:
                dpdt_dpdind1 *= (1-pd_list[ind2])
        pdCov += dpdt_dpdind1**2*pd_cov_list[ind1]


    
    
    
    return 1-probabilityOfNoDetection, pdCov


def plot_pd(x_test, pd):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    c = ax.pcolormesh(x_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), x_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), pd.reshape((params.numTestPoints,params.numTestPoints)))
    fig.colorbar(c, ax=ax)
    plt.show()



if __name__=="__main__":

    testDeterministicCase = False
    radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
    pd = ground_truth_probability_of_detection(np.array(params.X_test), radarList, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
    plot_pd(np.array(params.X_test), pd)
    

    #test deterministic case
    # if testDeterministicCase:
    #     radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
    #     # x_test = np.array([[100,100],[200,200],[150,150]],dtype=np.float32)
    #     x_test = np.random.normal(0,params.bounds[1],(100,2))

    #     pd = ground_truth_probability_of_detection(x_test, radarList, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)

    #     start = time()
    #     pd = ground_truth_probability_of_detection(x_test, radarList, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
    #     pd_time = time()-start
    #     print("time for pd computation", pd_time)
        
    #     start = time()
    #     pdmap = ProbabilityOfDetectionMap(x_test, radarList)
    #     original_pdmap_time = time()-start
    #     print("time for pdmap computation", original_pdmap_time)
    #     print("Speedup", original_pdmap_time/pd_time)
    #     print(pd == pdmap.ground_truth_probability_of_detection(x_test, radarList))
    # else:
    #     # x_test = np.array([[100,100],[200,200],[150,150]],dtype=np.float32)
    #     x_test = np.random.normal(0,params.bounds[1],(100,2))
    #     radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
    #     pdmap = ProbabilityOfDetectionMap(x_test, radarList)
    #     paramNum = 837
    #     radarParams = np.load("saved_data/estimated_params/"+str(837) + ".npy")
    #     radarParamsCov = np.load("saved_data/estimated_params_cov/"+str(837) + ".npy")
    #     pdAtXTestMean,pdAtXTestCov = pdmap.compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov)
    #     # print("pdAtXTestMean", pdAtXTestMean)
    #     # print("pdAtXTestCov", pdAtXTestCov)

    #     start = time()
    #     pdMeanJax, pdCovJax = compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov, params.radarRecieveGainPriorMean, params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidthPriorMean, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
    #     print("time to compile jax computation", time()-start)
    #     # print("pdMeanJax", pdMeanJax)
    #     # print("pdCovJax", pdCovJax)

    #     start = time()
    #     pdAtXTestMean,pdAtXTestCov = pdmap.compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov)
    #     print("time for original pd computation", time()-start)

    #     start = time()
    #     pdMeanJax, pdCovJax = compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov, params.radarRecieveGainPriorMean, params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidthPriorMean, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
    #     print("time for jax pd computation", time()-start)
        

        

        