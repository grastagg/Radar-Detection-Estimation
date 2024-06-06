import jax.numpy as jnp
import jax
import jax.scipy as jsp
from jax import jit, vmap
from functools import partial
from scipy.constants import k as boltzman
import params
import numpy as np
from time import time

# Ensure JAX uses 64-bit floating point numbers
jax.config.update("jax_enable_x64", True)


@jit
def signal_to_noise_ratio(effectiveRadarPower, radarReceiverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
    return (effectiveRadarPower * radarReceiverGain * wavelength**2 * radarCrossSection * radarPulseWidth) / ((4 * jnp.pi)**3 * distance**4 * boltzman * radarSystemTemperature)

@jit
def probability_of_detection(probabilityOfFalseAlarm, snr):
    return jnp.exp((jnp.log(probabilityOfFalseAlarm)) / (snr + 1))

@jit
def compute_probability_of_detection_vectorized(position, radarParams, radarReceiveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    radarXY = radarParams[:2]
    distance = jnp.linalg.norm(radarXY - position, axis=1)
    snr = signal_to_noise_ratio(radarParams[2], radarReceiveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, distance, radarSystemTemperature)
    return probability_of_detection(radarProbabilityOfFalseAlarm, snr)

@partial(jit, static_argnums=(1,))
def ground_truth_probability_of_detection(X_test, trueRadarParametersList, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    trueRadarParametersList = jnp.array([[radar.position[0], radar.position[1],radar.outputPower*radar.transmitGain*radar.recieveGain] for radar in trueRadarParametersList])
    def compute_pdi(radar):
        return compute_probability_of_detection_vectorized(X_test, radar, 1, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)

    pdis = vmap(compute_pdi,in_axes=0,out_axes=0)(trueRadarParametersList)
    probabilityOfNoDetection = jnp.prod(1 - pdis, axis=0)
    return 1 - probabilityOfNoDetection

@jit
def pd_jacobian_emittor_params(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    x, y = position[:, 0], position[:, 1]
    x_em, y_em, ERP = estimatedRadarParams[:3]
    Gr, Pfa, rcs, tau_p, wavelength, T_s, k = radarRecieveGain, radarProbabilityOfFalseAlarm, agentRadarCrossSection, radarPulseWidth, radarWavelength, radarSystemTemperature, boltzman

    SNR = signal_to_noise_ratio(ERP, Gr, wavelength, rcs, tau_p, jnp.sqrt((x - x_em)**2 + (y - y_em)**2), T_s)
    exp_term = jnp.exp(jnp.log(Pfa) / (SNR + 1))

    d_pd_d_erp = -(Gr * jnp.log(Pfa) * rcs * tau_p * wavelength**2 * exp_term) / (64 * jnp.pi**3 * T_s * k * ((y - y_em)**2 + (x - x_em)**2)**2 * (SNR + 1)**2)
    d_pd_d_xem = -(ERP * Gr * jnp.log(Pfa) * rcs * tau_p * wavelength**2 * (x - x_em) * exp_term) / (16 * jnp.pi**3 * T_s * k * ((x - x_em)**2 + (y - y_em)**2)**3 * (SNR + 1)**2)
    d_pd_d_yem = -(ERP * Gr * jnp.log(Pfa) * rcs * tau_p * wavelength**2 * (y - y_em) * exp_term) / (16 * jnp.pi**3 * T_s * k * ((y - y_em)**2 + (x - x_em)**2)**3 * (SNR + 1)**2)
    
    return jnp.array([d_pd_d_xem, d_pd_d_yem, d_pd_d_erp])

@jit
def pd_jacobian_unknown_radar_parameters(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    x, y = position[:, 0], position[:, 1]
    x_em, y_em, ERP = estimatedRadarParams[:3]
    Gr, Pfa, rcs, tau_p, wavelength, T_s, k = radarRecieveGain, radarProbabilityOfFalseAlarm, agentRadarCrossSection, radarPulseWidth, radarWavelength, radarSystemTemperature, boltzman

    SNR = signal_to_noise_ratio(ERP, Gr, wavelength, rcs, tau_p, jnp.sqrt((x - x_em)**2 + (y - y_em)**2), T_s)
    exp_term = jnp.exp(jnp.log(Pfa) / (SNR + 1))
    log_pfa_term = jnp.log(Pfa) / (SNR + 1)**2

    d_pd_d_Gr = -exp_term * log_pfa_term * (ERP * wavelength**2 * rcs * tau_p) / ((4 * jnp.pi)**3 * (x - x_em)**4 * k * T_s)
    d_pd_d_wavelength = -exp_term * log_pfa_term * (Gr * ERP * 2 * wavelength * rcs * tau_p) / ((4 * jnp.pi)**3 * (x - x_em)**4 * k * T_s)
    d_pd_d_rcs = -exp_term * log_pfa_term * (Gr * ERP * wavelength**2 * tau_p) / ((4 * jnp.pi)**3 * (x - x_em)**4 * k * T_s)
    d_pd_d_tau_p = -exp_term * log_pfa_term * (Gr * ERP * wavelength**2 * rcs) / ((4 * jnp.pi)**3 * (x - x_em)**4 * k * T_s)
    d_pd_d_T_s = exp_term * log_pfa_term * (Gr * ERP * wavelength**2 * rcs * tau_p) / ((4 * jnp.pi)**3 * (x - x_em)**4 * k * T_s**2)
    d_pd_d_Pfa = exp_term * 1 / Pfa

    return jnp.array([d_pd_d_Gr, d_pd_d_wavelength, d_pd_d_rcs, d_pd_d_tau_p, d_pd_d_T_s, d_pd_d_Pfa])

@jit
def probability_of_detection_uncertainty_single_radar_at_xy(position, estimatedRadarParams, estimatedRadarParamsCov, radarRecieveGain, radarRecieveGainVar, radarWavelength, radarWavelengthVar, agentRadarCrossSection, radarPulseWidth, radarPulseWidthVar, radarSystemTemperature, radarSystemTemperatureVar, radarProbabilityOfFalseAlarm, radarProbabilityOfFalseAlarmVar):
    estimatedRadarParamsJacobian = pd_jacobian_emittor_params(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm).T
    unknownRadarParametersJacobian = pd_jacobian_unknown_radar_parameters(position, estimatedRadarParams, radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm).T
    
    radarParametersCovariance = jnp.diag(jnp.array([radarRecieveGainVar, radarWavelengthVar, 0, radarPulseWidthVar, radarSystemTemperatureVar, radarProbabilityOfFalseAlarmVar]))

    uncertainty = vmap(lambda i: estimatedRadarParamsJacobian[i] @ estimatedRadarParamsCov @ estimatedRadarParamsJacobian[i].T + unknownRadarParametersJacobian[i] @ radarParametersCovariance @ unknownRadarParametersJacobian[i].T)(jnp.arange(len(position)))
    
    return uncertainty

def compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParamsList, estimatedRadarParamsCovList, radarRecieveGain, radarRecieveGainVar, radarWavelength, radarWavelengthVar, agentRadarCrossSection, radarPulseWidth, radarPulseWidthVar, radarSystemTemperature, radarSystemTemperatureVar, radarProbabilityOfFalseAlarm, radarProbabilityOfFalseAlarmVar):
    def compute_pdi(estimatedRadarParams):
        return compute_probability_of_detection_vectorized(X_test, jnp.array([estimatedRadarParams[0], estimatedRadarParams[1], estimatedRadarParams[2]]), radarRecieveGain, radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)

    def compute_pdi_cov(estimatedRadarParams, estimatedRadarParamsCov):
        return probability_of_detection_uncertainty_single_radar_at_xy(X_test, estimatedRadarParams, estimatedRadarParamsCov, radarRecieveGain, radarRecieveGainVar, radarWavelength, radarWavelengthVar, agentRadarCrossSection, radarPulseWidth, radarPulseWidthVar, radarSystemTemperature, radarSystemTemperatureVar, radarProbabilityOfFalseAlarm, radarProbabilityOfFalseAlarmVar)

    pdis = vmap(compute_pdi)(estimatedRadarParamsList)
    pdi_covs = vmap(compute_pdi_cov)(estimatedRadarParamsList, estimatedRadarParamsCovList)

    probabilityOfNoDetection = jnp.prod(1 - pdis, axis=0)
    pd = 1 - probabilityOfNoDetection

    pdCov = jnp.zeros(len(X_test))
    for ind1 in range(len(pdis)):
        dpdt_dpdind1 = jnp.prod(1 - pdis.at[ind1].set(2), axis=0)
        pdCov += dpdt_dpdind1**2 * pdi_covs[ind1]

    return pd, pdCov
if __name__=="__main__":

    testDeterministicCase = False
    # radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
    # pd = ground_truth_probability_of_detection(np.array(params.X_test), radarList, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
    # plot_pd(np.array(params.X_test), pd)
    

    # # test deterministic case
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
        # x_test = np.array([[100,100],[200,200],[150,150]],dtype=np.float32)
        x_test = params.X_test
        # radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
        # pdmap = ProbabilityOfDetectionMap(x_test, radarList)
        paramNum = 837
        radarParams = np.load("saved_data/estimated_params/"+str(837) + ".npy")
        radarParamsCov = np.load("saved_data/estimated_params_cov/"+str(837) + ".npy")
        # pdAtXTestMean,pdAtXTestCov = pdmap.compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov)
        # print("pdAtXTestMean", pdAtXTestMean)
        # print("pdAtXTestCov", pdAtXTestCov)

        start = time()
        pdMeanJax, pdCovJax = compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov, params.radarRecieveGainPriorMean, params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidthPriorMean, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
        print("time to compile jax computation", time()-start)
        # print("pdMeanJax", pdMeanJax)
        # print("pdCovJax", pdCovJax)

        # start = time()
        # pdAtXTestMean,pdAtXTestCov = pdmap.compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov)
        # print("time for original pd computation", time()-start)

        start = time()
        pdMeanJax, pdCovJax = compute_probability_of_detection_at_points_multiple_radar(x_test, radarParams, radarParamsCov, params.radarRecieveGainPriorMean, params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidthPriorMean, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
        print("time for jax pd computation", time()-start)
        # print("pdMeanJax", pdMeanJax)
        # print("pdCovJax", pdCovJax)
        

        

        