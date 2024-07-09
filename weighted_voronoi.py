import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from time import time
from jax import jit

from main_helper import create_radar_list
import params
from probabilityOfDetectionJax import ground_truth_probability_of_detection,compute_probability_of_detection_at_points_multiple_radar, compute_probability_of_detection_vectorized, probability_of_detection_uncertainty_single_radar_at_xy
# from test import compute_probability_of_detection_at_points_multiple_radar
from weightedVoronoiPathIntialzation import compute_path_weighted_voronoi
import pickle

def get_weighted_distance(p,x,weight):
    return np.linalg.norm(p-x)/weight

def plot_weighted_distance(point,weight,bounds,ax):
    x_test = np.linspace(0,bounds[0],100)
    y_test = np.linspace(0,bounds[1],100)
    [X,Y] = np.meshgrid(x_test,y_test)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = get_weighted_distance(point,np.array([X[i,j],Y[i,j]]),weight)
    
    levels = np.linspace(0,2,6)
    c = ax.contour(X,Y,Z,levels=levels)
    ax.scatter(point[0],point[1],marker='*',color='r')
    plt.colorbar(c,ax=ax)

def plot_vornoi(p1,p2,w1,w2,ax):
    x_test = np.linspace(0,3,100)
    y_test = np.linspace(0,3,100)
    [X,Y] = np.meshgrid(x_test,y_test)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            d1 = get_weighted_distance(p1,np.array([X[i,j],Y[i,j]]),w1)
            d2 = get_weighted_distance(p2,np.array([X[i,j],Y[i,j]]),w2)
            if d1 < d2:
                Z[i,j] = 1
            else:
                Z[i,j] = 2
    
    ax.pcolormesh(X,Y,Z)
    ax.scatter(p1[0],p1[1],marker='*',color='r')
    ax.scatter(p2[0],p2[1],marker='*',color='r')    

def ground_truth_radar_voronoi(X_test,radarList,ax):
    pdList = []
    

    for radar in radarList:
        pdList.append(ground_truth_probability_of_detection(X_test, (radar,),params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean))
    
    Z = np.argmax(pdList,axis=0)

    
    ax.pcolormesh(X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints))
    for radar in radarList:
        ax.scatter(radar.position[0],radar.position[1],marker='*',color='r')



def ground_truth_radar_voronoi_weighted(X_test,radarList,ax):
    weights = np.sqrt(np.sqrt(np.array([radar.outputPower*radar.transmitGain*radar.recieveGain for radar in radarList])))
    # weights = weights/np.max(weights)

    distances = []
    for i,radar in enumerate(radarList):
        distances.append(np.linalg.norm(X_test - radar.position,axis=1)/weights[i])
    
    distances = np.array(distances)
    Z = np.argmin(distances,axis=0)
    
    ax.contour(X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints))
    for radar in radarList:
        ax.scatter(radar.position[0],radar.position[1],marker='*',color='r')

def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))

    r = np.array(r)
    r[x<0] = 2-r[x<0]
    # r = r.at[x<0].set(2-r.at[x<0])
    return r
    # if (x >= 0.):
    #     return r
    # else:
    #     return 2. - r


def normcdf(x, mu, sigma):
    t = x-mu
    y = 0.5*erfcc(-t/(sigma*np.sqrt(2.0)))
    y = np.array(y)
    y[y>1.0] = 1.0
    return y



def chance_constraint(self,threshold, thresholdConfidence, mean, var):
    # return (mean - rho) > (-erfinv(2*delta-1)*np.sqrt(2*var))
    # return (mean - rho) > (erfinv(-2*delta+1)*np.sqrt(2*var))
    return (NormalDist(mu=mean, sigma=np.sqrt(var)).cdf(threshold)) > thresholdConfidence

def safe_corridors_uncertain_radar(X_test,pdThreshold, likleyhoodThreshold, estimatedRadarParams, estimatedRadarParamsCov,radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
    
    pdMean,pdCov = compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)

    pdSigma = np.sqrt(pdCov)
    
    probabilityPdLessThanThreshold = normcdf(pdThreshold,pdMean,pdSigma)
    # return probabilityPdLessThanThreshold > likleyhoodThreshold
    return probabilityPdLessThanThreshold
    # return pdMean > pdThreshold
    # return pdCov

def probability_pd_less_than_threshold(X_test,pdThreshold, estimatedRadarParams, estimatedRadarParamsCov,radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
    
    pdMean,pdCov = compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)

    pdSigma = np.sqrt(pdCov)
    
    probabilityPdLessThanThreshold = normcdf(pdThreshold,pdMean,pdSigma)
    return probabilityPdLessThanThreshold

def probability_pd_less_than_threshold_single_radar(X_test,pdThreshold, estimatedRadarParam, estimatedRadarParamsCov,radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
    
    # pdMean,pdCov = compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)

    pdMean = compute_probability_of_detection_vectorized(X_test, estimatedRadarParam, radarRecieveGain,radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)
    pdCov = probability_of_detection_uncertainty_single_radar_at_xy(X_test, estimatedRadarParam,estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)

    pdSigma = np.sqrt(pdCov)
    
    # probabilityPdLessThanThreshold = normcdf(pdThreshold,pdMean,pdSigma)
    # out = np.divide((pdThreshold - pdMean),pdSigma)
    out = pdMean

    # distance = np.linalg.norm(estimatedRadarParam[0:2]-X_test,axis=1)

    # return pdMean
    # return probabilityPdLessThanThreshold
    return out 
    # return distance

def ground_truth_safe_corridors(X_test, pdThreshold, radarList, ax):
    pd = ground_truth_probability_of_detection(X_test, radarList, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
    safe_corridor = pd < pdThreshold
    ax.pcolormesh(X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    return safe_corridor

    

# def compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParamsList, estimatedRadarParamsCovList, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
  

def weighted_voronoi_uncertain_radar_grid_method(X_test,estimatedRadarParamsList,estiamtedRadarParamsCovList):
    prob_list = []
    
    for i,radar in enumerate(estimatedRadarParamsList):
        # prob_list.append(probability_pd_less_than_threshold(X_test,params.probabilityOfDetectionThreshold, radar,estiamtedRadarParamsCovList[i],params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance))
        print("i",i)
        prob_list.append(probability_pd_less_than_threshold_single_radar(X_test,params.probabilityOfDetectionThreshold, radar,estiamtedRadarParamsCovList[i],params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance))
    
    prob_list = np.array(prob_list)
    
    # Z = np.argmin(prob_list,axis=0)
    Z = np.argmax(prob_list,axis=0)
    return Z

    
    
    
    

if __name__ == '__main__':
    # points = np.array([[1,1],[2,2]])
    # weights = np.array([1,2])
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))

    # ground_truth_radar_voronoi(params.X_test,radarList,ax)

    # plt.show()

    
    ### safe corridors
    # for i in range(1,1358,1):
    #     paramNum = i
    #     radarParams = np.load("saved_data/seed_91212_good/estimated_params/"+str(paramNum) + ".npy")
    #     radarParamsCov = np.load("saved_data/seed_91212_good/estimated_params_cov/"+str(paramNum) + ".npy")


    #     startTime = time()
    #     if len(radarParams) > 0:
    #         safe_corridor = safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold, params.thresholdConfidence, radarParams, radarParamsCov,params.radarRecieveGainPriorMean,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)

    #         fig, ax = plt.subplots()
    #         ax.set_aspect('equal')
    #         c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    #         ax.set_title("Estimated Safe Corridor for params: " + str(paramNum))
    #         plt.colorbar(c,ax=ax)
    #         plt.savefig("saved_data/seed_91212_good/safe_corridor_vid/"+str(paramNum)+".png")
    #         plt.close()

    radarParams = np.load("saved_data/currentData/estimated_params/639.npy")
    radarParamsCov = np.load("saved_data/currentData/estimated_params_cov/639.npy")
    # radarParams = np.load("saved_data/currentData/estimated_params/168.npy")
    # radarParamsCov = np.load("saved_data/currentData/estimated_params_cov/168.npy")
    safe_corridor = safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold, params.thresholdConfidence, radarParams, radarParamsCov,params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    # ax.set_title("Estimated Safe Corridor for params: " + str(1000))
    # plt.colorbar(c,ax=ax)


    fig3, ax3 = plt.subplots()
    ax3.set_aspect('equal')
    ground_truth_safe_corridors(params.X_test, params.probabilityOfDetectionThreshold, radarList, ax3)

    
    fig2, ax2 = plt.subplots()
    ax2.set_aspect('equal')
    Z = weighted_voronoi_uncertain_radar_grid_method(params.X_test,radarParams,radarParamsCov)
    labels = np.arange(0,len(radarParams))
    levels = np.linspace(-1,len(radarParams)+1,len(radarParams)+3)
    ax2.contour(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints),levels=levels)
    ax2.scatter(radarParams[:,0],radarParams[:,1],marker='*',color='r')
    for lab in labels:
        ax2.text(radarParams[lab,0],radarParams[lab,1],str(lab))
    levels = [0,params.thresholdConfidence,1]
    # c = ax2.contourf(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints),levels=levels)
    c = ax2.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints)>.8)
    fig2.colorbar(c,ax=ax2)


    
    ax2.set_aspect('equal')
    ax2.set_xlim([0,params.bounds[0]])
    ax2.set_ylim([0,params.bounds[1]])
    radarlist = create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm)
    compute_path_weighted_voronoi(radarList, plot=True, ax=ax2)
    
    # plt.show(block=False)
    # input("Press Enter to close all plots")
    # plt.close('all')
    
    

    

    

