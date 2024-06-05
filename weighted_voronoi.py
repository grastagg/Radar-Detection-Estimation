import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from time import time
from jax import jit

from main_helper import create_radar_list
import params
from probabilityOfDetectionJax import ground_truth_probability_of_detection,compute_probability_of_detection_at_points_multiple_radar
# from test import compute_probability_of_detection_at_points_multiple_radar
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
    print(weights)

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
    return probabilityPdLessThanThreshold > likleyhoodThreshold
    # return pdMean > pdThreshold
    # return pdCov

    

# def compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParamsList, estimatedRadarParamsCovList, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
  
    
    
    
    

if __name__ == '__main__':
    # points = np.array([[1,1],[2,2]])
    # weights = np.array([1,2])
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))

    # ground_truth_radar_voronoi(params.X_test,radarList,ax)

    # fig2, ax2 = plt.subplots()
    # ax2.set_aspect('equal')
    # ground_truth_radar_voronoi_weighted(params.X_test,radarList,ax2)
    # plt.show()
    paramNum = 130
    radarParams = np.load("saved_data/estimated_params/"+str(paramNum) + ".npy")
    radarParamsCov = np.load("saved_data/estimated_params_cov/"+str(paramNum) + ".npy")


    startTime = time()
    safe_corridor = safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold, params.thresholdConfidence, radarParams, radarParamsCov,params.radarRecieveGainPriorMean,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)
    print("time to run",time()-startTime)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    plt.colorbar(c,ax=ax)


    paramNum = 830
    radarParams = np.load("saved_data/estimated_params/"+str(paramNum) + ".npy")
    radarParamsCov = np.load("saved_data/estimated_params_cov/"+str(paramNum) + ".npy")

    startTime = time()
    safe_corridor = safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold, params.thresholdConfidence, radarParams, radarParamsCov,params.radarRecieveGainPriorMean,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)
    print("time to run",time()-startTime)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    plt.colorbar(c,ax=ax)
    plt.show()
    
    

    

    

