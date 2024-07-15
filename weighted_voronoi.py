import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from time import time
from jax import jit
from scipy.constants import k as boltzman
from scipy.spatial import Voronoi, voronoi_plot_2d

from main_helper import create_radar_list
import params
from probabilityOfDetectionJax import ground_truth_probability_of_detection,compute_probability_of_detection_at_points_multiple_radar, compute_probability_of_detection_vectorized, probability_of_detection_uncertainty_single_radar_at_xy, probability_of_detection_uncertainty_single_radar_at_xy_bounds
# from test import compute_probability_of_detection_at_points_multiple_radar
from weightedVoronoiPathIntialzation import compute_path_weighted_voronoi
import pickle

import cv2

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

def probability_pd_less_than_threshold_single_radar(X_test,pdThreshold, estimatedRadarParam, estimatedRadarParamsCov,radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar,useUpperBound=False):
    
    # pdMean,pdCov = compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)

    pdMean = compute_probability_of_detection_vectorized(X_test, estimatedRadarParam, radarRecieveGain,radarWavelength, agentRadarCrossSection, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)
    if not useUpperBound:
        pdCov = probability_of_detection_uncertainty_single_radar_at_xy(X_test, estimatedRadarParam,estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)
        pdSigma = np.sqrt(pdCov)
    else:
        pdVarLower,pdVarUpper = probability_of_detection_uncertainty_single_radar_at_xy_bounds(X_test, estimatedRadarParam,estimatedRadarParamsCov, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar)
        pdSigma = np.sqrt(pdVarUpper)

    


    offset = 100000
    out = np.divide((pdThreshold - pdMean),pdSigma)

    

    #test code
    # ERP = estimatedRadarParam[2]
    # x_em =estimatedRadarParam[0]
    # y_em = estimatedRadarParam[1]
    # Gr, Pfa, rcs, tau_p, wavelength, T_s, k = radarRecieveGain, radarProbabilityOfFalseAlarm, agentRadarCrossSection, radarPulseWidth, radarWavelength, radarSystemTemperature, boltzman
    # x = X_test[:, 0]
    # y = X_test[:, 1]
    # c = (Gr * wavelength**2 * rcs * radarPulseWidth) / ((4.0 * np.pi)**3 * boltzman * radarSystemTemperature)

    # R = np.sqrt((x - x_em)**2 + (y - y_em)**2)

    # w,v = np.linalg.eig(estimatedRadarParamsCov)
    # max_eig = np.max(w)
    # min_eig = np.min(w)
    # # out_test = np.divide((pdThreshold - pdMean),(np.sqrt(max_eig)*np.abs(4*c*np.log(Pfa)*pdMean*ERP)/(R**5*(ERP*c/R**4+1)**2)))
    # # sigmaupper_test = (np.sqrt(max_eig)*np.abs(4*c*np.log(Pfa)*pdMean*ERP)/(R**5*(ERP*c/R**4+1)**2))
    # # print("sig test",np.allclose(pdSigma,sigmaupper_test))
    # # out_test = (pdThreshold/pdMean-1)*((ERP*c+R**4)**2)/(R**3*np.sqrt(max_eig)*np.abs(4*c*np.log(Pfa)*1*ERP))
    # # out_test = np.log(pdThreshold/pdMean-1)+2*np.log(ERP*c+R**4)-np.log(R**3*np.sqrt(max_eig)*np.abs(4*c*np.log(Pfa)*1*ERP)) + np.log(offset)
    # # out_test = np.log(pdThreshold/pdMean-1)+2*np.log(ERP*c+R**4)-np.log(R**3*np.sqrt(max_eig)*np.abs(4*c*np.log(Pfa)*1*ERP)) + np.log(offset)

    # # out = 


    
    # # out = np.log
    # # print("out",out[~np.isclose(out,out_test,atol=1e-5)])
    # # print("out test",out_test[~np.isclose(out,out_test,atol=1e-5)])
    # # print("out",out)
    # # print("out test",out)
    
    


    return out 

def ground_truth_safe_corridors(X_test, pdThreshold, radarList, ax):
    pd = ground_truth_probability_of_detection(X_test, radarList, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
    safe_corridor = pd < pdThreshold
    ax.pcolormesh(X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    return safe_corridor

    

# def compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParamsList, estimatedRadarParamsCovList, radarRecieveGain,radarRecieveGainVar, radarWavelength,radarWavelengthVar, agentRadarCrossSection, radarPulseWidth,radarPulseWidthVar, radarSystemTemperature,radarSystemTemperatureVar, radarProbabilityOfFalseAlarm,radarProbabilityOfFalseAlarmVar):
  

def weighted_voronoi_uncertain_radar_grid_method(X_test,prob_list,estimatedRadarParamsList,estiamtedRadarParamsCovList,useUpperBound=False,ax = None):
    Z = np.argmin(prob_list,axis=0).reshape(params.numTestPoints,params.numTestPoints)
    pixelDist = params.bounds[0]/params.numTestPoints
    # print("Z",Z.shape)

    contours = []
    for i in range(len(estimatedRadarParamsList)):
        Z_temp = np.where(Z==i,1,0)
        # np.pad(Z_temp,10,constant_values=0)
        contour,_ = cv2.findContours(Z_temp.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # fig, ax = plt.subplots()
        contours.append(contour)
        # for cont in contour:
        #     # print(cont)
        #     ax.plot(cont[:,0,0]*pixelDist,cont[:,0,1]*pixelDist)
        # ax.set_aspect('equal')
        # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z_temp)
        

    # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z)
    if ax is not None:
        for contour in contours:
            for cont in contour:
                # print(cont)
                ax.plot(cont[:,0,0]*pixelDist,cont[:,0,1]*pixelDist)

    
    # img = np.zeros((params.numTestPoints,params.numTestPoints,3),dtype=np.uint8)
    # cv2.drawContours(img, countour, -1, (0, 255, 0), 3)
    # cv2.imshow("contour",img)

    # Z = np.argmax(prob_list,axis=0)
    return Z

def get_closest_neighbor_triplets(points):
    # Compute the Voronoi diagram for the given points
    vor = Voronoi(points[:, 0:2])
    num_vertices = len(vor.vertices)
    vertex_to_regions = [[] for _ in range(num_vertices)]

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        for vertex_idx in region:
            if vertex_idx != -1:  # Skip the point at infinity
                vertex_to_regions[vertex_idx].append(point_idx)


    return np.array(vertex_to_regions)

def create_probability_of_detection_below_threshold_grid_list(X_test,estimatedRadarParamsList,estiamtedRadarParamsCovList):
    prob_list = []
    
    for i,radar in enumerate(estimatedRadarParamsList):
        # prob_list.append(probability_pd_less_than_threshold(X_test,params.probabilityOfDetectionThreshold, radar,estiamtedRadarParamsCovList[i],params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance))
        prob_list.append(probability_pd_less_than_threshold_single_radar(X_test,params.probabilityOfDetectionThreshold, radar,estiamtedRadarParamsCovList[i],params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance,useUpperBound=False))
    
    prob_list = np.array(prob_list)
    return prob_list

def indecies_to_xy(ind):
    indexToXYConversion = params.bounds[0]/(params.numTestPoints-1)
    indx,indy = ind//params.numTestPoints,ind%params.numTestPoints
    x = indx*indexToXYConversion
    y = indy*indexToXYConversion
    return x,y

def xy_to_index(x,y):
    indexToXYConversion = params.bounds[0]/(params.numTestPoints-1)
    indx = x//indexToXYConversion
    indy = y//indexToXYConversion
    return int(indx*params.numTestPoints + indy)

def find_generalized_voronoi_verticies(prob_list,closetNeighborTriples):

    indexToXYConversion = params.bounds[0]/(params.numTestPoints-1)
    

    verticies = {}
    tol = 1e-5
    for index,neighbors in enumerate(closetNeighborTriples):
        i,j,k = neighbors 
        i = int(i)
        j = int(j)
        k = int(k)
        diffij = np.abs(prob_list[i] - prob_list[j])
        diffik = np.abs(prob_list[i] - prob_list[k])
        diffjk = np.abs(prob_list[j] - prob_list[k])
        sumDiff = diffij + diffik + diffjk
        ind = np.argmin(sumDiff)
        # indx,indy = ind//params.numTestPoints,ind%params.numTestPoints
        # x = indx*indexToXYConversion
        # y = indy*indexToXYConversion
        x,y = indecies_to_xy(ind)
        verticies[index] = {"neighbors":neighbors,"point":(x,y)}
                
    return verticies

def find_ridge_line(prob_list, ridgeNeighborIndecies,vertex1,vertex2):

    i,j = ridgeNeighborIndecies
    celliprob = prob_list[i]
    celljprob = prob_list[j]
    cellAssignment = np.where(celliprob < celljprob,0,1).reshape(params.numTestPoints,params.numTestPoints)
    fig,ax2 = plt.subplots()
    # ax2.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),cellAssignment.reshape(params.numTestPoints,params.numTestPoints))
    ax2.scatter(vertex1[0],vertex1[1],marker='*',color='r')
    ax2.scatter(vertex2[0],vertex2[1],marker='*',color='r')

    contour,_ = cv2.findContours(cellAssignment.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pixelDist = params.bounds[0]/params.numTestPoints
    points = contour[1].squeeze()*pixelDist
    print(points.shape)
    ax2.plot(points[:,0],points[:,1])
    # ax2.plot(contour[:,0,0]*pixelDist,contour[:,0,1]*pixelDist)

    plt.show()
    print("contour",contour[0])


    
    # vertex1Index = xy_to_index(vertex1[0],vertex1[1])
    # vertex2Index = xy_to_index(vertex2[0],vertex2[1])
    # diffij = np.abs(prob_list[i] - prob_list[j])
    # print("diffij",diffij[vertex1Index])
    # print("diffij",diffij[vertex2Index])
    # maxDiff = np.min([diffij[vertex1Index],diffij[vertex2Index]])/10
    # ridgeIndecies = np.where(diffij < 2*maxDiff)[0]
    # ridgeIndeciesX,ridgeIndeciesY = indecies_to_xy(ridgeIndecies)
    # return ridgeIndeciesX,ridgeIndeciesY
    

def find_generalized_voronoi_ridges(prob_list,verticies,ax = None):

    ridges = {}
    for i in range(len(verticies)):
        for j in range(len(verticies)):
            if i != j:
                commonNeighbors = np.intersect1d(verticies[i]['neighbors'],verticies[j]['neighbors'])
                if len(commonNeighbors) == 2:
                    print("i,j",i,j)
                    print("verticies[i]",verticies[i]['neighbors'])
                    print("verticies[j]",verticies[j]['neighbors'])
                    ridgeLinePoints = find_ridge_line(prob_list,commonNeighbors,verticies[i]['point'],verticies[j]['point'])
                    if ax is not None:
                        ax.scatter(ridgeLinePoints[0],ridgeLinePoints[1])
                    






    
    
    
    
    
    

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
    # safe_corridor = safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold, params.thresholdConfidence, radarParams, radarParamsCov,params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    # ax.set_title("Estimated Safe Corridor for params: " + str(1000))
    # plt.colorbar(c,ax=ax)


    # fig3, ax3 = plt.subplots()
    # ax3.set_aspect('equal')
    # ground_truth_safe_corridors(params.X_test, params.probabilityOfDetectionThreshold, radarList, ax3)

    
    # fig2, ax2 = plt.subplots()
    # ax2.set_aspect('equal')
    # # Z_upper = weighted_voronoi_uncertain_radar_grid_method(params.X_test,radarParams,radarParamsCov,useUpperBound=True)
    # labels = np.arange(0,len(radarParams))
    # levels = np.linspace(-1,len(radarParams)+1,len(radarParams)+3)
    # ax2.contourf(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints),levels=levels)
    # # ax2.contour(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z_upper.reshape(params.numTestPoints,params.numTestPoints),levels=levels,cmap = 'tab20')
    # ax2.scatter(radarParams[:,0],radarParams[:,1],marker='*',color='r')
    # for lab in labels:
    #     ax2.text(radarParams[lab,0],radarParams[lab,1],str(lab))
    # levels = [0,params.thresholdConfidence,1]
    # c = ax2.contourf(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints),levels=levels)
    # c = ax2.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    # fig2.colorbar(c,ax=ax2)


    
    # ax2.set_aspect('equal')
    # ax2.set_xlim([0,params.bounds[0]])
    # ax2.set_ylim([0,params.bounds[1]])
    # radarlist = create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm)
    # compute_path_weighted_voronoi(radarList, plot=True, ax=ax2)
    
    
    
    # new generailzed voronoi
    fig, ax = plt.subplots()
    prob_list = create_probability_of_detection_below_threshold_grid_list(params.X_test,radarParams,radarParamsCov)
    # Z = weighted_voronoi_uncertain_radar_grid_method(params.X_test,prob_list,radarParams,radarParamsCov,useUpperBound=False,ax=ax)
    closestNeighborTriples = get_closest_neighbor_triplets(radarParams[:,0:2])

    verticies = find_generalized_voronoi_verticies(prob_list,closestNeighborTriples)
    ridges = find_generalized_voronoi_ridges(prob_list,verticies,ax=ax)
    ax.set_aspect('equal')
    for i in range(len(verticies)):
        vertex = verticies[i]
        point = vertex['point']
        ax.scatter(point[0],point[1],marker='*',color='g')
        ax.text(point[0],point[1],str(i))
    
    

    # vor = Voronoi(radarParams[:,0:2])
    # voronoi_plot_2d(vor,ax=ax,show_vertices=False)

    # for i,point in enumerate(vor.vertices):
    #     ax.text(point[0],point[1],str(i))

    # for i in range(len(vor.points)):
    #     point = vor.points[i]
    #     ax.text(point[0],point[1],str(i))
    #     ax.scatter(point[0],point[1])

    plt.show()
    
    

    

    

