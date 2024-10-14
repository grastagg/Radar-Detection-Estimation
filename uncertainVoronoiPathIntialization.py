import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from time import time
from jax import jit
from scipy.constants import k as boltzman
import scipy.integrate
import scipy.interpolate
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy
import importlib
import sklearn.cluster

from main_helper import create_radar_list
# import params
from probabilityOfDetectionJax import ground_truth_probability_of_detection,compute_probability_of_detection_at_points_multiple_radar, compute_probability_of_detection_vectorized, probability_of_detection_uncertainty_single_radar_at_xy, probability_of_detection_uncertainty_single_radar_at_xy_bounds
# from test import compute_probability_of_detection_at_points_multiple_radar
from weightedVoronoiPathIntialzation import compute_path_weighted_voronoi
import pickle

import cv2

from scipy import interpolate
from scipy.interpolate import splrep

import igraph
import highPriorityHelperFunctions
from utils import create_test_points

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
    
    
    # probabilityPdLessThanThreshold = normcdf(pdThreshold,pdMean,pdSigma)
    probabilityPdLessThanThreshold = scipy.stats.norm.cdf(pdThreshold,pdMean,pdSigma)


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

    


    out = np.divide((pdThreshold - pdMean),pdSigma)
    return out 

def ground_truth_safe_corridors(X_test, pdThreshold, radarList, ax):
    pd = ground_truth_probability_of_detection(X_test, radarList, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
    safe_corridor = pd < pdThreshold
    ax.pcolormesh(X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),safe_corridor.reshape(params.numTestPoints,params.numTestPoints))
    return safe_corridor

    

  

def weighted_voronoi_uncertain_radar_grid_method(X_test,prob_list,estimatedRadarParamsList,estiamtedRadarParamsCovList,useUpperBound=False,ax = None):
    Z = np.argmin(prob_list,axis=0).reshape(params.numTestPoints,params.numTestPoints)
    pixelDist = params.bounds[0]/params.numTestPoints

    contours = []
    for i in range(len(estimatedRadarParamsList)):
        Z_temp = np.where(Z==i,1,0)
        contour,_ = cv2.findContours(Z_temp.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(contour)
    if ax is not None:
        for contour in contours:
            for cont in contour:
                ax.plot(cont[:,0,0]*pixelDist,cont[:,0,1]*pixelDist)

    
    return Z

def get_closest_neighbor_triplets(vor):
    # Compute the Voronoi diagram for the given points
    num_vertices = len(vor.vertices)
    vertex_to_regions = [[] for _ in range(num_vertices)]

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        for vertex_idx in region:
            if vertex_idx != -1:  # Skip the point at infinity
                vertex_to_regions[vertex_idx].append(point_idx)
            
    return np.array(vertex_to_regions)

def create_probability_of_detection_below_threshold_grid_list(X_test,estimatedRadarParamsList,estiamtedRadarParamsCovList,params):
    prob_list = []
    
    for i,radar in enumerate(estimatedRadarParamsList):
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

def find_contour_points(cellAssignment,index,bounds,numTestPoints):
    cellI = np.where(cellAssignment == index,1,0)

    contours,_ = cv2.findContours(cellI.reshape(numTestPoints,numTestPoints).astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    pixelDist = bounds[0]/numTestPoints
    # print(contours)
    # fig,ax = plt.subplots()
    # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),cellI.reshape(params.numTestPoints,params.numTestPoints))
    # # ax.plot(contours.squeeze()[:,0]*pixelDist,contours.squeeze()[:,1]*pixelDist)
    # plt.show()
    # points = contours[0].squeeze()*pixelDist
    points = contours[0].reshape((-1,2))*pixelDist
    for i in range(1,len(contours)):
        # points = np.append(points,contours[i].squeeze()*pixelDist,axis=0)
        points = np.append(points,contours[i].reshape((-1,2))*pixelDist,axis=0)

        
    if len(points) == 1:
        print("ERROR")
    
    return points


def cluster_points_and_find_centroids(points):
    # Perform clustering to find the centroids
    meanShift = sklearn.cluster.MeanShift(bandwidth=200).fit(points)
    centroids = meanShift.cluster_centers_
    return centroids

def find_closest_common_point_within_threshold(array1, array2, array3, threshold,indexI,indexJ,indexK):
    closest_points = ()

    # Calculate distances between points in array1 and array2
    dist12 = np.linalg.norm(array1[:, np.newaxis] - array2, axis=2)
    
    # Find pairs within the threshold distance
    close_pairs_12 = np.argwhere(dist12 < threshold)

        
    
    # Check if these pairs have a common point with array3
    # if indexI == 2 and indexJ == 3 and indexK == 4:
    #     fig,ax = plt.subplots()
    #     points1 = array1[close_pairs_12[:,0]]
    #     points2 = array2[close_pairs_12[:,1]]
    #     # ax.scatter(points1[:,0],points1[:,1])
    #     # ax.scatter(points2[:,0],points2[:,1])
    #     ax.plot(array1[:,0],array1[:,1])
    #     ax.plot(array2[:,0],array2[:,1])

    points3all = []
    for i, j in close_pairs_12:
        point1 = array1[i]
        point2 = array2[j]
        dist13 = np.linalg.norm(array3 - point1, axis=1)
        dist23 = np.linalg.norm(array3 - point2, axis=1)
        
        # Find common points in array3 within the threshold
        close_points_3 = np.argwhere((dist13 < threshold) & (dist23 < threshold))

        #####to remove
        #####
        
        for k in close_points_3:
            k = k[0]
            point3 = array3[k]
            points3all.append(point3)
            # # Calculate the total distance
            # total_distance = dist12[i, j] + dist13[k] + dist23[k]
            
            # if total_distance < min_distance:
            #     min_distance = total_distance
            #     closest_points = (point1, point2, point3)
    
    points3all = np.array(points3all)
    if len(points3all) > 0:
        closest_points = cluster_points_and_find_centroids(points3all)
    
    # if indexI == 2 and indexJ == 3 and indexK == 4:
    #     # ax.scatter(points3all[:,0],points3all[:,1])
    #     ax.scatter(closest_points[:,0],closest_points[:,1],marker='*',color='r')
    
    # if closest_points:
    if len(closest_points) > 0:
        # mean_point = np.mean(closest_points, axis=0)
        return True, closest_points
    else:
        return False, ()

def find_generalized_voronoi_verticies(prob_list,bounds,numTestPoints):

    verticies = {}

    contourPoints = []
    cellAssignment = np.argmin(prob_list,axis=0)
    
    radarIndeciesToIgnore = []
    for i in range(len(prob_list)):
        contourP = find_contour_points(cellAssignment,i,bounds,numTestPoints)
        if contourP is not None:
            contourPoints.append(contourP)
        else:
            radarIndeciesToIgnore.append(i)
    prob_list = np.delete(prob_list,radarIndeciesToIgnore,axis=0)
    # fig,ax = plt.subplots()
    # ax.set_xlim(0,bounds[0])
    # ax.set_ylim(0,bounds[1])
    # for contour in contourPoints:
    #     ax.plot(contour[:,0],contour[:,1])
    # ax.pcolor(prob_list)
    # plt.show()
    index = 0
    for i in range(len(prob_list)):
        for j in range(i+1,len(prob_list)):
            for k in range(j+1,len(prob_list)):
                pointsI = contourPoints[i]
                pointsJ = contourPoints[j]
                pointsK = contourPoints[k]
                

                commonPointExists,commonPoint = find_closest_common_point_within_threshold(pointsI,pointsJ,pointsK,2*bounds[0]/numTestPoints+100,i,j,k)
                
                if commonPointExists:
                    if len(commonPoint) == 1:
                        verticies[index] = {"neighbors":(i,j,k),"point":commonPoint.squeeze(),"edge":False}
                        index += 1
                    elif len(commonPoint) == 2:
                        verticies[index] = {"neighbors":(i,j,k),"point":commonPoint[0],"edge":False}
                        index += 1
                        verticies[index] = {"neighbors":(i,j,k),"point":commonPoint[1],"edge":False}
                        index += 1
                
    return verticies,contourPoints,cellAssignment,radarIndeciesToIgnore


def new_filter_points_by_vertices(contour, vertex1, vertex2, reference_point,secondReferencePoint,allVertices,vertex1Index,vertex2Index):
    
    distanceToVertex1 = np.linalg.norm(contour - vertex1,axis=1)
    distanceToVertex2 = np.linalg.norm(contour - vertex2,axis=1)
    
    closestPointToVertex1Index = np.argmin(distanceToVertex1)
    closestPointToVertex2Index = np.argmin(distanceToVertex2)

    mask = np.zeros(len(contour),dtype=bool)

    if closestPointToVertex1Index < closestPointToVertex2Index:
        # points_in_range = contour[closestPointToVertex1Index:closestPointToVertex2Index]
        mask[closestPointToVertex1Index:closestPointToVertex2Index] = True
    else:
        # points_in_range = contour[closestPointToVertex2Index:closestPointToVertex1Index]
        mask[closestPointToVertex2Index:closestPointToVertex1Index] = True
    
    allOtherVertices = []
    for vertexIndex in allVertices.keys():
        if vertexIndex == vertex1Index or vertexIndex == vertex2Index:
            continue
        allOtherVertices.append(allVertices[vertexIndex]["point"])
    
    allOtherVertices = np.array(allOtherVertices)

    points_in_range = contour[mask]
    if len(points_in_range) == 0:
        return np.array([vertex1,vertex2])
    
    distanceMatrix = np.linalg.norm(allOtherVertices - points_in_range[:,np.newaxis],axis=2)
    otherDistanceMatrix = np.linalg.norm(allOtherVertices - contour[~mask][:,np.newaxis],axis=2)
    # print()
    # print("min distance",np.min(distanceMatrix))
    if np.min(distanceMatrix) < np.min(otherDistanceMatrix):
        # print("TEST")
        mask = ~mask
        points_in_range = contour[mask]
    # print(mask)
    # print("min distance",np.min(distanceMatrix))
    # print("min other distance",np.min(otherDistanceMatrix))

    #sort points
    if mask[0] == True and mask[-1] == True:
        rollNumber = 0
        for i in range(len(mask)-1,0,-1):
            if mask[i] == False:
                rollNumber = len(mask)-i-1
                break
        points_in_range = np.roll(points_in_range,rollNumber,axis=0)
        # print(mask)

    
    
    # fig,ax = plt.subplots()
    # ax.set_xlim(0,params.bounds[0])
    # ax.set_ylim(0,params.bounds[1])
    # ax.scatter(contour[:,0],contour[:,1])
    # ax.plot(points_in_range[:,0],points_in_range[:,1],c = 'r')
    # ax.scatter(vertex1[0],vertex1[1],marker='*',color='g')
    # ax.scatter(vertex2[0],vertex2[1],marker='*',color='r')
    # ax.scatter(allOtherVertices[:,0],allOtherVertices[:,1])
    # plt.show()
    
    
    return points_in_range
        
    

def filter_points_by_vertices(contour, vertex1, vertex2, reference_point,secondReferencePoint,allVertices,vertex1Index,vertex2Index):
    """
    Filters points in the contour that lie between the two vertices.

    Parameters:
    contour (np.ndarray): Array of points representing the contour (shape: [n_points, 2]).
    vertex1 (tuple): Coordinates of the first vertex (x, y).
    vertex2 (tuple): Coordinates of the second vertex (x, y).
    reference_point (tuple): Coordinates of the reference point (x, y) to measure angles from.

    Returns:
    np.ndarray: Array of points that lie between the two vertices.
    """

    midpoint = (vertex1 + vertex2) / 2
    # Calculate the reference point
    centroid = reference_point.copy()
    normal = np.array([reference_point[0] - midpoint[0], reference_point[1] - midpoint[1]])
    normal = normal / np.linalg.norm(normal)
    reference_point = midpoint + 2500 * normal

    # reference_point = np.mean(contour,axis=0)
    
    # Calculate angles for each point in the contour relative to the reference point
    angles = np.arctan2(contour[:, 1] - reference_point[1], contour[:, 0] - reference_point[0])
    
    # Calculate angles for the vertices relative to the reference point
    angle_vertex1 = np.arctan2(vertex1[1] - reference_point[1], vertex1[0] - reference_point[0])
    angle_vertex2 = np.arctan2(vertex2[1] - reference_point[1], vertex2[0] - reference_point[0])

    # Normalize angles to the range [0, 2*pi]
    # angles = np.mod(angles + 2 * np.pi, 2 * np.pi)
    # angle_vertex1 = np.mod(angle_vertex1 + 2 * np.pi, 2 * np.pi)
    # angle_vertex2 = np.mod(angle_vertex2 + 2 * np.pi, 2 * np.pi)
    
    diff = np.abs(angle_vertex1 - angle_vertex2)
    singularity = False
    if diff > np.pi:
        singularity =True 
    
    
        

    if singularity:
        if angle_vertex1 > angle_vertex2:
            mask = (angles >= angle_vertex1) | (angles <= angle_vertex2)
        else:
            mask = (angles >= angle_vertex2) | (angles <= angle_vertex1)

    else:
        if angle_vertex1 > angle_vertex2:
            mask = (angles >= angle_vertex2) & (angles <= angle_vertex1)
        else:
            mask = (angles >= angle_vertex1) & (angles <= angle_vertex2)

    # v1 = vertex2-vertex1
    # v2 = vertex2 - reference_point
    # v3 = vertex2 - secondReferencePoint
    # cross1 = np.cross(v1,v2)
    # cross2 = np.cross(v1,v3)
    # if cross1*cross2 > 0:
    #     mask = ~mask
    
    # straightLine = np.linspace(contour[mask][0],contour[mask][-1],len(contour[mask]))
    # straightLine = np.linspace(vertex1,vertex2,len(contour[mask]))
    allOtherVertices = []
    for vertexIndex in allVertices.keys():
        if vertexIndex == vertex1Index or vertexIndex == vertex2Index:
            continue
        allOtherVertices.append(allVertices[vertexIndex]["point"])
    allOtherVertices = np.array(allOtherVertices)
        
    
    
    points_in_range = contour[mask]
    # points_in_range,_ = sort_points(points_in_range,reference_point)

    ###########################
    # fig,ax = plt.subplots()
    # ax.set_xlim(0,params.bounds[0])
    # ax.set_ylim(0,params.bounds[1])
    # ax.scatter(contour[:,0],contour[:,1])
    # ax.scatter(points_in_range[:,0],points_in_range[:,1])
    # ax.scatter(reference_point[0],reference_point[1],color='r')
    # ax.scatter(centroid[0],centroid[1],color='r')
    # # ax.scatter(secondReferencePoint[0],secondReferencePoint[1],marker='*',color='g')
    # # ax.plot([reference_point[0],secondReferencePoint[0]],[reference_point[1],secondReferencePoint[1]],c='g')
    # # ax.plot([vertex1[0],vertex2[0]],[vertex1[1],vertex2[1]],c='r')
    # ax.plot([reference_point[0],reference_point[0]+1000000*np.cos(angle_vertex1)],[reference_point[1],reference_point[1]+1000000*np.sin(angle_vertex1)],c='r')
    # ax.plot([reference_point[0],reference_point[0]+1000000*np.cos(angle_vertex2)],[reference_point[1],reference_point[1]+1000000*np.sin(angle_vertex2)],c='r')
    # # # # ax.scatter(secondReferencePoint[0],secondReferencePoint[1],marker='*',color='r')
    # ax.scatter(vertex1[0],vertex1[1],marker='*',color='g')
    # ax.scatter(vertex2[0],vertex2[1],marker='*',color='r')
    # # # ax.plot(straightLine[:,0],straightLine[:,1],c='g')
    # # # # # ax.plot(straightLine[:,0],straightLine[:,1],c='g')
    # ax.scatter(allOtherVertices[:,0],allOtherVertices[:,1])
    # plt.show()
    ###########################

    distanceMatrix = np.linalg.norm(allOtherVertices - points_in_range[:,np.newaxis],axis=2)
    if np.min(distanceMatrix) < 100:
        mask = ~mask
        points_in_range = contour[mask]
        # points_in_range,_ = sort_points(points_in_range,reference_point)

    if np.linalg.norm(vertex1 - points_in_range[0]) > np.linalg.norm(vertex1 - points_in_range[-1]):
        # points_in_range = np.append(vertex2, points_in_range,axis=0)
        points_in_range = np.insert(points_in_range, 0,vertex2.reshape((1,2)),axis=0)
        points_in_range = np.append(points_in_range, vertex1.reshape((1,2)),axis=0)
        # points_in_range = np.append(points_in_range,vertex1,axis=0)
        straightLine = np.linspace(vertex2,vertex1,len(points_in_range))
    else:
        points_in_range = np.insert(points_in_range, 0,vertex1.reshape((1,2)),axis=0)
        points_in_range = np.append(points_in_range, vertex2.reshape((1,2)),axis=0)
        # points_in_range = np.append(vertex1, points_in_range,axis=0)
        # points_in_range = np.append(points_in_range,vertex2,axis=0)
        straightLine = np.linspace(vertex1,vertex2,len(points_in_range))
    
    # distanceDiff = np.linalg.norm(straightLine - contour[mask],axis=1)
    # distanceDiff = np.linalg.norm(straightLine - points_in_range,axis=1)
    # diffMean = np.mean(distanceDiff)
    # if diffMean > 1600:
    #     mask = ~mask
    # points_in_range = contour[mask]
    # points_in_range,_ = sort_points(points_in_range,reference_point)
    
    # Filter points based on the mask
    # points_in_range = contour[mask]


    # fig,ax = plt.subplots()
    # ax.set_xlim(0,params.bounds[0])
    # ax.set_ylim(0,params.bounds[1])
    # ax.scatter(contour[:,0],contour[:,1])
    # ax.scatter(points_in_range[:,0],points_in_range[:,1])
    # ax.scatter(reference_point[0],reference_point[1],color='r')
    # ax.scatter(secondReferencePoint[0],secondReferencePoint[1],marker='*',color='g')
    # ax.plot([reference_point[0],secondReferencePoint[0]],[reference_point[1],secondReferencePoint[1]],c='g')
    # ax.plot([vertex1[0],vertex2[0]],[vertex1[1],vertex2[1]],c='r')
    # ax.plot([reference_point[0],reference_point[0]+1000000*np.cos(angle_vertex1)],[reference_point[1],reference_point[1]+1000000*np.sin(angle_vertex1)],c='r')
    # ax.plot([reference_point[0],reference_point[0]+1000000*np.cos(angle_vertex2)],[reference_point[1],reference_point[1]+1000000*np.sin(angle_vertex2)],c='r')
    # # # ax.scatter(secondReferencePoint[0],secondReferencePoint[1],marker='*',color='r')
    # ax.scatter(vertex1[0],vertex1[1],marker='*',color='g')
    # ax.scatter(vertex2[0],vertex2[1],marker='*',color='r')
    # # ax.plot(straightLine[:,0],straightLine[:,1],c='g')
    # # # # ax.plot(straightLine[:,0],straightLine[:,1],c='g')
    # ax.scatter(allOtherVertices[:,0],allOtherVertices[:,1])
    # plt.show()

    return points_in_range

def sort_points(points, reference_point):
    """
    Sorts points in the contour based on their angles relative to the reference point.

    Parameters:
    points (np.ndarray): Array of points representing the contour (shape: [n_points, 2]).
    reference_point (tuple): Coordinates of the reference point (x, y) to measure angles from.

    Returns:
    np.ndarray: Array of points sorted based on their angles.
    """
    # Calculate angles for each point in the contour relative to the reference point
    # angles = np.unwrap(np.arctan2(points[:, 1] - reference_point[1], points[:, 0] - reference_point[0]))
    angles = np.arctan2(points[:, 1] - reference_point[1], points[:, 0] - reference_point[0])
    # angles = np.mod(angles + 2 * np.pi, 2 * np.pi)

    indexes = np.argsort(angles)
    differenceAngles = np.empty_like(angles)
    differenceAngles[:-1] = np.abs(angles[indexes[1:]] - angles[indexes[:-1]])
    differenceAngles[-1] = np.abs(angles[indexes[0]]+2*np.pi - angles[indexes[-1]])
    maxDiffIndex = np.argmax(differenceAngles)
    
    indexes = np.roll(indexes,-maxDiffIndex-1)










    # sorted_points = points[np.argsort(angles)]
    sorted_points = points[indexes]

    # fig,ax = plt.subplots()
    # c = ax.scatter(points[:,0],points[:,1],c = angles)
    # fig.colorbar(c,ax=ax)
    # ax.scatter(reference_point[0],reference_point[1],marker='*',color='r')

    return sorted_points,indexes

def move_control_points_so_spline_passes_through_start_and_end(controlPoints, start, end):
    A = np.array([[1.0/6.0,0],[0,1.0/6.0]])
    c2x = controlPoints[1,0]
    c2y = controlPoints[1,1]
    c3x = controlPoints[2,0]
    c3y = controlPoints[2,1]
    b = np.array([[start[0] - (2.0/3.0)*c2x-(1.0/6.0)*c3x],[start[1]- (2.0/3.0)*c2y-(1.0/6.0)*c3y]])


    x = np.linalg.solve(A,b)
    controlPoints[0:1,0:2] = x.reshape((1,2))


    cn_minus_2_x = controlPoints[-3,0]
    cn_minus_2_y = controlPoints[-3,1]
    cn_minus_1_x = controlPoints[-2,0]
    cn_minus_1_y = controlPoints[-2,1]

    b = np.array([[end[0] - (2.0/3.0)*cn_minus_1_x-(1.0/6.0)*cn_minus_2_x],[end[1]- (2.0/3.0)*cn_minus_1_y-(1.0/6.0)*cn_minus_2_y]])

    controlPoints[-1:,0:2] = np.linalg.solve(A,b).reshape((1,2))


    return controlPoints

def create_unclamped_knot_points(t0, tf, numControlPoints,splineOrder):
    internalKnots = np.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = np.concatenate((np.linspace(t0-splineOrder*h,t0-h,splineOrder), internalKnots, np.linspace(tf+h,tf+splineOrder*h,splineOrder)))
    
    return knots

def fit_spline_to_path(path, num_control_points, spline_order,vertex1,vertex2):
    # if len(path) <= spline_order:
    #     path = np.linspace(path[0],path[-1],spline_order+3)
    if len(path) <= spline_order:
        path = np.linspace(path[0],path[-1],spline_order+1)
    if len(path) < num_control_points:
        num_control_points = len(path)
    tf = 1
    t = np.linspace(0,tf, len(path))
    
    # num_control_points = params.numControlPoints
    n_interior_knots = num_control_points - spline_order - 1
    qs = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
    knots = np.quantile(t, qs)

    tck_x = splrep(t,path[:,0],k=spline_order,t=knots,s=0)
    control_points_x = tck_x[1]
    mask = control_points_x != 0
    control_points_x = control_points_x[mask]
    

    tck_y = splrep(t,path[:,1],k=spline_order,t=knots,s=0)
    control_points_y = tck_y[1]
    control_points_y = control_points_y[mask]
    combined_control_points = np.hstack((control_points_x.reshape((len(control_points_x),1)), control_points_y.reshape((len(control_points_y),1))))
    num_control_points = len(combined_control_points)

    
    
    combined_control_points = move_control_points_so_spline_passes_through_start_and_end(combined_control_points, path[0],path[-1])

    combined_knot_points = tck_x[0]
    combined_knot_points = create_unclamped_knot_points(0,tf,num_control_points,spline_order)
    return combined_control_points,combined_knot_points

def get_cumulative_distances(points):
    # Calculate the distance between each consecutive point
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    # Calculate the cumulative distance
    cumulative_distances = np.cumsum(distances)
    # Add a zero at the beginning to represent the start point
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    return cumulative_distances

def resample_points(points, spacing):
    # Get the cumulative distances
    cumulative_distances = get_cumulative_distances(points)

    num_points = int(np.ceil(cumulative_distances[-1] / spacing))
    if num_points < 3:
        return points
    
    # Create an interpolation function for each dimension
    interp_func_x = scipy.interpolate.interp1d(cumulative_distances, points[:, 0], kind='linear')
    interp_func_y = scipy.interpolate.interp1d(cumulative_distances, points[:, 1], kind='linear')
    
    # Generate evenly spaced cumulative distances
    even_cumulative_distances = np.linspace(0, cumulative_distances[-1], num_points)
    
    # Interpolate to get the new points
    resampled_points_x = interp_func_x(even_cumulative_distances)
    resampled_points_y = interp_func_y(even_cumulative_distances)
    
    # Combine the x and y coordinates
    resampled_points = np.vstack((resampled_points_x, resampled_points_y)).T
    
    return resampled_points

def contours_to_points(contours,pixelDist):
    points = contours[0].reshape((-1,2))*pixelDist
    for i in range(1,len(contours)):
        points = np.append(points,contours[i].reshape((-1,2))*pixelDist,axis=0)
    return points

def find_ridge_line(cellAssign, ridgeNeighborIndecies,vertex1,vertex2,radarParams,radarParamsCovDeterminants,allVertecies,vertex1Index,vertex2Index,bounds,params=None):

    i,j = ridgeNeighborIndecies
    # celliprob = prob_list[i]
    # celljprob = prob_list[j]

    # if radarParamsCovDeterminants[i] > radarParamsCovDeterminants[j]:
    otherPoint = [0,0]
    # if radarParams[j][2] > radarParams[i][2]:
    #     referencePoint = radarParams[j,0:2]
    #     otherePoint = radarParams[i,0:2]
    #     cellAssignment = np.where(celliprob < celljprob,0,1).reshape(params.numTestPoints,params.numTestPoints)
    # else:
    #     referencePoint = radarParams[i,0:2]
    #     otherePoint = radarParams[j,0:2]
    #     cellAssignment = np.where(celliprob < celljprob,1,0).reshape(params.numTestPoints,params.numTestPoints)

    # cellAssignment = np.where(celliprob < celljprob,0,1).reshape(params.numTestPoints,params.numTestPoints)
    numTestPoints = int(np.sqrt(cellAssign.shape[0]))
    cellAssignment = np.where(cellAssign == i ,1,0).reshape(numTestPoints,numTestPoints)
    # nonZeroCount = np.count_nonzero(cellAssignment)
    # print("nonZeroCount",nonZeroCount)
    # zerosCount = params.numTestPoints**2 - nonZeroCount
    # print("zerosCount",zerosCount)
    # if nonZeroCount > zerosCount:
    #     cellAssignment = np.logical_not(cellAssignment).astype(int)
    
    pixelDist = bounds[0]/numTestPoints

    contour,_ = cv2.findContours(cellAssignment.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centroid = None
    for c in contour:
        if len(c) < 3:
            continue
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])*pixelDist
        cY = int(M["m01"] / M["m00"])*pixelDist
        centroid = np.array([cX,cY])

    points = contours_to_points(contour,pixelDist)
    
    # referencePoint = np.mean(points,axis=0)


    points = points[points[:,0]!=0]
    points = points[points[:,1]!=0]
    points = points[~np.isclose(points[:,0],bounds[0],atol=50)]
    points = points[~np.isclose(points[:,1],bounds[1],atol=50)]

    pointsInRange = new_filter_points_by_vertices(points, vertex1, vertex2, centroid,otherPoint,allVertecies,vertex1Index,vertex2Index)
    distVertex1ToFirstPoint = np.linalg.norm(pointsInRange[0] - vertex1)
    distVertex1ToLastPoint = np.linalg.norm(pointsInRange[-1] - vertex1)
    if distVertex1ToFirstPoint < distVertex1ToLastPoint:
        pointsInRange = np.append(vertex1,pointsInRange).reshape(-1,2)
        pointsInRange = np.append(pointsInRange, vertex2).reshape(-1,2)
    else:
        pointsInRange = np.append(vertex2,pointsInRange).reshape(-1,2)
        pointsInRange = np.append(pointsInRange, vertex1).reshape(-1,2)
    # pointsInRange = np.append(pointsInRange, vertex1).reshape(-1,2)
    # pointsInRange = np.append(pointsInRange, vertex2).reshape(-1,2)
    # pointsInRange,_ = sort_points(pointsInRange, centroid)

    pointsInRange = resample_points(pointsInRange, 100)


    


    
    if len(pointsInRange) == 4:
        pointsInRange = np.linspace(pointsInRange[0],pointsInRange[-1],5)

    # fig,ax = plt.subplots()
    # c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),cellAssign.reshape(params.numTestPoints,params.numTestPoints))
    # fig.colorbar(c,ax=ax)
    # ax.plot(points[:,0],points[:,1])
    # ax.plot(pointsInRange[:,0],pointsInRange[:,1])
    # ax.scatter(vertex1[0],vertex1[1],marker='*',color='r')
    # ax.scatter(vertex2[0],vertex2[1],marker='*',color='r')
    # plt.show()
        

    splineControlPoints,splineKnotPoints = fit_spline_to_path(pointsInRange, 10, 3,vertex1,vertex2)
    return splineControlPoints,splineKnotPoints,pointsInRange 

    
    
def plot_spline(spline,ax,c='g'):
    
    controlPoints = spline.c
    tf = spline.t[-1-spline.k]
    t = np.linspace(0, tf, 20)
    pos = spline(t)
    # ax.plot(pos[:,0], pos[:,1],c=c,marker='o')
    ax.plot(pos[:,0], pos[:,1],c=c,linewidth=3)

def find_generalized_voronoi_ridges(prob_list,verticies,radarParams,radarParamsCov,bounds,params,ax = None):
    radarParamsCovDeterminants = [np.linalg.det(cov) for cov in radarParamsCov]
    potentialRidges = {}
    for i in range(len(verticies)): 
        for j in range(i+1,len(verticies)):
            commonNeighbors = np.intersect1d(verticies[i]['neighbors'],verticies[j]['neighbors'])
            commonNeighbors = tuple(commonNeighbors)
            if len(commonNeighbors) == 2:
                if commonNeighbors not in potentialRidges:
                    potentialRidges[commonNeighbors] = [i,j]
                else:
                    if i not in potentialRidges[commonNeighbors]:
                        potentialRidges[commonNeighbors].append(i)
                    if j not in potentialRidges[commonNeighbors]:
                        potentialRidges[commonNeighbors].append(j)

    ignore = []
    for neighbors in potentialRidges.keys():
        # if len(potentialRidges[neighbors]) > 2:
        if len(potentialRidges[neighbors]) == 4:
            # points = []
            # for k in range(len(potentialRidges[neighbors])):
            #     point = verticies[potentialRidges[neighbors][k]]['point']
            #     points.append(point)
            # points = np.array(points)
            # _,indecies = sort_points(points,[params.bounds[0]/2,params.bounds[1]/2])
            # 
            
            

            point1 = verticies[potentialRidges[neighbors][0]]['point']
            point2 = verticies[potentialRidges[neighbors][1]]['point']
            point3 = verticies[potentialRidges[neighbors][2]]['point']
            point4 = verticies[potentialRidges[neighbors][3]]['point']
            points = np.array([point1,point2,point3,point4])
            _,indecies = sort_points(points,[params.bounds[0]/2,params.bounds[1]/2])
            ignore.append((potentialRidges[neighbors][indecies[0]],potentialRidges[neighbors][indecies[2]]))
            ignore.append((potentialRidges[neighbors][indecies[0]],potentialRidges[neighbors][indecies[3]]))
            ignore.append((potentialRidges[neighbors][indecies[1]],potentialRidges[neighbors][indecies[3]]))
            ignore.append((potentialRidges[neighbors][indecies[1]],potentialRidges[neighbors][indecies[2]]))
            
        
    ridges = {}
    for i in range(len(verticies)):
        for j in range(i+1,len(verticies)):
            commonNeighbors = np.intersect1d(verticies[i]['neighbors'],verticies[j]['neighbors'])
            if len(commonNeighbors) == 3:
                #need to add two ridges between these verticies
                ridgeLineControlPoints1,ridgeLineKnotPoints1,points1 = find_ridge_line(prob_list,commonNeighbors[0:2],verticies[i]['point'],verticies[j]['point'],radarParams,radarParamsCovDeterminants,verticies,i,j,bounds,params)
                t = np.linspace(0,1,1000)
                splinePoints1 = scipy.interpolate.BSpline(ridgeLineKnotPoints1,ridgeLineControlPoints1,3)(t)
                probPDLessThanThreshold1 = highPriorityHelperFunctions.get_prob_pd_less_than_threshold(splinePoints1, radarParams, radarParamsCov, params.probabilityOfDetectionThreshold, params.radarRecieveGain,0,params.radarWavelength, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance,params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
                minProb1 = np.min(probPDLessThanThreshold1)
                
                ridgeLineControlPoints2,ridgeLineKnotPoints2,points2 = find_ridge_line(prob_list,commonNeighbors[1:],verticies[i]['point'],verticies[j]['point'],radarParams,radarParamsCovDeterminants,verticies,i,j,bounds,params)
                splinePoints2 = scipy.interpolate.BSpline(ridgeLineKnotPoints2,ridgeLineControlPoints2,3)(t)
                probPDLessThanThreshold2 = highPriorityHelperFunctions.get_prob_pd_less_than_threshold(splinePoints2, radarParams, radarParamsCov, params.probabilityOfDetectionThreshold, params.radarRecieveGain,0,params.radarWavelength, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance,params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
                minProb2 = np.min(probPDLessThanThreshold2)

                if minProb2 > params.thresholdConfidence and minProb1 > params.thresholdConfidence:
                    if minProb1 > minProb2:
                        ridges[(i,j)] = {"control_points":ridgeLineControlPoints1,"knot_points":ridgeLineKnotPoints1}
                    else:
                        ridges[(i,j)] = {"control_points":ridgeLineControlPoints2,"knot_points":ridgeLineKnotPoints2}
                elif minProb1 > params.thresholdConfidence:
                    ridges[(i,j)] = {"control_points":ridgeLineControlPoints1,"knot_points":ridgeLineKnotPoints1}
                elif minProb2 > params.thresholdConfidence:
                    ridges[(i,j)] = {"control_points":ridgeLineControlPoints2,"knot_points":ridgeLineKnotPoints2}
                
                
            if len(commonNeighbors) == 2:
                
                if (i,j) in ignore or (j,i) in ignore:
                    continue
                else:
                    ridgeLineControlPoints,ridgeLineKnotPoints,points = find_ridge_line(prob_list,commonNeighbors,verticies[i]['point'],verticies[j]['point'],radarParams,radarParamsCovDeterminants,verticies,i,j,bounds,params)
                    #############
                    # fig,ax = plt.subplots()
                    # ax.scatter(points[:,0],points[:,1])
                    # ax.scatter(verticies[i]['point'][0],verticies[i]['point'][1],marker='*',color='r')
                    # ax.scatter(verticies[j]['point'][0],verticies[j]['point'][1],marker='*',color='r')
                    # spline = interpolate.BSpline(ridgeLineKnotPoints,ridgeLineControlPoints,3)
                    # plot_spline(spline,ax)
                    # plt.show()
                    ###############
                    t = np.linspace(0,1,1000)
                    splinePoints = scipy.interpolate.BSpline(ridgeLineKnotPoints,ridgeLineControlPoints,3)(t)
                    probPDLessThanThreshold = highPriorityHelperFunctions.get_prob_pd_less_than_threshold(splinePoints, radarParams, radarParamsCov, params.probabilityOfDetectionThreshold, params.radarRecieveGain,0,params.radarWavelength, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance,params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
                    minProb = np.min(probPDLessThanThreshold)
                    if minProb > params.thresholdConfidence:
                        ridges[(i,j)] = {"control_points":ridgeLineControlPoints,"knot_points":ridgeLineKnotPoints}
                        
                    if ax is not None:
                        spline = interpolate.BSpline(ridgeLineKnotPoints,ridgeLineControlPoints,3)
                        plot_spline(spline,ax)
    return ridges

def find_exterior_points(contourList,bounds):
    exteriorPoints = []
    for i,points in enumerate(contourList):
        tolerance = 50  # Adjust the tolerance as needed
        if (np.any(np.isclose(points[:, 0], 0, atol=tolerance)) or
            np.any(np.isclose(points[:, 0], bounds[0], atol=tolerance)) or
            np.any(np.isclose(points[:, 1], 0, atol=tolerance)) or
            np.any(np.isclose(points[:, 1], bounds[1], atol=tolerance))):
        # if np.any(points[:,0] == 0) or np.any(points[:,0] == params.bounds[0]) or np.any(points[:,1] == 0) or np.any(points[:,1] == params.bounds[1]):
            exteriorPoints.append(i)
    
        
    # exteriorPoints = []

    # for point_idx, region_idx in enumerate(vor.point_region):
    #     region = vor.regions[region_idx]
        
    #     if -1 in region:
    #         exteriorPoints.append(point_idx)
    
    return exteriorPoints

def find_edge_points(cellAssignment,bounds,numTestPoints):
    contour,_ = cv2.findContours(cellAssignment.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pixelDist = bounds[0]/numTestPoints
    points = contours_to_points(contour,pixelDist)

    # upperBound = np.max(points)
    # lowerBound = np.min(points)
    upperBound = bounds[0]
    lowerBound = 0
    
    corners = np.array([[lowerBound,lowerBound],[lowerBound,upperBound],[upperBound,upperBound],[upperBound,lowerBound]])
    mask = np.all(points[:,None]!=corners,axis=2).all(axis=1)
    mask = ~np.any((points[:, None] == corners).all(axis=2), axis=1)
    points = points[mask]
    tolerance = 50  # Define a tolerance level

    # Determine if the points are close to the lower or upper bounds
    close_to_x_lower = np.isclose(points[:, 0], lowerBound, atol=tolerance)
    close_to_x_upper = np.isclose(points[:, 0], upperBound, atol=tolerance)
    close_to_y_lower = np.isclose(points[:, 1], lowerBound, atol=tolerance)
    close_to_y_upper = np.isclose(points[:, 1], upperBound, atol=tolerance)

    # Combine conditions to select points close to any of the bounds
    edgePoints = points[close_to_x_lower | close_to_x_upper | close_to_y_lower | close_to_y_upper]

    # edgePoints = points[(points[:, 0] == lowerBound) | (points[:, 0] == upperBound) |
    #         (points[:, 1] == lowerBound) | (points[:, 1] == upperBound)]
    return edgePoints

def find_edge_vertex(cellAssignment,i,j,verticies,vertexIndex,bounds):
    # celliprob = prob_list[i]
    # celljprob = prob_list[j]
    # cellAssignment = np.where(celliprob < celljprob,0,1).reshape(params.numTestPoints,params.numTestPoints)
    numTestPoints = int(np.sqrt(cellAssignment.shape[0]))
    cellAssignmentI = np.where(cellAssignment == i,1,0).reshape(numTestPoints,numTestPoints)
    cellAssignmentJ = np.where(cellAssignment == j,1,0).reshape(numTestPoints,numTestPoints)

    # contour,_ = cv2.findContours(cellAssignmentI.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # pixelDist = params.bounds[0]/params.numTestPoints
    # points = contour[-1].squeeze()*pixelDist
    # upperBound = np.max(points)
    # lowerBound = np.min(points)
    # corners = np.array([[lowerBound,lowerBound],[lowerBound,upperBound],[upperBound,upperBound],[upperBound,lowerBound]])
    # mask = np.all(points[:,None]!=corners,axis=2).all(axis=1)
    # mask = ~np.any((points[:, None] == corners).all(axis=2), axis=1)
    # points = points[mask]

    # edgePoints = points[(points[:, 0] == lowerBound) | (points[:, 0] == upperBound) |
    #         (points[:, 1] == lowerBound) | (points[:, 1] == upperBound)]
    edgePointsI = find_edge_points(cellAssignmentI,bounds,numTestPoints)
    edgePointsJ = find_edge_points(cellAssignmentJ,bounds,numTestPoints)
    
    minDistance = np.inf
    closestPointI = None
    closestPointJ = None
    
    for point1 in edgePointsI:
        for point2 in edgePointsJ:
            dist = np.linalg.norm(point1 - point2)
            if dist < minDistance:
                minDistance = dist
                closestPointI = point1
                closestPointJ = point2

    if minDistance > 100:
        return None
    else:
        return (closestPointI+closestPointJ)/2
        
    # distances = np.linalg.norm(edgePoints - verticies[vertexIndex]['point'],axis=1)
    # fig,ax = plt.subplots()
    # mix = np.logical_and(cellAssignmentI,cellAssignmentJ)
    # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),mix.reshape(params.numTestPoints,params.numTestPoints))
    # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),cellAssignmentI.reshape(params.numTestPoints,params.numTestPoints))
    # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),cellAssignmentJ.reshape(params.numTestPoints,params.numTestPoints))
    # ax.plot(points[:,0],points[:,1])
    # ax.scatter(verticies[vertexIndex]['point'][0],verticies[vertexIndex]['point'][1],marker='*',color='r')
    # ax.scatter(edgePoints[:,0],edgePoints[:,1],marker='*',color='g')
    # plt.show()
    # return edgePoints[np.argmin(distances)]
    return (closestPointI+closestPointJ)/2

def find_possible_edge_vertex(prob_list,i,j,bounds,numTestPoints):
    celliprob = prob_list[i]
    celljprob = prob_list[j]
    
    cellAssignment = np.where(celliprob < celljprob,0,1).reshape(numTestPoints,numTestPoints)

    contour,_ = cv2.findContours(cellAssignment.astype(np.uint8).T, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pixelDist = bounds[0]/numTestPoints

    points = contours_to_points(contour,pixelDist)
    upperBound = np.max(points)
    lowerBound = np.min(points)
    corners = np.array([[lowerBound,lowerBound],[lowerBound,upperBound],[upperBound,upperBound],[upperBound,lowerBound]])
    # mask = np.all(points[:,None]!=corners,axis=2).all(axis=1)
    # mask = ~np.any((points[:, None] == corners).all(axis=2), axis=1)
    # points = points[mask]
    

    edgePoints = points[(points[:, 0] == lowerBound) | (points[:, 0] == upperBound) |
            (points[:, 1] == lowerBound) | (points[:, 1] == upperBound)]
    
    return edgePoints
    

def find_generalized_voronoi_edge_verticies(cellAssignments, exteriorPoints,verticies,points,bounds):
    currentVertex = len(verticies)

    interoirRidges = []
    
    for i in range(len(verticies)):
        for j in range(i+1,len(verticies)):
            if i != j:
                commonNeighbors = np.intersect1d(verticies[i]['neighbors'],verticies[j]['neighbors'])
                if len(commonNeighbors) == 2:
                    interoirRidges.append(commonNeighbors)

    center = np.mean(points,axis=0)
    for i in range(len(exteriorPoints)):
        for j in range(i+1,len(exteriorPoints)):
            vertex = find_edge_vertex(cellAssignments,exteriorPoints[i],exteriorPoints[j],verticies,currentVertex,bounds)
            if vertex is not None:
                verticies[currentVertex] = {"neighbors":[exteriorPoints[i],exteriorPoints[j]],"point":vertex,"edge":True}
                currentVertex += 1
        


    # for i in range(len(verticies)):
    #     intersection = np.intersect1d(verticies[i]['neighbors'],exteriorPoints)
    #     if len(intersection) ==2:
    #         vertex = find_edge_vertex(cellAssignments,intersection[0],intersection[1],verticies,i)

    #         verticies[currentVertex] = {"neighbors":intersection,"point":vertex,"edge":True}
    #         currentVertex += 1
    #     if len(intersection) == 3:
    #         for k in range(len(intersection)):
    #             for l in range(k+1,len(intersection)):
    #                 for ridge in interoirRidges:
    #                     if len(np.intersect1d(ridge,[intersection[k],intersection[l]])) > 2:
                            
    #                         tangent = points[intersection[k]] - points[intersection[l]]
    #                         tangent = tangent/np.linalg.norm(tangent)
    #                         normal = np.array([-tangent[1],tangent[0]])
    #                         midpoint = (points[intersection[k]] + points[intersection[l]])/2
    #                         direction = np.sign(np.dot(midpoint - center,normal))*normal
                            
    #                         possibleVerticies = find_possible_edge_vertex(prob_list,intersection[k],intersection[l])
    #                         vertexVectors = possibleVerticies - verticies[i]['point']
    #                         vertexVectors = vertexVectors/np.linalg.norm(vertexVectors,axis=1)[:,None]

    #                         dotProducts = np.dot(vertexVectors,direction)
    #                         closestVectorIndex = np.argmax(dotProducts)
                            
    #                         vertex = possibleVerticies[closestVectorIndex]
    #                         verticies[currentVertex] = {"neighbors":[intersection[k],intersection[l]],"point":vertex,"edge":True}
    #                         currentVertex += 1
                    
            

    return verticies

def add_boundary_segment(boundarySegments, segment, vertex1,vertex2, radarParams, radarParamsCov,params):
    points = np.linspace(segment[0],segment[1],100)
    probPdLessThanThreshold = highPriorityHelperFunctions.get_prob_pd_less_than_threshold(points, radarParams, radarParamsCov, params.probabilityOfDetectionThreshold, params.radarRecieveGain,0,params.radarWavelength, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance,params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
    minProb = np.min(probPdLessThanThreshold)
    if minProb > params.thresholdConfidence:
        boundarySegments[(vertex1,vertex2)] = segment
    
    return boundarySegments
    
    

def find_boundary_segments(verticies,radarParams,radarParamsCov,bounds,params):
    startVertexIndex = len(verticies)
    verticies[len(verticies)] = {"point":np.array([0,0]),"edge":True,"neighbors":[],"type":"start"}
    verticies[len(verticies)] = {"point":np.array([bounds[0],0]),"edge":True,"neighbors":[]}
    verticies[len(verticies)] = {"point":np.array([0,bounds[1]]),"edge":True,"neighbors":[]}
    endVertexIndex = len(verticies)
    verticies[len(verticies)] = {"point":np.array([bounds[0],bounds[1]]),"edge":True,"neighbors":[],"type":"end"}

    boundarySegments = {}

    edgeVerticies = np.array([verticies[vertex]["point"] for vertex in verticies if verticies[vertex]['edge']])
    edgeVerticiesIndicies = np.array([vertex for vertex in verticies if verticies[vertex]['edge']])

    leftEdgeVertexIndicies = np.where(np.isclose(edgeVerticies[:,0],0,atol=50))[0]
    sortedLeftEdgeVertexIndicies = leftEdgeVertexIndicies[np.argsort(edgeVerticies[leftEdgeVertexIndicies][:,1])]
    rightEdgeVertexIndicies = np.where(np.isclose(edgeVerticies[:,0],bounds[0],atol=50))[0]
    sortedRightEdgeVertexIndicies = rightEdgeVertexIndicies[np.argsort(edgeVerticies[rightEdgeVertexIndicies][:,1])]
    topEdgeVertexIndicies = np.where(np.isclose(edgeVerticies[:,1],bounds[1],atol=50))[0]
    sortedTopEdgeVertexIndicies = topEdgeVertexIndicies[np.argsort(edgeVerticies[topEdgeVertexIndicies][:,0])]
    bottomEdgeVertexIndicies = np.where(np.isclose(edgeVerticies[:,1],0,atol=50))[0]
    sortedBottomEdgeVertexIndicies = bottomEdgeVertexIndicies[np.argsort(edgeVerticies[bottomEdgeVertexIndicies][:,0])]
    
    for i in range(len(sortedLeftEdgeVertexIndicies)-1):
        index = sortedLeftEdgeVertexIndicies[i]
        # boundarySegments[(edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedLeftEdgeVertexIndicies[i+1]])] = np.array([edgeVerticies[index],edgeVerticies[sortedLeftEdgeVertexIndicies[i+1]]])
        boundarySegments = add_boundary_segment(boundarySegments, np.array([edgeVerticies[index],edgeVerticies[sortedLeftEdgeVertexIndicies[i+1]]]), edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedLeftEdgeVertexIndicies[i+1]], radarParams, radarParamsCov,params)
    for i in range(len(sortedRightEdgeVertexIndicies)-1):
        index = sortedRightEdgeVertexIndicies[i]
        # boundarySegments[(edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedRightEdgeVertexIndicies[i+1]])] = np.array([edgeVerticies[index],edgeVerticies[sortedRightEdgeVertexIndicies[i+1]]])
        boundarySegments = add_boundary_segment(boundarySegments, np.array([edgeVerticies[index],edgeVerticies[sortedRightEdgeVertexIndicies[i+1]]]), edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedRightEdgeVertexIndicies[i+1]], radarParams, radarParamsCov,params)
    for i in range(len(sortedTopEdgeVertexIndicies)-1):
        index = sortedTopEdgeVertexIndicies[i]
        # boundarySegments[(edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedTopEdgeVertexIndicies[i+1]])] = np.array([edgeVerticies[index],edgeVerticies[sortedTopEdgeVertexIndicies[i+1]]])
        boundarySegments = add_boundary_segment(boundarySegments, np.array([edgeVerticies[index],edgeVerticies[sortedTopEdgeVertexIndicies[i+1]]]), edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedTopEdgeVertexIndicies[i+1]], radarParams, radarParamsCov, params)
    for i in range(len(sortedBottomEdgeVertexIndicies)-1):
        index = sortedBottomEdgeVertexIndicies[i]
        # boundarySegments[(edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedBottomEdgeVertexIndicies[i+1]])] = np.array([edgeVerticies[index],edgeVerticies[sortedBottomEdgeVertexIndicies[i+1]]])
        boundarySegments = add_boundary_segment(boundarySegments, np.array([edgeVerticies[index],edgeVerticies[sortedBottomEdgeVertexIndicies[i+1]]]), edgeVerticiesIndicies[index],edgeVerticiesIndicies[sortedBottomEdgeVertexIndicies[i+1]], radarParams, radarParamsCov, params)
        
    return boundarySegments,startVertexIndex,endVertexIndex


def remove_radar_from_cell_assignments(cellAssignments,radarIndeciesToIgnore):
    for i in radarIndeciesToIgnore:
        cellAssignments[cellAssignments > i] -= 1
    return cellAssignments
    
    
def find_generalized_voronoi(radarParams, radarParamsCov, bounds,params):
    points = radarParams[:,0:2]
    # vor = Voronoi(points[:, 0:2])
    numTestPoints = 500
    X_test = create_test_points(numTestPoints,bounds)

    prob_list = create_probability_of_detection_below_threshold_grid_list(X_test,radarParams,radarParamsCov,params)
    # Z = weighted_voronoi_uncertain_radar_grid_method(params.X_test,prob_list,radarParams,radarParamsCov,useUpperBound=False,ax=ax)

    # closestNeighborTriples = get_closest_neighbor_triplets(vor)

    verticies,contourPoints,cellAssignments,radarIndeciesToIgnore = find_generalized_voronoi_verticies(prob_list,bounds=params.bounds,numTestPoints=numTestPoints)
    
    cellAssignments = remove_radar_from_cell_assignments(cellAssignments,radarIndeciesToIgnore)

    
    radarParams = np.delete(radarParams,radarIndeciesToIgnore,axis=0)
    radarParamsCov = np.delete(radarParamsCov,radarIndeciesToIgnore,axis=0)
    
    
         
    exteriorPoints = find_exterior_points(contourPoints,bounds)
    # verticies = find_generalized_voronoi_edge_verticies(prob_list, exteriorPoints, verticies,points)
    verticies = find_generalized_voronoi_edge_verticies(cellAssignments, exteriorPoints, verticies,points,bounds)
    boundarySegments,startVertexIndex,endVertexIndex = find_boundary_segments(verticies,radarParams,radarParamsCov,bounds,params)

    # fig,ax = plt.subplots()
    # ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),np.argmin(prob_list,axis=0).reshape(params.numTestPoints,params.numTestPoints))
    # for vertex in verticies:
    #     ax.scatter(verticies[vertex]["point"][0],verticies[vertex]["point"][1],marker='*',color='r')
    #     ax.text(verticies[vertex]["point"][0],verticies[vertex]["point"][1],str(vertex))
    # plt.show()
    
    
    
    ridges = find_generalized_voronoi_ridges(cellAssignments,verticies,radarParams,radarParamsCov,bounds,params,ax=None)

    return verticies,ridges,boundarySegments,startVertexIndex,endVertexIndex,cellAssignments
    
    
    
def plot_generalized_voronoi(verticies,ridges,boundarySegments,ax):
    c = 'g'
    for ridge in ridges:
        controlPoints = ridges[ridge]["control_points"]
        knotPoints = ridges[ridge]["knot_points"]
        spline = interpolate.BSpline(knotPoints,controlPoints,3)
        plot_spline(spline,ax,c)
    for vertex in verticies:
        ax.scatter(verticies[vertex]["point"][0],verticies[vertex]["point"][1],marker='*',color=c,s=100)
    for segment in boundarySegments:
        ax.plot(boundarySegments[segment][:,0],boundarySegments[segment][:,1],c=c,linewidth=3)

def integrate_spline(spline):
    numPoints = 100
    t = np.linspace(0,1,numPoints)
    points = spline(t)
    distance = np.linalg.norm(points[1:] - points[:-1],axis=1)
    distance = np.sum(distance)
    return distance

def create_adjacency_matrix_from_ridges_and_boundary(ridges,boundarySegments,verticies):

    adjacencyMatrix = np.zeros((len(verticies),len(verticies)))

    for ridge in ridges.keys():
        spline = interpolate.BSpline(ridges[ridge]["knot_points"],ridges[ridge]["control_points"],3)
        splineDistance = integrate_spline(spline)
        # straitlineDistance = np.linalg.norm(np.array(verticies[ridge[0]]["point"]) - np.array(verticies[ridge[1]]["point"]))
        i,j = ridge
        adjacencyMatrix[i,j] = splineDistance
        adjacencyMatrix[j,i] = splineDistance
    
    for segment in boundarySegments.keys():
        i,j = segment
        distance = np.linalg.norm(boundarySegments[segment][0] - boundarySegments[segment][1])
        adjacencyMatrix[i,j] = distance
        adjacencyMatrix[j,i] = distance
    
    return adjacencyMatrix

def create_graph_and_find_shortest_path(adejacenyMatrix,startIndex,endIndex):
    g = igraph.Graph.Weighted_Adjacency(adejacenyMatrix.tolist(),mode=igraph.ADJ_UNDIRECTED,attr="weight")
    path = g.get_shortest_paths(startIndex,to=endIndex,weights=g.es["weight"])

    return path[0]

def path_to_points(path,ridges,boundarySegments,verticies,spacing):

    pointsCombined = []

    previousPoint = verticies[path[0]]['point']

    for i in range(len(path)-1):
        ridgeVertex = (path[i],path[i+1])
        flippedRidgeVertex = (path[i+1],path[i])
        numPoints = int(np.linalg.norm(verticies[path[i]]['point'] - verticies[path[i+1]]['point'])/spacing)+1
        if ridgeVertex in ridges.keys():
            spline = interpolate.BSpline(ridges[ridgeVertex]["knot_points"],ridges[ridgeVertex]["control_points"],3)
            points = spline(np.linspace(0,1,numPoints))
        elif ridgeVertex in boundarySegments.keys():
            points = np.linspace(verticies[path[i]]['point'],verticies[path[i+1]]['point'],numPoints)
        elif flippedRidgeVertex in ridges.keys():
            spline = interpolate.BSpline(ridges[flippedRidgeVertex]["knot_points"],ridges[flippedRidgeVertex]["control_points"],3)
            points = spline(np.linspace(0,1,numPoints))
        elif flippedRidgeVertex in boundarySegments.keys():
            points = np.linspace(verticies[path[i+1]]['point'],verticies[path[i]]['point'],numPoints)
        
        previousPointIndex = np.argmin(np.linalg.norm(points - previousPoint,axis=1))
        if previousPointIndex != 0:
            points = np.flip(points,axis=0)
        previousPoint = points[-1]

        # fig, ax = plt.subplots()
        # ax.plot(points[:,0],points[:,1])
        # ax.scatter(verticies[path[i]]['point'][0],verticies[path[i]]['point'][1],marker='*',color='r')
        # ax.scatter(verticies[path[i+1]]['point'][0],verticies[path[i+1]]['point'][1],marker='*',color='r')
        # plt.show()
        pointsCombined.append(points)
    
    
    return np.vstack(pointsCombined)

def find_initial_trajectory_uncertain_radar(radarParams,radarParamsCov,spacing,bounds,params,ax=None):
    # fig,ax = plt.subplots()
    start = time()
    verticies,ridges,boundarySegments,startVertexIndex,endVertexIndex,cellAssinments = find_generalized_voronoi(radarParams,radarParamsCov,bounds,params)
    print("Time",time()-start)

    adjecencyMatrix = create_adjacency_matrix_from_ridges_and_boundary(ridges,boundarySegments,verticies)
    path = create_graph_and_find_shortest_path(adjecencyMatrix,startVertexIndex,endVertexIndex)

    
    #test code
    # vor = Voronoi(radarParams[:,0:2])
    


    if len(path) == 0:
        return path

    pathPoints = path_to_points(path,ridges,boundarySegments,verticies,spacing)

    if ax is not None:
        plot_generalized_voronoi(verticies,ridges,boundarySegments,ax)
        if pathPoints is not None:
            ax.plot(pathPoints[:,0],pathPoints[:,1],color='r')
        # ax.plot(pathPoints[:,0],pathPoints[:,1],color='r')
    
        plt.show()
    return pathPoints
    

        
    
def load_estimated_params(dataFilePath,dataIndex,numRadar):
    radarParamsAll = []
    radarParamsCovAll = []
    for i in range(numRadar):
        radarParams = np.load(dataFilePath+"/radar_"+str(i)+"/estimated_params/"+str(dataIndex)+".npy")
        radarParamsCov = np.load(dataFilePath+"/radar_"+str(i)+"/estimated_params_cov/"+str(dataIndex)+".npy")
        if len(radarParams) != 0:
            radarParamsAll.append(radarParams)
            radarParamsCovAll.append(radarParamsCov)
    
    return np.array(radarParamsAll),np.array(radarParamsCovAll)
    
    

def main():
    dataFilePath = "saved_data/mc_runs/4052005650/optimization/"

    importDir = "saved_data.mc_runs.4052005650.optimization.params"
    params = importlib.import_module(importDir)

    radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))


    numFiles = 2161
    dataIndex = 1800

    radarParams, radarParamsCov = load_estimated_params(dataFilePath,dataIndex,len(radarList))


    Z = safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold,params.thresholdConfidence,radarParams,radarParamsCov, params.radarRecieveGain,params.radarRecieveGainPriorVariance, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)
    # Z = Z > params.thresholdConfidence


    
    fig, ax = plt.subplots()
    c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints))
    fig.colorbar(c, ax=ax)

    pathPoints = find_initial_trajectory_uncertain_radar(radarParams,radarParamsCov,100,bounds=params.bounds,params=params,ax=ax)



    



        
    
    ax.set_aspect('equal')
    
    

    ax.set_xlim([-1000,params.bounds[0]+1000])
    ax.set_ylim([-1000,params.bounds[1]+1000])


    plt.show()
    

if __name__ == "__main__":
    main()
    

    

