import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
from time import time

from bspline.matrix_evaluation import matrix_bspline_derivative_evaluation_for_dataset, matrix_bspline_evaluation_for_dataset

from main_helper import create_radar_list
from probabilityOfDetectionJax import ground_truth_probability_of_detection,compute_probability_of_detection_at_points_multiple_radar




@partial(jit, static_argnums=(2,3))
def create_unclamped_knot_points(t0, tf, numControlPoints,splineOrder):
    internalKnots = jnp.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = jnp.concatenate((jnp.linspace(t0-splineOrder*h,t0-h,splineOrder), internalKnots, jnp.linspace(tf+h,tf+splineOrder*h,splineOrder)))
    
    return knots

@partial(jit, static_argnums=(2,3,4))
def evaluate_spline_derivative(controlPoints, knotPoints,splineOrder, derivativeOrder, numSamplesPerInterval):
    scaleFactor = knotPoints[-splineOrder-1]/(len(knotPoints)-2*splineOrder-1)
    return matrix_bspline_derivative_evaluation_for_dataset(derivativeOrder, scaleFactor, controlPoints.T, knotPoints, numSamplesPerInterval)

@partial(jit, static_argnums=(1,2))
def evaluate_spline(controlPoints, knotPoints,numSamplesPerInterval):
    return matrix_bspline_evaluation_for_dataset(controlPoints.T, knotPoints, numSamplesPerInterval)

@partial(jit, static_argnums=(2,3))
def get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval):
    print(controlPoints.shape)
    numControlPoints = int(len(controlPoints)/2)
    controlPoints = controlPoints.reshape((numControlPoints,2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints,splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints,knotPoints,splineOrder,1,numSamplesPerInterval)
    return jnp.linalg.norm(out_d1,axis=1)

@partial(jit, static_argnums=(2,3,4,5)) 
def get_pd_along_spline(controlpoints, tf, radarlist, numcontrolpoints, splineorder, numsamplesperinterval, radaroutputpower, radartransmitgain, radarrecievegainpriormean, radarwavelengthpriormean, agentradarcrosssection, radarpulsewidth, radarsystemtemperaturepriormean, radarprobabilityoffalsealarmpriormean):
    controlpoints = controlpoints.reshape((numcontrolpoints,2))
    knotpoints = create_unclamped_knot_points(0, tf, numcontrolpoints,splineorder)
    pos = evaluate_spline(controlpoints,knotpoints,numsamplesperinterval)
    pd = ground_truth_probability_of_detection(pos, radarlist, radarwavelengthpriormean, agentradarcrosssection, radarpulsewidth, radarsystemtemperaturepriormean, radarprobabilityoffalsealarmpriormean)
    return pd
    
@partial(jit, static_argnums=(2,3))
def get_spline_turn_rate(controlPoints, tf, splineOrder,numSamplesPerInterval):
    numControlPoints = int(len(controlPoints)/2)
    controlPoints = controlPoints.reshape((numControlPoints,2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints,splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints,knotPoints, splineOrder,1,numSamplesPerInterval)
    out_d2 = evaluate_spline_derivative(controlPoints,knotPoints, splineOrder,2,numSamplesPerInterval)
    v = jnp.linalg.norm(out_d1,axis=1)
    u = jnp.cross(out_d1,out_d2) / (v**2)
    return u

@jit
def dist_of_points_to_line_segment(p1,p2,points): # p3 is the point
    #https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    # px = x2-x1
    # py = y2-y1
    diff = p2-p1
    # px = x2-x1
    # py = y2-y1

    norm = jnp.sum(diff**2)
    # norm = px*px + py*py

    # u =  ((x3 - x1) * diff[0] + (y3 - y1) * diff[1]) / norm
    u = jnp.dot(points-p1,diff) / norm
    u =u.reshape((-1,1))

    u = jnp.clip(u, 0, 1)

    # x = x1 + u * diff[0]
    # y = y1 + u * diff[1]
    p = p1 + u * diff


    # dx = x - x3
    # dy = y - y3

    dp = p - points

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    # dist = (dx*dx + dy*dy)
    dist = jnp.linalg.norm(dp,axis=1)

    return dist

    
    

@partial(jit, static_argnums=(2,3,4)) 
def get_pd_cov_and_mean_along_spline(controlpoints, tf, numcontrolpoints, splineorder, numsamplesperinterval, estimatedRadarParamsList, estimatedRadarParamsCovList, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar):
    controlpoints = controlpoints.reshape((numcontrolpoints,2))
    knotpoints = create_unclamped_knot_points(0, tf, numcontrolpoints,splineorder)
    pos = evaluate_spline(controlpoints,knotpoints,numsamplesperinterval)
    pdMean, pdCov = compute_probability_of_detection_at_points_multiple_radar(pos, estimatedRadarParamsList, estimatedRadarParamsCovList, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar)
    return pd


if __name__ == '__main__':

    p1 = jnp.array([0.0,0.0])
    p2 = jnp.array([1.0,1.0])
    points = jnp.array([[0.0,1.0],[1.0,0.0]])
    start = time()
    print(dist_of_points_to_line_segment(p1,p2,points)) 
    print("time to complie", time()-start)
    start  = time()
    time_to_run = 0
    for i in range(1000):
        dist_of_points_to_line_segment(p1,p2,points)
    print("average time to run", (time()-start)/1000)
    