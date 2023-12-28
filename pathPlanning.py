import numpy as np
from scipy.special import erfinv
import params

# class SplinePathPlanning():
    

def chance_constraint(rho, delta, mean, var):
    return (mean - rho) < (-erfinv(2*delta-1)*np.sqrt(2*var))

def objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
    objectivFunctionVal = np.zeros((len(X_test),1))
    constraintMet = np.zeros((len(X_test),1))
    # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
    pdMean = probabilityOfDetectionMap.pdMap
    pdVar = probabilityOfDetectionMap.pdCovMap
    rho = params.probabilityOfDetectionThreshold
    delta = params.thresholdConfidence
    for i, mean in enumerate(pdMean):
        var = pdVar[i]
        objectivFunctionVal[i] = mean,
        constraintMet[i] = chance_constraint(rho, delta, mean, var)
    return objectivFunctionVal,  constraintMet 


def plot_objective_and_constraint(ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
    objectiveFunctionVal, constraintMet = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)

    objectiveFunctionVal[constraintMet ==0] = -1

    c = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), objectiveFunctionVal.reshape((params.numTestPoints,params.numTestPoints)))
    return c
    
    