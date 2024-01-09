import numpy as np
from scipy.special import erfinv
import params
from scipy import interpolate
from pyoptsparse import Optimization, OPT, IPOPT


class SplinePathPlanning():

    def spline_seg(self,control_points,t0,tf):
        '''
        Wrapper function for scipy bspline class, this creates a clamped bpline with evenly spaced knot points (expect for first few and last few which are repeated)
            with control points and start and stop time specified by parameters
        params:
            control_points: control points of the spline
            t0: intial time of the spline (usually 0)
            tf: final time of the spline (this is changed by the optimizer
        returns:
            scipy bspline class
        '''

        #the number of control points
        l = len(control_points)

        #create evenly spaced knot points
        t = np.linspace(t0, tf, l - 2, endpoint=True)

        #add repeated knot points at begining and end
        t = np.append([t0, t0, t0], t)
        t = np.append(t, [tf, tf, tf])

        #create scipy bpline object
        spline = interpolate.BSpline(t, control_points, 3)

        return spline
    
    


    

def next_measurement_covariance_determinant(pos, estimatedRadarParams_list, estimatedRadarCovariance_list):
    x = pos[0]
    y = pos[1]
    out = 0
    currCovSum = 0
    for i,estimatedRadarParams in enumerate(estimatedRadarParams_list):
        estimatedRadarCovariance = estimatedRadarCovariance_list[i]
        x_em = estimatedRadarParams[0]
        y_em = estimatedRadarParams[1]
        erp = estimatedRadarParams[2]
        H = params.measurement_jacobian(x_em, y_em, erp, x, y)
        R = params.measurementCov
        K = estimatedRadarCovariance @ H.T @ np.linalg.inv(H@estimatedRadarCovariance@H.T + R)
        nextCovariance = (np.eye(3) - K@H)@estimatedRadarCovariance
        currentCovDet =np.linalg.det(estimatedRadarCovariance) 
        currCovSum += currentCovDet
        out += (currentCovDet - np.linalg.det(nextCovariance))

    # nextCovariance = estimatedRadarCovariance - estimatedRadarCovariance@H.T@np.linalg.inv(H@estimatedRadarCovariance@H.T+R)@H@estimatedRadarCovariance
    # return 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(estimatedRadarCovariance)) - 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(nextCovariance))
    return out/currCovSum

def chance_constraint(rho, delta, mean, var):
    return (mean - rho) < (-erfinv(2*delta-1)*np.sqrt(2*var))

def objective_function_chance_constraints(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
    objectivFunctionVal = np.zeros((len(X_test),1))
    constraintMet = np.zeros((len(X_test),1))
    # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
    pdMean = probabilityOfDetectionMap.pdMap
    pdVar = probabilityOfDetectionMap.pdCovMap
    rho = params.probabilityOfDetectionThreshold
    delta = params.thresholdConfidence
    for i, mean in enumerate(pdMean):
        var = pdVar[i]
        objectivFunctionVal[i] = next_measurement_covariance_determinant(X_test[i,:], estimatedRadarParams, estimatedRadarParamsCov)   
        constraintMet[i] = chance_constraint(rho, delta, mean, var)
    return objectivFunctionVal,  constraintMet

def objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
    alpha = 1.0
    objectivFunctionVal = np.zeros((len(X_test),1))
    # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
    pdMean = probabilityOfDetectionMap.pdMap
    pdVar = probabilityOfDetectionMap.pdCovMap
    for i, mean in enumerate(pdMean):
        var = pdVar[i]
        objectivFunctionVal[i] = alpha * next_measurement_covariance_determinant(X_test[i,:], estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * mean 
    return objectivFunctionVal

def objective_function_at_pos(pos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
    alpha = 0.7
    pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar([pos], estimatedRadarParams, estimatedRadarParamsCov)
    # var = pdVar[i]
    objectivFunctionVal = alpha * next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * pdMean 
    return objectivFunctionVal


def optimize_next_best_measurement(currPos, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap):
    print("currPos", currPos)
    def objective_function(xdict):
        pos = xdict['pos']
        # obj = -next_measurement_covariance_determinant(pos, estimatedParams_list, estimatedRadarCovariance_list)
        obj = -objective_function_at_pos(pos, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap)
        funcs = {}
        funcs['obj'] = obj
        fail = False
        return funcs, fail
    optProb = Optimization("find best measurement location", objective_function)
    optProb.addVarGroup(name = "pos", nVars = 2, varType = 'c', value = currPos[0:2], lower = 0, upper=params.bounds[1])
    optProb.addObj("obj")
    opt = OPT("ipopt")
    opt.options['print_level'] = 5
    opt.options['tol'] = 1e-10
    sol = opt(optProb, sens = 'FD')
    return sol.xStar["pos"]
    
    
    

        


def plot_objective_and_constraint(ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos):
    bestPos = optimize_next_best_measurement(currPos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
    print("bestPos", bestPos)
    # objectiveFunctionVal, constraintMet = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
    objectiveFunctionVal = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)

    # objectiveFunctionVal[constraintMet ==0] = -1

    c = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), objectiveFunctionVal.reshape((params.numTestPoints,params.numTestPoints)))
    ax.scatter(bestPos[0], bestPos[1],zorder = 1000000)
    return c
    
    
    
    
    