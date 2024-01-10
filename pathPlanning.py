import numpy as np
from scipy.special import erfinv
import params
from scipy import interpolate
from pyoptsparse import Optimization, OPT, IPOPT


class SplinePathPlanningLowPriority():
    def __init__(self):
        self.numMeasurements = 0
        self.bestMesurementPlot = None
        self.splinePath = None
        self.previousEmittorParamsList = []

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
    
    def newParamsDifferent(self, estimatedRadarParamsList):
        for i in range(len(estimatedRadarParamsList)):
            for j in range(len(estimatedRadarParamsList[i])):
                if estimatedRadarParamsList[i][j]!=self.previousEmittorParamsList[i][j]:
                    return True
                
        return False
    
    def update_path(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, numMeasurements):
        if numMeasurements != self.numMeasurements and len(estimatedRadarParamsList)>0:
            self.numMeasurements = numMeasurements
            if len(estimatedRadarParamsList)>0:
                if (len(self.previousEmittorParamsList)!=len(estimatedRadarParamsList) or self.newParamsDifferent(estimatedRadarParamsList)):
                    bestMeasurementLoc = self.optimize_next_best_measurement(currPos, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap)
                    self.optimize_spline_path(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, bestMeasurementLoc)
                    self.numMeasurements = numMeasurements
                    self.previousEmittorParamsList = estimatedRadarParamsList.copy()

        
    
    def create_evenly_spaced_control_points(self, start, stop, numControlPoints):
        xPoints = np.linspace(start[0], stop[0], numControlPoints, endpoint=False) 
        yPoints = np.linspace(start[1], stop[1], numControlPoints, endpoint=False) 
        points = np.hstack((xPoints.reshape((len(xPoints),1)), yPoints.reshape((len(xPoints),1))))
        return points
    
    def spline_objective_function(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, controlPoints, tf):
        spline = self.spline_seg(controlPoints, 0, tf)
        tObjective = np.linspace(0,tf, params.numObjectiveFunctionSamples)
        pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(spline(tObjective), estimatedRadarParamsList, estimatedRadarParamsCovList, False)
        return np.max(pdMean)

    def optimize_spline_path(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, bestMeasurementLocation):
        straitLineDist = np.linalg.norm(currPos[0:2] - bestMeasurementLocation)
        initialControlPoints = self.create_evenly_spaced_control_points(currPos, bestMeasurementLocation, params.numControlPoints-2)
        def objective_function(xDict):
            controlPoints = np.zeros((params.numControlPoints,2))
            controlPoints[0,:] = currPos[0:2]
            controlPoints[-1,:] = bestMeasurementLocation 
            controlPoints[1:-1,:] = xDict['control_points'].reshape((params.numControlPoints-2,2))
            tf = xDict['tf']
            funcs = {}
            obj = self.spline_objective_function(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, controlPoints, tf[0])
            funcs['obj'] = obj
            return funcs, False
            

        optProb = Optimization("low priority path", objective_function)
        optProb.addVarGroup(name = "control_points", nVars = 2*(params.numControlPoints-2), varType = 'c', value = initialControlPoints.reshape((2*(params.numControlPoints-2))), lower = 0, upper=params.bounds[1])
        optProb.addVarGroup(name = "tf", nVars = 1, varType = 'c', value = straitLineDist/params.agentSpeed, lower = 0, upper=params.pathLengthMultiplier * straitLineDist/params.agentSpeed)
        optProb.addObj("obj")
        opt = OPT("ipopt")
        opt.options['print_level'] = 5
        opt.options['tol'] = 1e-10
        sol = opt(optProb, sens = 'FD')
        controlPoints = np.zeros((params.numControlPoints,2))
        controlPoints[0,:] = currPos[0:2]
        controlPoints[-1,:] = bestMeasurementLocation 
        controlPoints[1:-1,:] = sol.xStar['control_points'].reshape((params.numControlPoints-2,2))
        tf = sol.xStar['tf']
        self.splinePath = self.spline_seg(controlPoints, 0, tf)
        print()
    

    def next_measurement_covariance_determinant(self, pos, estimatedRadarParams_list, estimatedRadarCovariance_list):
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

    def chance_constraint(self,rho, delta, mean, var):
        return (mean - rho) < (-erfinv(2*delta-1)*np.sqrt(2*var))

    def objective_function_chance_constraints(self, X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
        objectivFunctionVal = np.zeros((len(X_test),1))
        constraintMet = np.zeros((len(X_test),1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        rho = params.probabilityOfDetectionThreshold
        delta = params.thresholdConfidence
        for i, mean in enumerate(pdMean):
            var = pdVar[i]
            objectivFunctionVal[i] = self.next_measurement_covariance_determinant(X_test[i,:], estimatedRadarParams, estimatedRadarParamsCov)   
            constraintMet[i] = self.chance_constraint(rho, delta, mean, var)
        return objectivFunctionVal,  constraintMet

    def objective_function(self, X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
        alpha = params.lowPrioritySafetyBestMeasurementTradeoff
        objectivFunctionVal = np.zeros((len(X_test),1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        for i, mean in enumerate(pdMean):
            var = pdVar[i]
            objectivFunctionVal[i] = alpha * self.next_measurement_covariance_determinant(X_test[i,:], estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * mean 
        return objectivFunctionVal

    def objective_function_at_pos(self, pos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
        alpha = params.lowPrioritySafetyBestMeasurementTradeoff
        pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar([pos], estimatedRadarParams, estimatedRadarParamsCov, False)
        # var = pdVar[i]
        objectivFunctionVal = alpha * self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * pdMean 
        return objectivFunctionVal


    def optimize_next_best_measurement(self, currPos, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap):
        print("currPos", currPos)
        def objective_function(xdict):
            pos = xdict['pos']
            # obj = -next_measurement_covariance_determinant(pos, estimatedParams_list, estimatedRadarCovariance_list)
            obj = -self.objective_function_at_pos(pos, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap)
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
        
        
        

            
    def plot_spline(self,ax,spl,num_points):
        t0 = spl.t[0]
        tf = spl.t[-1]
        t = np.linspace(t0, tf, num_points, endpoint=True)
        x = spl(t)[:,0]
        y = spl(t)[:,1]
        ax.plot(x,y)
        # plt.scatter(x,y,c = spline_color)
        control_points = spl.c
        ax.plot(control_points[:, 0], control_points[:, 1], 'k--', label='Control polygon', marker='o', zorder = 100000)


    def plot_objective_and_constraint(self, ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos, numMeasurements):
        # objectiveFunctionVal, constraintMet = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        objectiveFunctionVal = self.objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)

        # objectiveFunctionVal[constraintMet ==0] = -1

        c = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), objectiveFunctionVal.reshape((params.numTestPoints,params.numTestPoints)))
        if self.splinePath is not None:
            self.plot_spline(ax, self.splinePath, 100)
            
        # if numMeasurements != self.numMeasurements:
        #     bestPos = self.optimize_next_best_measurement(currPos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        #     print("bestPos", bestPos)
        #     if self.bestMesurementPlot is not None:
        #         self.bestMesurementPlot.remove()
        #     ax.scatter(bestPos[0], bestPos[1],zorder = 1000000)
        #     self.numMeasurements = numMeasurements
        return c
    
    
    
    
    