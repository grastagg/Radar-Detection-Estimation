import numpy as np
from scipy.special import erfinv
import params
from scipy import interpolate
from pyoptsparse import Optimization, OPT, IPOPT
from matplotlib.patches import Circle


class SplinePathPlanningLowPriority():
    def __init__(self):
        self.numMeasurements = 0
        self.bestMesurementPlot = None
        self.splinePath = None
        self.previousEmittorParamsList = []
        self.splinePathPlot = None
        self.splineControlPointsPlot = None
        self.currentSplineTime = 0
        self.currentVelocity = (params.velocityBounds[0]+params.velocityBounds[1])/2
        self.first = False

        self.useSpline = False
        self.bestMeasurementLoc = None
        self.kp = 1

        self.bestMeasurementLocPlot = None
        self.firstPlot = True

    def spline_seg(self,control_points,t):
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


        #create scipy bpline object
        spline = interpolate.BSpline(t, control_points, 3)

        return spline
    
    def get_control(self, dt, currentPose):
        u = 0
        v = (params.velocityBounds[0]+params.velocityBounds[1])/2
        if self.useSpline:
            if self.splinePath is not None:
                u,v = self.get_turn_rate_and_velocity(np.array([self.currentSplineTime]), self.splinePath)
                u = u[0]
                v = v[0]
                self.currentSplineTime += dt
            self.currentVelocity = v
        else:
            if self.bestMeasurementLoc is not None:
                u,v = self.get_turn_rate_and_velocity_waypoint(self.bestMeasurementLoc, currentPose)
                
        return u,v

    def get_turn_rate_and_velocity_waypoint(self, bestMeasurementLoc, currentPose):
        desiredHeading = np.arctan2(bestMeasurementLoc[1]-currentPose[1], bestMeasurementLoc[0]-currentPose[0])
        v = (params.velocityBounds[0]+params.velocityBounds[1])/2
        u = self.kp * (desiredHeading - currentPose[2])
        return u,v
            
    def get_turn_rate_and_velocity(self, t, spl):
        out_d1 = spl.derivative(1)(t)
        out_d2 = spl.derivative(2)(t)
        x1_dot = out_d1[:,0]
        x2_dot = out_d1[:,1]
        x1_ddot = out_d2[:,0]
        x2_ddot = out_d2[:,1]
        f_num = (np.multiply(x1_dot, x2_ddot) - np.multiply(x2_dot, x1_ddot))
        g_den = (np.square(x1_dot) + np.square(x2_dot))
        u = f_num / g_den

        v = np.sqrt(np.square(x1_dot) + np.square(x2_dot))
        return u,v
            
    
    def newParamsDifferent(self, estimatedRadarParamsList):
        for i in range(len(estimatedRadarParamsList)):
            for j in range(len(estimatedRadarParamsList[i])):
                if estimatedRadarParamsList[i][j]!=self.previousEmittorParamsList[i][j]:
                    return True
                
        return False
    
    def update_path(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos,numMeasurements):
        if numMeasurements != self.numMeasurements and len(estimatedRadarParamsList)>0:
            self.numMeasurements = numMeasurements
            if len(estimatedRadarParamsList)>0:
                if (len(self.previousEmittorParamsList)!=len(estimatedRadarParamsList) or self.newParamsDifferent(estimatedRadarParamsList)):
                    self.bestMeasurementLoc = self.optimize_next_best_measurement(currPos, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap)
                    if self.useSpline:
                        self.optimize_spline_path(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, self.bestMeasurementLoc)
                    self.numMeasurements = numMeasurements
                    # self.previousEmittorParamsList = estimatedRadarParamsList.copy()
                    self.currentSplineTime = 0
                # if not self.first:
                    # bestMeasurementLoc = self.optimize_next_best_measurement(currPos, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap)
                    # self.optimize_spline_path(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, bestMeasurementLoc)
                    # self.first = True
                    

        
    
    def create_evenly_spaced_control_points(self, start, stop, numControlPoints):
        xPoints = np.linspace(start[0] + .1, stop[0], numControlPoints, endpoint=False) 
        yPoints = np.linspace(start[1]+.1, stop[1], numControlPoints, endpoint=False) 
        points = np.hstack((xPoints.reshape((len(xPoints),1)), yPoints.reshape((len(xPoints),1))))
        return points
    
    
    def compute_spline_constraints(self,tf, spl):
        t = np.linspace(0,tf,params.numConstraintSamples)
        u,v = self.get_turn_rate_and_velocity(t, spl)

        pos = spl(t)

        return u,v,pos

    def spline_objective_function(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, controlPoints, knotPoints):
        spline = self.spline_seg(controlPoints, knotPoints)
        tObjective = np.linspace(0,knotPoints[-1], params.numObjectiveFunctionSamples)
        pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(spline(tObjective), estimatedRadarParamsList, estimatedRadarParamsCovList, False)
        u,v,pos = self.compute_spline_constraints(knotPoints[-1], spline)
        # return np.max(pdMean), u, v, pos
        return np.sum(pdMean), u, v, pos

    def create_knot_points(self, t0, tf, numControlPoints):
        #the number of control points
        l = numControlPoints

        #create evenly spaced knot points
        t = np.linspace(t0, tf, l - 2, endpoint=True)

        #add repeated knot points at begining and end
        t = np.append([t0, t0, t0], t)
        t = np.append(t, [tf, tf, tf])
        return t
    
    def create_control_points(self, optimizedControlPoints, currPose, currVelocity, splineOrder, bestMeasurementLocation, knotPoints):
        controlPoints = np.zeros((params.numControlPoints,2))
        controlPoints[0,:] = currPose[0:2]
        controlPoints[2:-1,:] = optimizedControlPoints.reshape((params.numControlPoints-3,2))

        controlPoints[1,0] = np.cos(currPose[2]) * self.currentVelocity * knotPoints[splineOrder + 1] / splineOrder + currPose[0]
        controlPoints[1,1] = np.sin(currPose[2]) * self.currentVelocity * knotPoints[splineOrder + 1] / splineOrder + currPose[1]


        controlPoints[-1,:] = bestMeasurementLocation 
        return controlPoints
        


    def optimize_spline_path(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, bestMeasurementLocation):
        straitLineDist = np.linalg.norm(currPos[0:2] - bestMeasurementLocation)
        initialControlPoints = self.create_evenly_spaced_control_points(currPos, bestMeasurementLocation, params.numControlPoints-3)
        def objective_function(xDict):
            tf = xDict['tf']
            knotPoints = self.create_knot_points(0, tf, params.numControlPoints)
            controlPoints = self.create_control_points(xDict['control_points'], currPos, self.currentVelocity, params.splineOrder, bestMeasurementLocation, knotPoints)
            funcs = {}
            obj,u,v,pos = self.spline_objective_function(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, controlPoints, knotPoints)
            funcs['obj'] = obj
            funcs['turn_rate'] = u 
            funcs['velocity'] = v 
            funcs['position'] = pos
            return funcs, False
            

        optProb = Optimization("low priority path", objective_function)
        optProb.addVarGroup(name = "control_points", nVars = 2*(params.numControlPoints-3), varType = 'c', value = initialControlPoints.reshape((2*(params.numControlPoints-3))), lower = 0, upper=params.bounds[1])
        optProb.addVarGroup(name = "tf", nVars = 1, varType = 'c', value = straitLineDist/params.agentSpeed, lower = 0, upper=params.pathLengthMultiplier * straitLineDist/params.agentSpeed)
        optProb.addConGroup("turn_rate", params.numConstraintSamples, lower=-params.maxTurnRate, upper=params.maxTurnRate, scale=1.0 / params.maxTurnRate)
        optProb.addConGroup("velocity", params.numConstraintSamples, lower=-params.velocityBounds[0], upper=params.velocityBounds[1], scale=1.0 / params.velocityBounds[1])
        optProb.addObj("obj")
        opt = OPT("ipopt")
        opt.options['print_level'] = 5
        opt.options['tol'] = 1e-10
        sol = opt(optProb, sens = 'FD')
        knotPoints = self.create_knot_points(0, sol.xStar['tf'], params.numControlPoints)
        controlPoints = self.create_control_points(sol.xStar['control_points'], currPos, self.currentVelocity, params.splineOrder, bestMeasurementLocation, knotPoints)
        # controlPoints = np.zeros((params.numControlPoints,2))
        # controlPoints[0,:] = currPos[0:2]
        # controlPoints[-1,:] = bestMeasurementLocation 
        # controlPoints[1:-1,:] = sol.xStar['control_points'].reshape((params.numControlPoints-2,2))
        # tf = sol.xStar['tf']
        self.splinePath = self.spline_seg(controlPoints, knotPoints)
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

    def objective_function(self, X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currentPosition):
        alpha = params.lowPrioritySafetyBestMeasurementTradeoff
        objectivFunctionVal = np.zeros((len(X_test),1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        for i, pos in enumerate(X_test):
            objectivFunctionVal[i] = self.objective_function_at_pos(pos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currentPosition) 
        return objectivFunctionVal

    def probability_of_detection_at_points(self, X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap):
        objectivFunctionVal = np.zeros((len(X_test),1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        for i, mean in enumerate(pdMean):
            var = pdVar[i]
            objectivFunctionVal[i] = mean 
        return objectivFunctionVal

    def objective_function_at_pos(self, pos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currentPosition):
        # alpha = params.lowPrioritySafetyBestMeasurementTradeoff
        alpha = params.lowPriorityDistanceBestMeasurementTradeoff
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar([pos], estimatedRadarParams, estimatedRadarParamsCov, False)
        # var = pdVar[i]
        # objectivFunctionVal = alpha * self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * pdMean 
        dist = np.linalg.norm(pos - currentPosition) / 20000
        objectivFunctionVal = alpha * self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * dist 
        return objectivFunctionVal


    def optimize_next_best_measurement(self, currPos, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap):
        def objective_function(xdict):
            pos = xdict['pos']
            # obj = -next_measurement_covariance_determinant(pos, estimatedParams_list, estimatedRadarCovariance_list)
            obj = -self.objective_function_at_pos(pos, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap, currPos[0:2])
            funcs = {}
            funcs['obj'] = obj
            fail = False
            return funcs, fail
        optProb = Optimization("find best measurement location", objective_function)
        optProb.addVarGroup(name = "pos", nVars = 2, varType = 'c', value = currPos[0:2], lower = 0, upper=params.bounds[1])
        optProb.addObj("obj")
        opt = OPT("ipopt")
        opt.options['print_level'] = 0
        opt.options['tol'] = 1e-8
        sol = opt(optProb, sens = 'FD')
        return sol.xStar["pos"]
        
        
        

            
    def plot_spline(self,ax,spl,num_points):
        t0 = spl.t[0]
        tf = spl.t[-1]
        t = np.linspace(t0, tf, num_points, endpoint=True)
        x = spl(t)[:,0]
        y = spl(t)[:,1]
        if self.splinePathPlot is not None:
            for line in self.splinePathPlot:
                line.remove()
            # for line in self.splineControlPointsPlot:
            #     line.remove()
        self.splinePathPlot = ax.plot(x,y, color = 'black')
        # plt.scatter(x,y,c = spline_color)
        control_points = spl.c
        # self.splineControlPointsPlot = ax.plot(control_points[:, 0], control_points[:, 1], 'k--', label='Control polygon', marker='o', zorder = 100000)


    def plot_objective_and_constraint(self, ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos, numMeasurements):
        # objectiveFunctionVal, constraintMet = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        objectiveFunctionVal = self.objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos[0:2])
        # objectiveFunctionVal = self.probability_of_detection_at_points(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)

        # objectiveFunctionVal[constraintMet ==0] = -1

        c = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), objectiveFunctionVal.reshape((params.numTestPoints,params.numTestPoints)))

        if self.splinePath is not None:
            self.plot_spline(ax, self.splinePath, 100)
        
        if self.bestMeasurementLoc is not None:
            if self.firstPlot:
                self.bestMeasurementLocPlot = Circle((self.bestMeasurementLoc[0],self.bestMeasurementLoc[1]),radius = 200,fill = True, color = 'r', zorder = 100000000)
                ax.add_patch(self.bestMeasurementLocPlot)
                self.firstPlot = False
            else:
                self.bestMeasurementLocPlot.center = self.bestMeasurementLoc[0], self.bestMeasurementLoc[1]
            
        # if numMeasurements != self.numMeasurements:
        #     bestPos = self.optimize_next_best_measurement(currPos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        #     print("bestPos", bestPos)
        #     if self.bestMesurementPlot is not None:
        #         self.bestMesurementPlot.remove()
        #     ax.scatter(bestPos[0], bestPos[1],zorder = 1000000)
    #     self.numMeasurements = numMeasurements
        return c

    def plot_chance_constraints(self, ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos, numMeasurements):
        objectiveFunctionVal, constraintMet = self.objective_function_chance_constraints(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        objectiveFunctionVal[constraintMet ==0] = 1
        objectiveFunctionVal[constraintMet ==1] = 0
        
        c = ax.contourf(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), objectiveFunctionVal.reshape((params.numTestPoints,params.numTestPoints)), levels = [-1,0,1], cmap = 'coolwarm')
        # if self.splinePath is not None:
        #     self.plot_spline(ax, self.splinePath, 100)
            
        # if numMeasurements != self.numMeasurements:
        #     bestPos = self.optimize_next_best_measurement(currPos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        #     print("bestPos", bestPos)
        #     if self.bestMesurementPlot is not None:
        #         self.bestMesurementPlot.remove()
        #     ax.scatter(bestPos[0], bestPos[1],zorder = 1000000)
        #     self.numMeasurements = numMeasurements
        return c
    
    
    
    
    