# import numpy as np
import numpy
from scipy.special import erfinv
import params
from scipy import interpolate
from scipy.stats.qmc import LatinHypercube,scale
from pyoptsparse import Optimization, OPT, IPOPT
from matplotlib.patches import Circle
import jax.numpy as np
# import numpy as np
from jax import grad, jacfwd
import jax
import time

from statistics import NormalDist
import matplotlib.pyplot as plt

# from casadi import *

@jax.jit
def find_closest_emitter(pos, estimatedRadarParams):
    closetEmittorIndex = 0
    for i in range(len(estimatedRadarParams)):
        if np.linalg.norm(estimatedRadarParams[i][0:2]-pos) < np.linalg.norm(estimatedRadarParams[closetEmittorIndex][0:2]-pos):
            closetEmittorIndex = i
    return closetEmittorIndex

@jax.jit
def next_measurement_covariance(pos, estimatedRadarParams, estimatedRadarCovariance):
    x = pos[0]
    y = pos[1]
    estimatedRadarParams = estimatedRadarParams
    estimatedRadarCovariance = estimatedRadarCovariance
    x_em = estimatedRadarParams[0]
    y_em = estimatedRadarParams[1]
    erp = estimatedRadarParams[2]
    H = params.measurement_jacobian_jax(x_em, y_em, erp, x, y)
    R = params.measurementCov
    K = estimatedRadarCovariance @ H.T @ np.linalg.inv(H@estimatedRadarCovariance@H.T + R)
    nextCovariance = (np.eye(3) - K@H)@estimatedRadarCovariance
    currentCovDet =np.linalg.det(estimatedRadarCovariance) 
    return nextCovariance

@jax.jit
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
        H = params.measurement_jacobian_jax(x_em, y_em, erp, x, y)
        R = params.measurementCov
        K = estimatedRadarCovariance @ H.T @ np.linalg.inv(H@estimatedRadarCovariance@H.T + R)
        nextCovariance = (np.eye(3) - K@H)@estimatedRadarCovariance
        currentCovDet =np.linalg.det(estimatedRadarCovariance) 
        currCovSum += currentCovDet
        out += (currentCovDet - np.linalg.det(nextCovariance))

    # nextCovariance = estimatedRadarCovariance - estimatedRadarCovariance@H.T@np.linalg.inv(H@estimatedRadarCovariance@H.T+R)@H@estimatedRadarCovariance
    # return 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(estimatedRadarCovariance)) - 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(nextCovariance))
    # return out/currCovSum
    return out

@jax.jit
def distance_from_line(x0,y0,x1,y1,x2,y2):
    return np.abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))/np.sqrt((x2-x1)**2+(y2-y1)**2)

@jax.jit
def kernel_seperation(allAgentPathHistory, next_measurement):
    return -np.sum(np.exp(-np.linalg.norm(allAgentPathHistory-next_measurement, axis=1)/params.lengthScale))


@jax.jit
def objective_function_at_pos_new(pos, estimatedRadarParams, estimatedRadarParamsCov, measuremetLocations):

    # # covScale = 3e15
    # covScale = 1
    # distFromStraitScale = np.sqrt(params.bounds[0]**2+bounds[1]**2)
    # coeffSeperation = .6
    # coeffCovariance = .4
    # coeffDistanceFromStrait = 1
    lengthScale = params.lengthScale
    x0 = pos[0]
    y0 = pos[1]
    x1 = params.highPriorityStart[0]
    y1 = params.highPriorityStart[1]
    x2 = params.highPriorityEnd[0]
    y2 = params.highPriorityEnd[1]


    distanceFromStraitLinePath = -distance_from_line(x0,y0,x1,y1,x2,y2)/params.distFromStraitScale

    # dist = np.linalg.norm(pos - currentPosition) / 20000
    # objectivFunctionVal = alpha * self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * dist 
    
    # minDistFromOtherMeasurements = np.average(np.linalg.norm(measuremetLocations-pos, axis=1))/self.minSeperationScale
    kernelSeperation = -np.sum(np.exp(-np.linalg.norm(measuremetLocations-pos, axis=1)/params.lengthScale))/params.seperationScale
    nexMeasCov = next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov)/params.nextCovarianceScale

    objectiveFunctionVal = params.nextCovarianceWeight * nexMeasCov + params.seperationWeight*kernelSeperation + params.distFromStraitWeight * distanceFromStraitLinePath
    
    return objectiveFunctionVal

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
        # self.bestMeasurementLoc = None
        self.bestMeasurementLocList = None
        self.kp = 1

        self.bestMeasurementLocPlotList = [] 
        self.firstPlot = True
        
        
        #desired heading list to unwrap heading
        self.numHeadingsToSave = 10
        self.savedHeadings = []
        self.savedDesiredHeadings = []

        self.minSeperationScale = np.sqrt(2)*params.bounds[0]
        self.covScale = 3e20
        self.distFromStraitScale = 5000

        self.bestMeasurementLocSeperation = 5000.0

        
        self.numOptStartLocations = 2
        
        xStart = np.linspace(params.bounds[0]/(self.numOptStartLocations+1),params.bounds[0]-params.bounds[0]/(self.numOptStartLocations+1), self.numOptStartLocations)
        yStart = np.linspace(params.bounds[1]/(self.numOptStartLocations+1),params.bounds[1]-params.bounds[1]/(self.numOptStartLocations+1), self.numOptStartLocations)
        self.optStartLocationsX, self.optStartLocationsY = np.meshgrid(xStart, yStart)


        self.figNum = 0
        self.timeSinceLastOpt = params.pathOptTime+1

        self.initialHeadings = np.linspace(-numpy.pi,numpy.pi,self.numOptStartLocations)

        lowerBound = -numpy.pi
        upperBound = numpy.pi

        self.numOptStartLocations = 10

        self.initialHeadingList = scale(LatinHypercube(params.numAgents).random(self.numOptStartLocations),lowerBound,upperBound)
        
        for initialHeading in self.initialHeadingList:
            print("initialHeading",initialHeading)



        
            


        
        

        
    

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
    
    def get_control(self, dt, currentPose, low_priority_agent_index):
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
            if self.bestMeasurementLocList is not None:
                u,v = self.get_turn_rate_and_velocity_waypoint(self.bestMeasurementLocList[low_priority_agent_index], currentPose)
                
        return u,v

    def get_turn_rate_and_velocity_waypoint(self, bestMeasurementLoc, currentPose):
        desiredHeading = np.arctan2(bestMeasurementLoc[1]-currentPose[1], bestMeasurementLoc[0]-currentPose[0])
        currentHeading = currentPose[2]
        if len(self.savedHeadings) > self.numHeadingsToSave:
            self.savedHeadings.pop(0)
            self.savedHeadings.append(currentHeading)
            self.savedDesiredHeadings.append(desiredHeading)
        else:
            self.savedDesiredHeadings.append(desiredHeading)
            self.savedHeadings.append(currentHeading)
        self.savedHeadings = np.unwrap(np.array(self.savedHeadings)).tolist()
        self.savedDesiredHeadings = np.unwrap(np.array(self.savedDesiredHeadings)).tolist()
        currentHeading = self.savedHeadings[-1]
        desiredHeading = self.savedDesiredHeadings[-1]
        v = (params.velocityBounds[0]+params.velocityBounds[1])/2
        # u = self.kp * (desiredHeading - currentPose[2])
        u = self.kp * (desiredHeading - currentHeading)
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
    
    def update_path(self, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, agentList,numMeasurements, measurementLocations,dt, allAgentesCurrentPos):
        # print("path planning")
        # print("numMeasurements", numMeasurements)
        # print("self.numMeasurements", self.numMeasurements)
        if len(estimatedRadarParamsList)>0:
            self.timeSinceLastOpt += dt
            if self.timeSinceLastOpt > params.pathOptTime:
                self.timeSinceLastOpt = 0
                self.bestMeasurementLocList = self.optimize_next_best_measurement_distance_constrained(agentList, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, measurementLocations, allAgentesCurrentPos)
                
        # if numMeasurements != self.numMeasurements and len(estimatedRadarParamsList)>0:
        #     self.numMeasurements = numMeasurements
        #     if len(estimatedRadarParamsList)>0:

        #         if (len(self.previousEmittorParamsList)!=len(estimatedRadarParamsList) or self.newParamsDifferent(estimatedRadarParamsList)):
        #             # self.bestMeasurementLocList = self.optimize_next_best_measurement(agentList, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, measurementLocations)
        #             self.bestMeasurementLocList = self.optimize_next_best_measurement_distance_constrained(agentList, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, measurementLocations)
        #             if self.useSpline:
        #                 self.optimize_spline_path(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, self.bestMeasurementLoc)
        #             self.numMeasurements = numMeasurements
        #             # self.previousEmittorParamsList = estimatedRadarParamsList.copy()
        #             self.currentSplineTime = 0
        #         # if not self.first:
        #             # bestMeasurementLoc = self.optimize_next_best_measurement(currPos, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap)
        #             # self.optimize_spline_path(estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, currPos, bestMeasurementLoc)
        #             # self.first = True
                    

        
    
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
        opt.options['print_level'] = 0
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
    


    def chance_constraint(self,rho, delta, mean, var):
        # return (mean - rho) > (-erfinv(2*delta-1)*np.sqrt(2*var))
        # return (mean - rho) > (erfinv(-2*delta+1)*np.sqrt(2*var))
        return (NormalDist(mu=mean, sigma=np.sqrt(var)).cdf(rho)) > delta
        # return (mean - rho) > (erfinv(-2*delta+1)*np.sqrt(2*var))

    def chance_constraints_at_points(self, X_test, probabilityOfDetectionMap):
        constraintMet = np.zeros((len(X_test),1))

        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        rho = params.probabilityOfDetectionThreshold
        delta = params.thresholdConfidence
        for i, mean in enumerate(pdMean):
            var = pdVar[i]
            constraintMet[i] = self.chance_constraint(rho, delta, mean, var)
        return constraintMet

    def objective_function(self, X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currentPosition, pathHistory):
        alpha = params.lowPrioritySafetyBestMeasurementTradeoff
        objectivFunctionVal = numpy.zeros((len(X_test),1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        for i, pos in enumerate(X_test):
            objectivFunctionVal[i] = objective_function_at_pos_new(pos, estimatedRadarParams, estimatedRadarParamsCov, pathHistory) 
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
    
    def agent_seperation_constraint(self, pos, agentBestMeausrementLocList):
        return np.linalg.norm(agentBestMeausrementLocList-pos) - self.bestMeasurementLocSeperation
    def agent_speeration_constraint_jacobian(self, pos, agentBestMeausrementLocList):
        return ((agentBestMeausrementLocList-pos)/numpy.linalg.norm(agentBestMeausrementLocList-pos)).reshape((1,2))
    
    
    # def get_control_gradient_based(self, agentList, estimatedParamsList, estimatedCovarianceList):
    #     allAgentPathHistory = []
    #     for agent in agentList:
    #         allAgentPathHistory += agent.pathHistory
    #     # allAgentPathHistory = np.concatenate([agent.pathHistory for agent in agentList])
    #     allAgentPathHistory = np.array(allAgentPathHistory)
    #     for k in range(len(agentList)):

    #             funcsSens['obj'] = {"pos" : -grad(objective_function_at_pos_new)(pos, estimatedParams_list, tempEstimatedParamsCovList, allAgentPathHistory)}


############# NEW TEST PATH PLANNER ####################
    def find_closest_emitter(self, pos, estimatedRadarParams):
        closetEmittorIndex = 0
        for i in range(len(estimatedRadarParams)):
            if np.linalg.norm(estimatedRadarParams[i][0:2]-pos) < np.linalg.norm(estimatedRadarParams[closetEmittorIndex][0:2]-pos):
                closetEmittorIndex = i
        return closetEmittorIndex

    def find_most_uncertain_emitter(self, estimatedRadarParamsCov):
        mostUncertainIndex = 0
        for i in range(len(estimatedRadarParamsCov)):
            if np.linalg.det(estimatedRadarParamsCov[i]) > np.linalg.det(estimatedRadarParamsCov[mostUncertainIndex]):
                mostUncertainIndex = i
        return mostUncertainIndex
    
    def find_agent_optimization_order(self, agentList, estimatedRadarParams,estimatedRadarParamsCov):

        agentOrder = []
        mostUncertainIndex = self.find_most_uncertain_emitter(estimatedRadarParamsCov)

        distToMostUncertain = numpy.zeros((len(agentList),))
        for i, agent in enumerate(agentList):
            distToMostUncertain[i] = numpy.linalg.norm(agent.position[0:2]-estimatedRadarParams[mostUncertainIndex][0:2])

        agentOrder = numpy.argsort(distToMostUncertain, axis=0)
        print("agentOrder",agentOrder)
        return agentOrder

        
    def get_agent_future_path(self, heading, agent, time):
        num_future_points = time//params.agentPathHistorydt
        first_point = np.array(agent.position[0:2])
        last_point = np.array(agent.position[0:2]) + time * params.agentSpeed * np.array([np.cos(heading), np.sin(heading)])
        points = np.linspace(first_point, last_point, num_future_points)
        # print("points",points)
        return points
            
    def objective_function_for_best_measurement_dist_constrained(self, headings, agentList, estimatedRadarParams, estimatedRadarParamsCov, allAgentPathHistory, velocity, pathTime,agentOrder):
        x1 = params.highPriorityStart[0]
        y1 = params.highPriorityStart[1]
        x2 = params.highPriorityEnd[0]
        y2 = params.highPriorityEnd[1]
        estimatedRadarParamsCov_temp = estimatedRadarParamsCov.copy()
        distanceFromStraitLinePathObj = 0
        kernelSeperationObj = 0
        allAgentPathHistory_temp = allAgentPathHistory.copy()

        for k in range(len(headings)):
            next_measurement = np.array(agentList[agentOrder[k]].position[0:2]) + velocity * pathTime * np.array([np.cos(headings[agentOrder[k]]), np.sin(headings[agentOrder[k]])])
            closest_emitter_index = self.find_closest_emitter(next_measurement, estimatedRadarParams)
            # closest_emitter_index = find_closest_emitter(next_measurement, estimatedRadarParams)
            # estimatedRadarParamsCov_temp[closest_emitter_index] = self.next_measurement_covariance(next_measurement, estimatedRadarParams, estimatedRadarParamsCov_temp, closest_emitter_index)
            estimatedRadarParamsCov_temp[closest_emitter_index] = next_measurement_covariance(next_measurement, estimatedRadarParams[closest_emitter_index], estimatedRadarParamsCov_temp[closest_emitter_index])

            x0 = next_measurement[0]
            y0 = next_measurement[1]
            distanceFromStraitLinePathObj += distance_from_line(x0,y0,x1,y1,x2,y2)/params.distFromStraitScale
            # kernelSeperationObj += -np.sum(np.exp(-np.linalg.norm(allAgentPathHistory-next_measurement, axis=1)/lengthScale))/params.seperationScale
            kernelSeperationObj += kernel_seperation(allAgentPathHistory_temp, next_measurement)/params.seperationScale
            futurePath = self.get_agent_future_path(headings[agentOrder[k]], agentList[agentOrder[k]], pathTime)
            allAgentPathHistory_temp = np.vstack((allAgentPathHistory_temp, futurePath))
            

            
        
        obj_cov = 0
        for i in range(len(estimatedRadarParams)):
            obj_cov += np.linalg.det(estimatedRadarParamsCov_temp[i])
            # fim = np.linalg.inv(estimatedRadarParamsCov_temp[i])
            # print("fim",fim)
            # obj_cov += np.linalg.det(fim)
        # obj_cov = 1/(obj_cov*params.nextCovarianceScale)
        # print(obj_cov._value)
        # return -params.nextCovarianceWeight * obj_cov/params.nextCovarianceScale + params.seperationWeight*kernelSeperationObj/params.seperationScale + params.distFromStraitWeight * distanceFromStraitLinePathObj/params.distFromStraitScale
        return params.nextCovarianceWeight * obj_cov/params.nextCovarianceScale - params.seperationWeight*kernelSeperationObj/params.seperationScale + params.distFromStraitWeight * distanceFromStraitLinePathObj/params.distFromStraitScale
    
    def waypoint_position_constraint(self,heading, allAgentCurrentPos, velocity, pathTime):
        # print("heading",heading)
        # print("velocity",velocity)
        # print("pathTime",pathTime)
        # print("allAgentCurrentPos",allAgentCurrentPos)
        return (np.hstack((np.cos(heading).reshape((len(heading),1)),np.sin(heading).reshape((len(heading),1))))*velocity*pathTime + allAgentCurrentPos).reshape((2*len(heading),))
    
    def waypoint_position_constraint_jac(self, heading, allAgentCurrentPos, velocity, pathTime):
        out = numpy.hstack((numpy.array([-numpy.sin(heading[0]),numpy.cos(heading[0])])*velocity*pathTime,numpy.zeros((2*(len(heading)-1),)))).reshape((2*len(heading),1))
        for i in range(1,heading.shape[0]):
            col = numpy.array([-numpy.sin(heading[i]),numpy.cos(heading[i])])*velocity*pathTime
            for k in range(i):
                col = numpy.hstack((numpy.zeros((2,)),col))
            for k in range(i,heading.shape[0]-1):
                col = numpy.hstack((col,numpy.zeros((2,))))
            col = col.reshape((2*len(heading),1))
                
            # out = np.hstack((out,np.hstack((np.array([-np.sin(heading[0]),np.cos(heading[0])])*velocity*pathTime,np.zeros((2*(len(heading)-1),)))).reshape((2*len(heading),1))))
            out = numpy.hstack((out,col))
            
        # return np.hstack((-np.sin(heading).reshape((len(heading),1)),np.cos(heading).reshape((len(heading),1))))*velocity*pathTime
        return out
    
    def find_initial_headings(self, agentList, estimatedParams_list):
        initialHeadings = numpy.linspace(0,2*np.pi,len(agentList))
        for i in range(len(agentList)):
            closesEmmiterIndex = self.find_closest_emitter(agentList[i].position[0:2], estimatedParams_list)
            initialHeadings[i] = np.arctan2(-agentList[i].position[1]+ estimatedParams_list[closesEmmiterIndex][1], -agentList[i].position[0]+ estimatedParams_list[closesEmmiterIndex][0])
        # print("initialHeadings",initialHeadings)
        # print("initial waypoint", np.array(agentList[0].position[0:2]) + params.pathOptTime *params.agentSpeed * np.array([np.cos(initialHeadings[0]), np.sin(initialHeadings[0])]))
        # print("initial waypoint", np.array(agentList[1].position[0:2]) + params.pathOptTime *params.agentSpeed * np.array([np.cos(initialHeadings[1]), np.sin(initialHeadings[1])]))
        # print("constraint", self.waypoint_position_constraint(initialHeadings, np.array([agentList[0].position[0:2], agentList[1].position[0:2]]), params.agentSpeed, params.pathOptTime))
        # print()
        return initialHeadings
        
    def optimize_next_best_measurement_distance_constrained(self, agentList, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap, measurementLocations, allAgentesCurrentPos):
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        # allAgentPathHistory = np.concatenate([agent.pathHistory for agent in agentList])
        allAgentPathHistory = np.array(allAgentPathHistory)
        
        # agentOrder = self.find_agent_optimization_order(agentList, estimatedParams_list, estimatedRadarCovariance_list)
        # print("agentOrder",agentOrder)
        agentOrder = range(len(agentList))
        
        initialHeadings = self.find_initial_headings(agentList, estimatedParams_list)
        initialHeadingsList = []
        
        
        tempObjectiveFuncScale = np.abs(self.objective_function_for_best_measurement_dist_constrained(initialHeadings, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory, params.agentSpeed, params.pathOptTime, agentOrder))

        # print("TEST constraint", self.waypoint_position_constraint(initialHeadings, allAgentesCurrentPos, params.agentSpeed, params.pathOptTime))
        # print("TEST constraint jac", jacfwd(self.waypoint_position_constraint)(initialHeadings, allAgentesCurrentPos, params.agentSpeed, params.pathOptTime))
        # print("test my jac", self.waypoint_position_constraint_jac(initialHeadings, allAgentesCurrentPos, params.agentSpeed, params.pathOptTime))
        # print("TEST")


        def objective_function(xdict):
            headings = xdict['headings']
            funcs = {}
            funcs['obj'] = self.objective_function_for_best_measurement_dist_constrained(headings, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory, params.agentSpeed, params.pathOptTime, agentOrder)/tempObjectiveFuncScale
            pos_con = self.waypoint_position_constraint(headings, allAgentesCurrentPos, params.agentSpeed, params.pathOptTime)
            funcs['pos_con'] = pos_con
            # print("funcs", funcs)
            return funcs, False
        def sens(xDict, funcs):
            # print("in sens")
            # print("xDict",xDict)
            # print("funcs",funcs)
            headings = xDict['headings']
            funcsSens = {}
            funcsSens['obj'] = {"headings" : grad(self.objective_function_for_best_measurement_dist_constrained)(headings, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory, params.agentSpeed, params.pathOptTime,agentOrder)/tempObjectiveFuncScale}
            # funcsSens['pos_con'] = {"headings" : jacfwd(lambda x: self.waypoint_position_constraint(x,allAgentesCurrentPos, params.agentSpeed, params.pathOptTime))(headings).reshape((len(agentList)*2, len(agentList)))}
            funcsSens['pos_con'] = {"headings" : self.waypoint_position_constraint_jac(headings, allAgentesCurrentPos, params.agentSpeed, params.pathOptTime)}
            # print("funcsSens['obj']",funcsSens['obj'])
            # print("funcsSens['pos_con']",funcsSens['pos_con'])
            return funcsSens, False


        optHeadings = None
        bestOptVal = np.inf
        totalTime = 0
        numOptimization = 0
        for itialHeadings in self.initialHeadingList:
            start = time.time()
            constraints = self.waypoint_position_constraint(itialHeadings, allAgentesCurrentPos, params.agentSpeed, params.pathOptTime)
            if np.any(constraints < 0):
                print("initial heading out of bounds")
                continue
            elif np.any(constraints > params.bounds[1]):
                print("initial heading out of bounds")
                continue
            else:
                headings, fStar = self.run_optimization(objective_function, sens, agentList, itialHeadings)
                numOptimization += 1
                totalTime += time.time()-start
                if fStar is not None:
                    if fStar < bestOptVal:
                        bestOptVal = fStar
                        optHeadings = headings 
        print("totalTime",totalTime)
        print("averageTime",totalTime/numOptimization)
        print("number of optimizations",numOptimization)

        
        
        
        
        
        fig,ax = plt.subplots()
        headingPlot1 = numpy.linspace(-1*numpy.pi,1*numpy.pi,50)
        headingPlot2 = numpy.linspace(-1*numpy.pi,1*numpy.pi,50)
        [X,Y] = numpy.meshgrid(headingPlot1,headingPlot2)
        Z = numpy.zeros_like(X)
        for i in range(len(headingPlot1)):
            for j in range(len(headingPlot2)):
                Z[j,i] = -self.objective_function_for_best_measurement_dist_constrained(numpy.array([X[j,i],Y[j,i]]), agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory, params.agentSpeed, params.pathOptTime,agentOrder)/tempObjectiveFuncScale
        ax.pcolormesh(X,Y,Z)
        # objPlot = numpy.zeros_like(headingPlot)
        # for k in range(len(headingPlot)):
        #     objPlot[k] = self.objective_function_for_best_measurement_dist_constrained(np.array([headingPlot[k]]), agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory, params.agentSpeed, params.pathOptTime)/tempObjectiveFuncScale
        # ax.plot(headingPlot, objPlot)
        # ax.scatter(optHeadings, sol.fStar)
        
        # image = np.array(fig.canvas.renderer.buffer_rgba())
        # plt.imsave("images/headingPlot/"+str(self.figNum)+".png", image)
        # plt.show()
        ax.scatter(optHeadings[0],optHeadings[1])
        fig.savefig("images/headingPlot/"+str(self.figNum)+".png")
        self.figNum += 1
        
        bestMeasurementLocList = []
        for k in range(len(agentList)):
            bestMeasurementLocList.append(np.array(agentList[k].position[0:2]) + params.pathOptTime * params.agentSpeed * np.array([np.cos(optHeadings[k]), np.sin(optHeadings[k])]))
        return bestMeasurementLocList

    
    def run_optimization(self,objective_function, sens, agentList, initialHeadings):
        optProb = Optimization("find best measurement location", objective_function)
        optProb.addVarGroup(name = "headings", nVars = len(agentList), varType = 'c', value = initialHeadings, lower = -2*np.pi, upper=2*np.pi)
        optProb.addConGroup("pos_con", 2*len(agentList), lower = 0, upper=params.bounds[1])
        optProb.addObj("obj")
        opt = OPT("ipopt")
        # opt.options['hsllib'] = '/home/grant/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        opt.options['hsllib'] = '/home/ggs24/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        opt.options['linear_solver'] = 'ma97'
        opt.options['print_level'] = 0
            
        # opt.options['derivative_test'] = 'first-order'
        # opt.options['derivative_test_perturbation'] = 1e-5
        opt.options['max_iter'] = 200
        opt.options['tol'] = 1e-8
        sol = opt(optProb, sens = sens)
        # sol = opt(optProb, sens = 'fd')
        print(sol)
        
        optHeadings = sol.xStar['headings']

        if sol.optInform['value'] == 0:
            print("OPTIMIZATION SUCCESSFUL")
            return optHeadings, sol.fStar
        else:
            print("OPTIMIZATION FAILED")
            return None,None 
############# NEW TEST PATH PLANNER ####################





    def optimize_next_best_measurement(self, agentList, estimatedParams_list, estimatedRadarCovariance_list, probabilityOfDetectionMap, measurementLocations):
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        # allAgentPathHistory = np.concatenate([agent.pathHistory for agent in agentList])
        allAgentPathHistory = np.array(allAgentPathHistory)


        bestMeasurementLocList = []
        tempEstimatedParamsCovList = estimatedRadarCovariance_list.copy()
        for k in range(len(agentList)):
                
            
            def objective_function(xdict):
                pos = xdict['pos']
                # obj = -next_measurement_covariance_determinant(pos, estimatedParams_list, estimatedRadarCovariance_list)
                # obj = -objective_function_at_pos_new(pos, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory)
                obj = -objective_function_at_pos_new(pos, estimatedParams_list, tempEstimatedParamsCovList, allAgentPathHistory)
                funcs = {}
                funcs['obj'] = obj
                if k > 0:
                    funcs['sep'] = self.agent_seperation_constraint(np.array(pos), np.array(bestMeasurementLocList)).squeeze()
                fail = False
                return funcs, fail
            minObjectiveFunctionVal = np.inf
            minObjFuncValLoc = None
            def sens(xDict, funcs):
                pos = xDict['pos']
                funcsSens = {}
                # funcsSens['obj'] = {"pos" : -grad(objective_function_at_pos_new)(pos, estimatedParams_list, estimatedRadarCovariance_list, np.array(measurementLocations))}
                # funcsSens['obj'] = {"pos" : -grad(objective_function_at_pos_new)(pos, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory)}
                funcsSens['obj'] = {"pos" : -grad(objective_function_at_pos_new)(pos, estimatedParams_list, tempEstimatedParamsCovList, allAgentPathHistory)}

                if k > 0:
                    # funcsSens['seperation'] = {"pos" : grad(self.agent_seperation_constraint)( np.array(pos), np.array(bestMeasurementLocList))}
                    funcsSens['sep'] = {"pos" : self.agent_speeration_constraint_jacobian(numpy.array(pos), numpy.array(bestMeasurementLocList))}
                    # funcsSens['seperation'] = {"pos" : [0.0,0.0]}
                return funcsSens, False
        
            
            start = time.time()
            for i in range(self.numOptStartLocations):
                for j in range(self.numOptStartLocations):
                    x_start = self.optStartLocationsX[i][j]
                    y_start = self.optStartLocationsY[i][j]
                    if k > 0:
                        print()

                    optProb = Optimization("find best measurement location"+str(k), objective_function)
                    optProb.addVarGroup(name = "pos", nVars = 2, varType = 'c', value = [x_start,y_start], lower = 0, upper=params.bounds[1])
                    optProb.addObj("obj")
                    if k > 0:
                        optProb.addConGroup("sep", k, lower = 0, upper=10000000)
                    opt = OPT("ipopt")
                    opt.options['hsllib'] = '/home/grant/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
                    # opt.options['hsllib'] = '/home/ggs24/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
                    opt.options['linear_solver'] = 'ma97'
                    opt.options['print_level'] = 0
                        
                    # opt.options['derivative_test'] = 'first-order'
                    opt.options['max_iter'] = 2000
                    opt.options['tol'] = 1e-3
                    sol = opt(optProb, sens = sens)
                    if sol.optInform['value'] == 0:
                        print("OPTIMIZATION SUCCESSFUL")
                    else:
                        print("OPTIMIZATION FAILED")
                        print(sol.optInform)
                        print(sol)
                    
                    print("VAL:", sol.fStar)
                    if sol.fStar < minObjectiveFunctionVal and (sol.optInform['value'] == 0 or sol.optInform['value'] == 1):
                        print("TEST")
                        minObjectiveFunctionVal = sol.fStar
                        minObjFuncValLoc =sol.xStar['pos']
            
            if minObjFuncValLoc is None:
                print("NO OPTIMAL SOLUTION FOUND")
                minObjFuncValLoc = self.bestMeasurementLocList[k]
            
            bestMeasurementLocList.append(minObjFuncValLoc)
            futurePath = self.get_future_path(agentList[k].position, minObjFuncValLoc, params.agentSpeed)
            tempEstimatedParamsCovList = self.get_future_covariance(estimatedParams_list, tempEstimatedParamsCovList, minObjFuncValLoc)

            allAgentPathHistory = np.vstack((allAgentPathHistory, futurePath))
            print("BEST VAL:", minObjectiveFunctionVal)
            print("BEST Loc:", bestMeasurementLocList[-1])
            print("time for optimization", time.time()-start)
                    

        # return minObjFuncValLoc
        return bestMeasurementLocList 
    
    def get_future_path(self, currentPose, bestMeasurementLoc, speed):
        dist = np.linalg.norm(currentPose[0:2] - bestMeasurementLoc)
        time = dist/speed
        numPoints = int(time/params.agentPathHistorydt)
        x = np.linspace(currentPose[0], bestMeasurementLoc[0], numPoints)
        y = np.linspace(currentPose[1], bestMeasurementLoc[1], numPoints)
        return np.hstack((x.reshape((len(x),1)), y.reshape((len(y),1))))

    def get_future_covariance(self, estimatedRadarParamsList, estimatedRadarParamsCovList, bestMeasurementLoc):
        closetEmittorIndex = 0
        for i in range(len(estimatedRadarParamsList)):
            if np.linalg.norm(estimatedRadarParamsList[i][0:2]-bestMeasurementLoc) < np.linalg.norm(estimatedRadarParamsList[closetEmittorIndex][0:2]-bestMeasurementLoc):
                closetEmittorIndex = i

        x_em = estimatedRadarParamsList[closetEmittorIndex][0]
        y_em = estimatedRadarParamsList[closetEmittorIndex][1]
        erp = estimatedRadarParamsList[closetEmittorIndex][2]
        x = bestMeasurementLoc[0]
        y = bestMeasurementLoc[1]
        
        H = params.measurement_jacobian_jax(x_em, y_em, erp, x, y)
        R = params.measurementCov
        K = estimatedRadarParamsCovList[closetEmittorIndex] @ H.T @ np.linalg.inv(H@estimatedRadarParamsCovList[closetEmittorIndex]@H.T + R)
        nextCovariance = (np.eye(3) - K@H)@estimatedRadarParamsCovList[closetEmittorIndex]

        estimatedRadarParamsCovList[closetEmittorIndex] = nextCovariance
        return estimatedRadarParamsCovList
        
        
        

            
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


    def plot_objective_and_constraint(self, ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos, numMeasurements, agentList,agentPlotIndex):
        
        
        # objectiveFunctionVal, constraintMet = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        allAgentPathHistory = np.array(allAgentPathHistory)

        tempEstimatedParamsCovList = estimatedRadarParamsCov.copy()
        for i in range(agentPlotIndex):
            # futurePath = self.get_future_path(agentList[i].position, self.bestMeasurementLocList[i], params.agentSpeed)
            tempEstimatedParamsCovList = self.get_future_covariance(estimatedRadarParams, tempEstimatedParamsCovList, self.bestMeasurementLocList[i])
            # allAgentPathHistory = np.vstack((allAgentPathHistory, futurePath))
            # allAgentPathHistory = np.vstack((allAgentPathHistory, futurePath))
        

        objectiveFunctionVal = self.objective_function(X_test, estimatedRadarParams, tempEstimatedParamsCovList, probabilityOfDetectionMap, currPos[0:2], allAgentPathHistory)

        
        
        # objectiveFunctionVal = np.zeros((len(X_test)))

        # for i, pos in enumerate(X_test):
        #     heading = [np.arctan2(pos[1]-currPos[1], pos[0]-currPos[0])]
        #     objectiveFunctionVal[i] = self.objective_function_for_best_measurement_dist_constrained(heading, agentList, estimatedRadarParams, tempEstimatedParamsCovList, allAgentPathHistory, params.agentSpeed, params.pathOptTime)

        # objectiveFunctionVal[constraintMet ==0] = -1

        c = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), objectiveFunctionVal.reshape((params.numTestPoints,params.numTestPoints)))

        if self.splinePath is not None:
            self.plot_spline(ax, self.splinePath, 100)
        
        # if self.bestMeasurementLoc is not None:
        if self.bestMeasurementLocList is not None:
            for i in range(params.numAgents):
                if self.firstPlot:
                    self.bestMeasurementLocPlotList.append(Circle((self.bestMeasurementLocList[i][0],self.bestMeasurementLocList[i][1]),radius = 200,fill = True, color = 'r', zorder = 100000000))
                    ax.add_patch(self.bestMeasurementLocPlotList[i])
                else:
                    self.bestMeasurementLocPlotList[i].center = self.bestMeasurementLocList[i][0], self.bestMeasurementLocList[i][1]
            self.firstPlot = False
            
        # if numMeasurements != self.numMeasurements:
        #     bestPos = self.optimize_next_best_measurement(currPos, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        #     print("bestPos", bestPos)
        #     if self.bestMesurementPlot is not None:
        #         self.bestMesurementPlot.remove()
        #     ax.scatter(bestPos[0], bestPos[1],zorder = 1000000)
    #     self.numMeasurements = numMeasurements
        return c

    def plot_chance_constraints(self, ax,X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap, currPos, numMeasurements):
        constraintMet = self.chance_constraints_at_points(X_test, probabilityOfDetectionMap)
        groundTruthConstraint = probabilityOfDetectionMap.groundTruthpdMap

        c1 = ax.contour(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), groundTruthConstraint.reshape((params.numTestPoints,params.numTestPoints)), levels = [params.probabilityOfDetectionThreshold])

        c2 = ax.contour(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), probabilityOfDetectionMap.pdMap.reshape((params.numTestPoints,params.numTestPoints)), levels = [params.probabilityOfDetectionThreshold], cmap = 'hsv')

        # c4 = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), probabilityOfDetectionMap.pdCovMap.reshape((params.numTestPoints,params.numTestPoints)))
        
        c3 = ax.contour(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), constraintMet.reshape((params.numTestPoints,params.numTestPoints)), levels = [-1,0,1], cmap = 'coolwarm')
        # c3 = ax.contourf(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), constraintMet.reshape((params.numTestPoints,params.numTestPoints)), levels = [-1,0,1], cmap = 'coolwarm')
        return [c1,c2,c3]
    
    
    
    
    
if __name__ == "__main__":
    pathPlanner = SplinePathPlanningLowPriority()
