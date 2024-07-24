import numpy as np
import jax.numpy as jnp
from jax import jacfwd,grad
from jax import jit
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu') # Sets the default device to CPU
from scipy.constants import k as boltzman
import matplotlib
import getpass


import matplotlib.pyplot as plt
import params

from pyoptsparse import Optimization, OPT

from scipy import interpolate

from main_helper import create_radar_list

from probabilityOfDetectionMap import ProbabilityOfDetectionMap


from PythonRobotics.PathPlanning.RRTStar import rrt_star

from scipy.interpolate import make_lsq_spline,make_smoothing_spline, splrep, splev, splprep
from scipy.interpolate import BSpline

from scipy.spatial import Voronoi, voronoi_plot_2d

import igraph as ig
import time

from bspline.matrix_evaluation import matrix_bspline_evaluation,derivative_matrix_bspline_evaluation,matrix_bspline_derivative_evaluation_for_dataset, matrix_bspline_evaluation_for_dataset

# from bspline.matrix_evaluation import matrix_bspline_evaluation_for_dataset_jax, matrix_bspline_evaluation_jax

from highPriorityHelperFunctions import create_unclamped_knot_points, get_spline_velocity, get_pd_along_spline, get_spline_turn_rate, dist_of_points_to_line_segment,get_prob_pd_less_than_threshold,get_prob_along_spline


from weightedVoronoiPathIntialzation import compute_path_weighted_voronoi

import uncertainVoronoiPathIntialization





class HighPriorityPathPlannerDeterministic:
    def __init__(self,radarList):
        #run jax function once to compile
        start = time.time()
        self.evaluate_spline_derivative(0,np.zeros((params.numControlPoints,2)),np.zeros(params.numControlPoints+params.splineOrder+1),params.splineOrder,1)
        self.evaluate_spline(0,np.zeros((params.numControlPoints,2)),np.zeros(params.numControlPoints+params.splineOrder+1),params.splineOrder)
        create_unclamped_knot_points(0, 1, params.numControlPoints,params.splineOrder)
        get_spline_velocity(jnp.zeros((2*params.numControlPoints,)), 1, params.splineOrder, params.numSamplesPerInterval)
        
        self.dVelocityDControlPoints = jacfwd(get_spline_velocity)
        self.dVelocityDtf = jacfwd(get_spline_velocity,argnums=1)

        self.dPdDControlPoints = jacfwd(get_pd_along_spline,argnums=0)
        self.dPdDtf = jacfwd(get_pd_along_spline,argnums=1)

        self.dProbDControlPoints = jacfwd(get_prob_along_spline,argnums=0)
        self.dProbDtf = jacfwd(get_prob_along_spline,argnums=1)

        self.dTurnRateDControlPoints = jacfwd(get_spline_turn_rate)
        self.dTurnRateTf = jacfwd(get_spline_turn_rate,argnums=1)

        self.deterministicRadarPositions = np.array([radar.position for radar in radarList])

        dist_of_points_to_line_segment(np.array([0,0]),np.array([1,1]),self.deterministicRadarPositions)
        print("Time to compile jax functions", time.time()-start)

        self.useWeightedVoronoi = True
        self.uncertainRadar =True 

        
    
    
    def get_turn_rate_and_velocity(self, t, controlPoints, knotPoints):
        out_d1 = self.evaluate_spline_derivative(t,controlPoints,knotPoints,params.splineOrder,1)
        out_d2 = self.evaluate_spline_derivative(t,controlPoints,knotPoints,params.splineOrder,2)
        v = np.linalg.norm(out_d1,axis=1)
        u = np.cross(out_d1,out_d2) / (v**2)
        return u,v
            

    def create_evenly_spaced_control_points(self, start, stop, numControlPoints):
        xPoints = np.linspace(start[0] + .1, stop[0], numControlPoints, endpoint=False) 
        yPoints = np.linspace(start[1]+.1, stop[1], numControlPoints, endpoint=False) 
        points = np.hstack((xPoints.reshape((len(xPoints),1)), yPoints.reshape((len(xPoints),1))))
        return points

    def get_start_constraint(self, controlPoints):
        cp1 = controlPoints[0:2]
        cp2 = controlPoints[2:4]
        cp3 = controlPoints[4:6]
        return np.array((1/6)*cp1 + (2/3)*cp2 + (1/6)*cp3)

    def get_start_constraint_jacobian(self, controlPoints):
        jac = np.zeros((2,2*params.numControlPoints))
        jac[0,0] = 1/6
        jac[0,2] = 2/3
        jac[0,4] = 1/6
        jac[1,1] = 1/6
        jac[1,3] = 2/3
        jac[1,5] = 1/6
        return jac
        
    def get_end_constraint(self, controlPoints):
        cpnMinus2 = controlPoints[-6:-4]
        cpnMinus1 = controlPoints[-4:-2]
        cpn = controlPoints[-2:]
        return (1/6)*cpnMinus2 + (2/3)*cpnMinus1 + (1/6)*cpn
    
    def get_end_constraint_jacobian(self, controlPoints):
        jac = np.zeros((2,2*params.numControlPoints))
        jac[0,-6] = 1/6
        jac[0,-4] = 2/3
        jac[0,-2] = 1/6
        jac[1,-5] = 1/6
        jac[1,-3] = 2/3
        jac[1,-1] = 1/6
        return jac
        
        

    def spline_constraints(self, radarList, controlPoints, knotPoints,numConstraintSamples):
        # spline = self.spline_seg(controlPoints, knotPoints)
        tf = knotPoints[-params.splineOrder-1]
        t = np.linspace(0,tf,numConstraintSamples)
        u,v = self.get_turn_rate_and_velocity(t, controlPoints, knotPoints)

        pos = self.evaluate_spline(t,controlPoints,knotPoints,params.splineOrder)

        # pd = self.ground_truth_probability_of_detection(pos, radarList)

        pd = get_pd_along_spline(controlPoints, tf, radarList, params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
        # return np.max(pdMean), u, v, pos
        return pd, u, v, pos
    
    def spline_constraints_uncertain_radar(self, radarParams, radarParamsCov, controlPoints, knotPoints,numConstraintSamples):
        # spline = self.spline_seg(controlPoints, knotPoints)
        tf = knotPoints[-params.splineOrder-1]
        t = np.linspace(0,tf,numConstraintSamples)
        u,v = self.get_turn_rate_and_velocity(t, controlPoints, knotPoints)

        pos = self.evaluate_spline(t,controlPoints,knotPoints,params.splineOrder)

        # pd = self.ground_truth_probability_of_detection(pos, radarList)
# def get_prob_pd_less_than_threshold(pos, estimatedRadarParamsList, estimatedRadarParamsCovList, pdThreshold, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar):
        prob = get_prob_pd_less_than_threshold(pos, radarParams, radarParamsCov,params.probabilityOfDetectionThreshold, params.radarRecieveGain, 0, params.radarWavelength,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean,params.radarProbabilityOfFalseAlarmPriorVariance)
        # prob = get_prob_pd_less_than_threshold(pos,radarParams, radarParamsCov, params.probabilityOfDetectionThreshold, params.radarRecieveGain, params.radarRecieveGain, params.radarWavelength, params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean, params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)

        # pd = get_pd_along_spline(controlPoints, tf, radarParams, radarParamsCov, params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean)
        # return np.max(pdMean), u, v, pos
        return prob, u, v, pos



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
        spline = interpolate.BSpline(t.squeeze(), control_points, 3)

        return spline
    
    def plan_deterministic_path(self, radar_list,plot=False,ax=None):
        startTimer = time.time()
        # initialControlPoints, tfIntial,ax = self.find_initial_guess_rrt_star(radarList,plot=plot)
        initialControlPoints, tfIntial = self.get_initial_guess_voronoi(radarList,params.bounds,plot=plot,ax=ax)
        print("Time to find initial guess", time.time()-startTimer)
        spline = self.spline_seg(initialControlPoints,create_unclamped_knot_points(0, tfIntial, params.numControlPoints,params.splineOrder))


        def objective_function(xDict):
            tf = xDict['tf']
            knotPoints = create_unclamped_knot_points(0, tf, params.numControlPoints,params.splineOrder)
            controlPoints = xDict['control_points']
            funcs = {}
            funcs['start'] = self.get_start_constraint(controlPoints)
            funcs['end'] = self.get_end_constraint(controlPoints)
            controlPoints = controlPoints.reshape((params.numControlPoints,2))
            # v = self.get_spline_velocity(controlPoints, tf)
            pd, u, v, pos = self.spline_constraints(radar_list, controlPoints, knotPoints,params.numConstraintSamples)
            # funcs['start'] = self.get_start_constraint_jax(controlPoints)
            # funcs['start'] = pos[0]
            # funcs['end'] = pos[-1]
            funcs['obj'] = tf 
            funcs['turn_rate'] = u 
            funcs['velocity'] = v 
            # funcs['position'] = pos
            funcs['pd'] = pd
            return funcs, False
    
        def sens(xDict, funcs):
            funcsSens = {}
            controlPoints = jnp.array(xDict['control_points'])
            tf = xDict['tf']

            dStartDControlPoints = self.get_start_constraint_jacobian(controlPoints)
            dEndDControlPoints = self.get_end_constraint_jacobian(controlPoints)
            # dVelocityDControlPoints = jacfwd(get_spline_velocity)(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval)
            dVelocityDControlPoints = self.dVelocityDControlPoints(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval)
            dVelocityDtf = np.array(self.dVelocityDtf(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval),dtype=np.float64)
            # dTurnRateDControlPoints = jacfwd(self.get_spline_turn_rate)(controlPoints, tf)
            # dTurnRateDtf = np.array(jacfwd(self.get_spline_turn_rate,argnums=1)(controlPoints, tf),dtype=np.float64)
            dPdDControlPoints = self.dPdDControlPoints(controlPoints, tf, radar_list,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean )
            dPdDtf = np.array(self.dPdDtf(controlPoints, tf, radar_list,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean),dtype=np.float64)

            dTurnRateDControlPoints = self.dTurnRateDControlPoints(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval)
            dTurnRateDtf = np.array(self.dTurnRateTf(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval),dtype=np.float64)

            funcsSens['obj'] = {"control_points": np.zeros((1,2*params.numControlPoints)), "tf": 1}
            funcsSens['start'] = {"control_points": dStartDControlPoints, "tf": np.zeros((2,1))}
            funcsSens['end'] = {"control_points": dEndDControlPoints, "tf": np.zeros((2,1))}
            # funcsSens['turn_rate'] = {"control_points": np.zeros((params.numConstraintSamples,2*params.numControlPoints)), "tf": np.zeros((params.numConstraintSamples,1))}
            # funcsSens['velocity'] = {"control_points": np.zeros((params.numConstraintSamples,2*params.numControlPoints)), "tf": np.zeros((params.numConstraintSamples,1))}
            funcsSens['velocity'] = {"control_points": dVelocityDControlPoints, "tf": dVelocityDtf}
            funcsSens['turn_rate'] = {"control_points": dTurnRateDControlPoints, "tf": dTurnRateDtf}
            funcsSens['pd'] = {"control_points": dPdDControlPoints, "tf": dPdDtf}
            return funcsSens, False
            
        
        straitLineDist = np.linalg.norm(np.array(params.highPriorityEnd) - np.array(params.highPriorityStart))
            

        optProb = Optimization("low priority path", objective_function)
        optProb.addVarGroup(name = "control_points", nVars = 2*(params.numControlPoints), varType = 'c', value = initialControlPoints.reshape((2*(params.numControlPoints))), lower = 0-2000, upper=params.bounds[1]+2000)
        optProb.addVarGroup(name = "tf", nVars = 1, varType = 'c', value = tfIntial, lower = 0, upper=params.pathLengthMultiplier * straitLineDist/params.agentSpeed)
        optProb.addConGroup("turn_rate", params.numConstraintSamples, lower=-params.maxTurnRate, upper=params.maxTurnRate, scale=1.0 / params.maxTurnRate)
        optProb.addConGroup("velocity", params.numConstraintSamples, lower=-params.velocityBounds[0], upper=params.velocityBounds[1], scale=1.0 / params.velocityBounds[1])
        optProb.addConGroup("pd", params.numConstraintSamples, lower=0, upper=params.probabilityOfDetectionThreshold, scale=1.0)
        optProb.addConGroup("start", 2, lower = params.highPriorityStart, upper = params.highPriorityStart)
        optProb.addConGroup("end", 2, lower = params.highPriorityEnd, upper = params.highPriorityEnd)
        optProb.addObj("obj")
        opt = OPT("ipopt")
        # opt.options['derivative_test'] = 'first-order'
        username = getpass.getuser()

        opt.options['hsllib'] = '/home/' + username + '/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        opt.options['linear_solver'] = 'ma97'
        opt.options['print_level'] = 0
        opt.options['max_iter'] = 1000
        opt.options['tol'] = 1e-8
        sol = opt(optProb, sens = sens)
        # sol = opt(optProb, sens = 'FD')
        print(sol)
        knotPoints = create_unclamped_knot_points(0, sol.xStar['tf'], params.numControlPoints,params.splineOrder)
        # controlPoints = self.create_control_points(sol.xStar['control_points'], params.highPriorityStart, params.highPriorityEnd, params.splineOrder, knotPoints)
        controlPoints = sol.xStar['control_points'].reshape((params.numControlPoints,2))
        self.spline = self.spline_seg(controlPoints, knotPoints)
        return ax

    def plan_uncertain_path(self,radarList, estimateRadarParams,estimatedRadarParamsCov,plot=False,ax=None):
        startTimer = time.time()
        # initialControlPoints, tfIntial,ax = self.find_initial_guess_rrt_star(radarList,plot=plot)
        initialControlPoints, tfIntial = self.get_initial_guess_voronoi(radarList,params.bounds,estimateRadarParams, estimatedRadarParamsCov,plot=plot,ax=ax)
        print("Time to find initial guess", time.time()-startTimer)
        spline = self.spline_seg(initialControlPoints,create_unclamped_knot_points(0, tfIntial, params.numControlPoints,params.splineOrder))


        def objective_function(xDict):
            tf = xDict['tf']
            knotPoints = create_unclamped_knot_points(0, tf, params.numControlPoints,params.splineOrder)
            controlPoints = xDict['control_points']
            funcs = {}
            funcs['start'] = self.get_start_constraint(controlPoints)
            funcs['end'] = self.get_end_constraint(controlPoints)
            controlPoints = controlPoints.reshape((params.numControlPoints,2))
            # v = self.get_spline_velocity(controlPoints, tf)
            pd, u, v, pos = self.spline_constraints_uncertain_radar(estimateRadarParams,estimatedRadarParamsCov, controlPoints, knotPoints,params.numConstraintSamples)
            # funcs['start'] = self.get_start_constraint_jax(controlPoints)
            # funcs['start'] = pos[0]
            # funcs['end'] = pos[-1]
            funcs['obj'] = tf 
            funcs['turn_rate'] = u 
            funcs['velocity'] = v 
            # funcs['position'] = pos
            funcs['pd'] = pd
            return funcs, False
    
        def sens(xDict, funcs):
            funcsSens = {}
            controlPoints = jnp.array(xDict['control_points'])
            tf = xDict['tf']

            dStartDControlPoints = self.get_start_constraint_jacobian(controlPoints)
            dEndDControlPoints = self.get_end_constraint_jacobian(controlPoints)
            # dVelocityDControlPoints = jacfwd(get_spline_velocity)(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval)
            dVelocityDControlPoints = self.dVelocityDControlPoints(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval)
            dVelocityDtf = np.array(self.dVelocityDtf(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval),dtype=np.float64)
            # dTurnRateDControlPoints = jacfwd(self.get_spline_turn_rate)(controlPoints, tf)
            # dTurnRateDtf = np.array(jacfwd(self.get_spline_turn_rate,argnums=1)(controlPoints, tf),dtype=np.float64)
            # dPdDControlPoints = self.dProbDControlPoints(controlPoints, tf, estimateRadarParams,estimatedRadarParamsCov,params.probabilityOfDetectionThreshold,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean )
# def get_prob_along_spline(controlpoints, tf, estimatedRadarParams, estimatedRadarParamsCov,pdThreshold, numcontrolpoints, splineorder, numsamplesperinterval,radarReceiveGain,radarReceiveGainVar, radarwavelengthpriormean,radarwavelengthvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthvar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar):
            dPdDControlPoints = self.dProbDControlPoints(controlPoints, tf, estimateRadarParams,estimatedRadarParamsCov,params.probabilityOfDetectionThreshold,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarRecieveGain,0,params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance)
            dPdDtf = np.array(self.dProbDtf(controlPoints, tf, estimateRadarParams,estimatedRadarParamsCov,params.probabilityOfDetectionThreshold,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarRecieveGain,0,params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance))
            # dPdDControlPoints = np.array(self.dProbDtf(controlPoints, tf, estimateRadarParams,estimatedRadarParamsCov,params.probabilityOfDetectionThreshold,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth, params.radarPulseWidthPriorVariance, params.radarSystemTemperaturePriorMean,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarmPriorMean, params.radarProbabilityOfFalseAlarmPriorVariance))

            # dPdDtf = np.array(self.dProbDtf(controlPoints, tf, estimateRadarParams,estimatedRadarParamsCov,params.probabilityOfDetectionThreshold,params.numControlPoints, params.splineOrder, params.numSamplesPerInterval, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, params.radarSystemTemperaturePriorMean, params.radarProbabilityOfFalseAlarmPriorMean),dtype=np.float64)

            dTurnRateDControlPoints = self.dTurnRateDControlPoints(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval)
            dTurnRateDtf = np.array(self.dTurnRateTf(controlPoints, tf, params.splineOrder,params.numSamplesPerInterval),dtype=np.float64)

            funcsSens['obj'] = {"control_points": np.zeros((1,2*params.numControlPoints)), "tf": 1}
            funcsSens['start'] = {"control_points": dStartDControlPoints, "tf": np.zeros((2,1))}
            funcsSens['end'] = {"control_points": dEndDControlPoints, "tf": np.zeros((2,1))}
            # funcsSens['turn_rate'] = {"control_points": np.zeros((params.numConstraintSamples,2*params.numControlPoints)), "tf": np.zeros((params.numConstraintSamples,1))}
            # funcsSens['velocity'] = {"control_points": np.zeros((params.numConstraintSamples,2*params.numControlPoints)), "tf": np.zeros((params.numConstraintSamples,1))}
            funcsSens['velocity'] = {"control_points": dVelocityDControlPoints, "tf": dVelocityDtf}
            funcsSens['turn_rate'] = {"control_points": dTurnRateDControlPoints, "tf": dTurnRateDtf}
            funcsSens['pd'] = {"control_points": dPdDControlPoints, "tf": dPdDtf}
            return funcsSens, False
            
        
        straitLineDist = np.linalg.norm(np.array(params.highPriorityEnd) - np.array(params.highPriorityStart))
            

        optProb = Optimization("low priority path", objective_function)
        optProb.addVarGroup(name = "control_points", nVars = 2*(params.numControlPoints), varType = 'c', value = initialControlPoints.reshape((2*(params.numControlPoints))), lower = 0-2000, upper=params.bounds[1]+2000)
        optProb.addVarGroup(name = "tf", nVars = 1, varType = 'c', value = tfIntial, lower = 0, upper=params.pathLengthMultiplier * straitLineDist/params.agentSpeed)
        optProb.addConGroup("turn_rate", params.numConstraintSamples, lower=-params.maxTurnRate, upper=params.maxTurnRate, scale=1.0 / params.maxTurnRate)
        optProb.addConGroup("velocity", params.numConstraintSamples, lower=-params.velocityBounds[0], upper=params.velocityBounds[1], scale=1.0 / params.velocityBounds[1])
        optProb.addConGroup("pd", params.numConstraintSamples, lower=params.thresholdConfidence, upper=1, scale=1.0)
        optProb.addConGroup("start", 2, lower = params.highPriorityStart, upper = params.highPriorityStart)
        optProb.addConGroup("end", 2, lower = params.highPriorityEnd, upper = params.highPriorityEnd)
        optProb.addObj("obj")
        opt = OPT("ipopt")
        # opt.options['derivative_test'] = 'first-order'
        username = getpass.getuser()

        opt.options['hsllib'] = '/home/' + username + '/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        opt.options['linear_solver'] = 'ma97'
        opt.options['print_level'] = 5
        opt.options['max_iter'] = 200
        opt.options['tol'] = 1e-5
        sol = opt(optProb, sens = sens)
        # sol = opt(optProb, sens = 'FD')
        print(sol)
        knotPoints = create_unclamped_knot_points(0, sol.xStar['tf'], params.numControlPoints,params.splineOrder)
        # controlPoints = self.create_control_points(sol.xStar['control_points'], params.highPriorityStart, params.highPriorityEnd, params.splineOrder, knotPoints)
        controlPoints = sol.xStar['control_points'].reshape((params.numControlPoints,2))
        self.spline = self.spline_seg(controlPoints, knotPoints)
        return ax

    def find_radius_from_radar_pd(self,radar, pd):
        erp = params.radarOutputPower*params.radarTransmitGain
        G_r = params.radarRecieveGain
        lamb = params.radarWavelength
        sigma = params.agentRadarCrossSection
        tua = params.radarPulseWidth
        Pfa = params.radarProbabilityOfFalseAlarm
        Ts = params.radarSystemTemperature
        k = boltzman
        R = (((erp*G_r*lamb**2*sigma*tua)/((np.log(Pfa)/np.log(pd))-1))*(1/((4*np.pi)**3*k*Ts)))**.25
        return R
    
    def find_obsticle_list_from_radars(self, radarList):

        obsticleList = []
        for radar in radarList:
            radius = self.find_radius_from_radar_pd(radar, params.probabilityOfDetectionThreshold)
            obsticleList.append((radar.position[0], radar.position[1], radius))
        return obsticleList

    def fit_spline_to_path(self, path, num_control_points):
        tf = 1
        t = np.linspace(0,tf, len(path))
        
        # num_control_points = params.numControlPoints
        n_interior_knots = num_control_points - params.splineOrder - 1
        qs = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
        knots = np.quantile(t, qs)

        tck_x = splrep(t,path[:,0],k=params.splineOrder,t=knots,s=1)
        control_points_x = tck_x[1]
        control_points_x = control_points_x[control_points_x != 0]
        control_points_x[0] = params.highPriorityStart[0]
        control_points_x[-1] = params.highPriorityEnd[0]

        tck_y = splrep(t,path[:,1],k=params.splineOrder,t=knots,s=1)
        control_points_y = tck_y[1]
        control_points_y = control_points_y[control_points_y != 0]
        control_points_y[0] = params.highPriorityStart[1]
        control_points_y[-1] = params.highPriorityEnd[1]
        combined_control_points = np.hstack((control_points_x.reshape((len(control_points_x),1)), control_points_y.reshape((len(control_points_y),1))))

        combined_knot_points = tck_x[0]
        return combined_control_points,combined_knot_points
    
    def assure_pd_less_than_threshold(self, radarList, controlPoints, knotPoints):
        pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,params.numConstraintSamples)
        num_control_points = len(controlPoints)
        while np.max(pd) > params.probabilityOfDetectionThreshold:
            num_control_points += 1
            combined_control_points, combined_knot_points = self.fit_spline_to_path(pos,num_control_points)
            pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)

        return combined_control_points,combined_knot_points
    
    def assure_velocity_constraint(self, radarList, controlPoints, knotPoints,num_control_points):
        # pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,params.numConstraintSamples)
        tf=np.linalg.norm(controlPoints[0]-controlPoints[-1])/params.agentSpeed
        v = get_spline_velocity(controlPoints, tf,params.splineOrder,params.numSamplesPerInterval)
        while np.max(v) > params.velocityBounds[1]:
            tf += 10
            # combined_knot_points = self.create_unclamped_knot_points(0, tf, num_control_points,params.splineOrder)
            v = get_spline_velocity(controlPoints, tf,params.splineOrder,params.numSamplesPerInterval)
            # pd, u, v, pos = self.spline_constraints(radarList, controlPoints, combined_knot_points,params.numConstraintSamples)
        combined_knot_points = create_unclamped_knot_points(0, tf, num_control_points,params.splineOrder)
        return combined_knot_points,tf
        
    
    def find_initial_guess_rrt_star(self,radarList, plot=False):
        obsticle_list = self.find_obsticle_list_from_radars(radarList)
            # Set Initial parameters
        rrt = rrt_star.RRTStar(
            start=[params.highPriorityStart[0], params.highPriorityStart[1]],
            goal=[params.highPriorityEnd[0], params.highPriorityEnd[1]],
            rand_area=[0, params.bounds[0]],
            obstacle_list=obsticle_list,
            expand_dis=500,
            # robot_radius=0.8,
            robot_radius=100,
            max_iter=10000)
        path = rrt.planning(animation=plot)
        path = np.flip(np.array(path),axis=0)

        num_control_points = params.numControlPoints
        combined_control_points, combined_knot_points = self.fit_spline_to_path(path,num_control_points)

        combined_knot_points,tf = self.assure_velocity_constraint(radarList, combined_control_points, combined_knot_points,num_control_points)
        
        

        spline = self.spline_seg(combined_control_points,combined_knot_points) 
        


        ax = None
        if path is None:
            print("Cannot find path")
        else:
            print("found path!!")

            if plot:
                rrt.draw_graph()
                t = np.linspace(0,tf,1000)
                plt.plot([x for (x, y) in path], [y for (x, y) in path], 'b--')
                plt.plot(spline(t)[:,0], spline(t)[:,1], 'r')
                # plt.plot(control_points_x, control_points_y, 'k--',marker='o',alpha=.5)
                plt.plot(combined_control_points[:,0], combined_control_points[:,1], 'k--',marker='o',alpha=.5)
                plt.grid(True)
                ax = plt.gca()
                # self.plot_constraints(spline, radarList)
        return combined_control_points, tf,ax
    
    
    def find_closest_segements_to_point(self, point, segments):
        distances = []
        whichPoint = []
        for seg in segments:
            seg = np.array(seg)
            #only check segments on boundary
            if np.any(seg == 0) or np.any(seg == params.bounds):
                # dist = self.dist_of_point_to_line_segment(seg[0][0], seg[0][1], seg[1][0], seg[1][1], point[0], point[1])
                dist = np.min(np.linalg.norm(np.array(seg) - np.array(point),axis=1))
                whichPoint.append(np.argmin(np.linalg.norm(np.array(seg) - np.array(point),axis=1)))
                distances.append(dist)
            else:
                distances.append(np.inf)
                whichPoint.append(None)
        minIndex = np.argmin(distances)
        # distances[minIndex] = np.inf
        # secondMinIndex = np.argmin(distances)
        return [point, segments[minIndex][whichPoint[minIndex]]]#, [point, segments[secondMinIndex][whichPoint[secondMinIndex]]]
        
    

    def intersection_of_two_line_segments(self, p0, p1, p2, p3 ):
        #https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/1201356#1201356

        s10_x = p1[0] - p0[0]
        s10_y = p1[1] - p0[1]
        s32_x = p3[0] - p2[0]
        s32_y = p3[1] - p2[1]

        denom = s10_x * s32_y - s32_x * s10_y

        if denom == 0 : return None # collinear

        denom_is_positive = denom > 0

        s02_x = p0[0] - p2[0]
        s02_y = p0[1] - p2[1]

        s_numer = s10_x * s02_y - s10_y * s02_x

        if (s_numer < 0) == denom_is_positive : return None # no collision

        t_numer = s32_x * s02_y - s32_y * s02_x

        if (t_numer < 0) == denom_is_positive : return None # no collision

        if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : return None # no collision


        # collision detected

        t = t_numer / denom

        intersection_point = [ p0[0] + (t * s10_x), p0[1] + (t * s10_y) ]


        return intersection_point

    def find_intersection_with_boundary(self, A, B, bounds):
        intersectionList = []
        C = np.array([0,0])
        D = np.array([0,bounds[1]])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            intersectionList.append(intersection)
            # return intersection
        C = np.array([0,0])
        D = np.array([bounds[0],0])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            intersectionList.append(intersection)
            # return intersection
        C = np.array([0,bounds[1]])
        D = np.array([bounds[0],bounds[1]])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            intersectionList.append(intersection)
            # return intersection
        C = np.array([bounds[0],0])
        D = np.array([bounds[0],bounds[1]])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            intersectionList.append(intersection)
            # return intersection
        return intersectionList
        # return None
    
    # def find_line_segments_from_point_to_closest
    def add_boundary_segments(self,segments,bounds):
        tolerance = 1e-5
        segments_to_add = []
        if np.any(np.isclose(segments[:,:,0],0,atol=tolerance)):
            y_coords = np.sort(segments[np.isclose(segments[:,:,0],0,atol=tolerance)][:,1])
            # temp_segments = []
            for i in range(len(y_coords)+1):
                if i == 0:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,0]),np.array([0,y_coords[i]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[0,0],[0,y_coords[i]]])
                elif i == len(y_coords):
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,y_coords[i-1]]),np.array([0,bounds[1]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[0,y_coords[i-1]],[0,bounds[1]]])
                else:
                    distToClosestRadar1 = np.min(dist_of_points_to_line_segment(np.array([0,y_coords[i-1]]),np.array([0,y_coords[i]]),self.deterministicRadarPositions))
                    if distToClosestRadar1 > params.safeRadius:
                        segments_to_add.append([[0,y_coords[i-1]],[0,y_coords[i]]])    
        else:
            distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,0]),np.array([0,bounds[1]]),self.deterministicRadarPositions))
            if distToClosestRadar > params.safeRadius:
                segments_to_add.append([[0,0],[0,bounds[1]]])
        if np.any(np.isclose(segments[:,:,0], bounds[0],atol=tolerance)):
            y_coords = np.sort(segments[np.isclose(segments[:,:,0], bounds[0],atol=tolerance)][:,1])
            for i in range(len(y_coords)+1):
                if i == 0:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([bounds[0],0]),np.array([bounds[0],y_coords[i]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[bounds[0],0],[bounds[0],y_coords[i]]])
                elif i == len(y_coords):
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([bounds[0],y_coords[i-1]]),np.array([bounds[0],bounds[1]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[bounds[0],y_coords[i-1]],[bounds[0],bounds[1]]])
                else:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([bounds[0],y_coords[i-1]]),np.array([bounds[0],y_coords[i]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[bounds[0],y_coords[i-1]],[bounds[0],y_coords[i]]])    
        else:
            distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([bounds[0],0]),np.array([bounds[0],bounds[1]]),self.deterministicRadarPositions))
            if distToClosestRadar > params.safeRadius:
                segments_to_add.append([[bounds[0],0],[bounds[0],bounds[1]]])
        if np.any(np.isclose(segments[:,:,1],0,atol=1e-5)):
            x_coords = np.sort(segments[np.isclose(segments[:,:,1],0,atol=1e-5)][:,0])
            for i in range(len(x_coords)+1):
                if i == 0:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,0]),np.array([x_coords[i],0]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[0,0],[x_coords[i],0]])
                elif i == len(x_coords):
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([x_coords[i-1],0]),np.array([bounds[0],0]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[x_coords[i-1],0],[bounds[0],0]])
                else:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([x_coords[i-1],0]),np.array([x_coords[i],0]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[x_coords[i-1],0],[x_coords[i],0]])
        else:
            distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,0]),np.array([bounds[0],0]),self.deterministicRadarPositions))
            if distToClosestRadar > params.safeRadius:
                segments_to_add.append([[0,0],[bounds[0],0]])
        if np.any(np.isclose(segments[:,:,1], bounds[1],atol=tolerance)):
            x_coords = np.sort(segments[np.isclose(segments[:,:,1], bounds[1],atol=tolerance)][:,0])
            for i in range(len(x_coords)+1):
                if i == 0:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,bounds[1]]),np.array([x_coords[i],bounds[1]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[0,bounds[1]],[x_coords[i],bounds[1]]])
                elif i == len(x_coords):
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([x_coords[i-1],bounds[1]]),np.array([bounds[0],bounds[1]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[x_coords[i-1],bounds[1]],[bounds[0],bounds[1]]])
                else:
                    distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([x_coords[i-1],bounds[1]]),np.array([x_coords[i],bounds[1]]),self.deterministicRadarPositions))
                    if distToClosestRadar > params.safeRadius:
                        segments_to_add.append([[x_coords[i-1],bounds[1]],[x_coords[i],bounds[1]]])
        else:
            distToClosestRadar = np.min(dist_of_points_to_line_segment(np.array([0,bounds[1]]),np.array([bounds[0],bounds[1]]),self.deterministicRadarPositions))
            if distToClosestRadar > params.safeRadius:
                segments_to_add.append([[0,bounds[1]],[bounds[0],bounds[1]]])
                    
        segments = np.vstack((segments,segments_to_add))
        return segments
        
        
    # def remove_segments_too_close_to_point(self, point, segments, minDist):

    def remove_unfeasible_segments(self,segments,radarList):
        minRadarDists = np.array([np.min(dist_of_points_to_line_segment(seg[0],seg[1],self.deterministicRadarPositions)) for seg in segments])
        
        return segments[minRadarDists > params.safeRadius]
        
    
    def transform_to_unweighted_segment(self,segment,weights):
        p1 = segment[0]
        p2 = segment[1]

        unweighted_p1 = np.sum(p1 * weights[:, np.newaxis], axis=0) / np.sum(weights)
        unweighted_p2 = np.sum(p2 * weights[:, np.newaxis], axis=0) / np.sum(weights)
        print(unweighted_p1)
        return [unweighted_p1,unweighted_p2]


    def get_weighted_voronoi_ridge_segements(self,radarList,weights, bounds,plot=False):
        startTimer = time.time()
        weights = np.array(weights)
        weights = weights/np.max(weights)
        points = np.array([[radar.position[0], radar.position[1]]/weights[i] for i,radar in enumerate(radarList)])
        vor = Voronoi(points)
        print("time for scipy voronoi", time.time()-startTimer)


        startTimer = time.time()
        segments = []
        ptp_bound = vor.points.ptp(axis=0)
        center = vor.points.mean(axis=0)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                segment = self.transform_to_unweighted_segment(vor.vertices[simplex],weights)
                intersections = self.find_intersection_with_boundary(segment[0], segment[1], bounds)
                if len(intersections) >0:
                    if len(intersections) == 1:
                        if np.any(segment[0] <0) or np.any(segment[1] > bounds[0]):
                            segments.append([intersections[0], segment[1]]) 
                        elif np.any(segment[1] <0) or np.any(segment[1] > bounds[0]):
                            segments.append([intersections[0], segment[0]])
                        else:
                            segments.append([segment[0], intersections[0]])
                    else:
                        segments.append(intersections)
                else:
                    segments.append(segment)
                # segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if (vor.furthest_site):
                    direction = -direction
                aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
                far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor
                
                intersections = self.find_intersection_with_boundary(vor.vertices[i], far_point, bounds)

                if len(intersections)>0:
                    segments.append([vor.vertices[i], intersections[0]])

        print("time to find segments", time.time()-startTimer)

        segments = np.array(segments)
        startTimer = time.time()
        segments = self.remove_unfeasible_segments(segments,radarList)
        print("time to remove unfeasible segments", time.time()-startTimer)
        startTimer = time.time()
        segments = self.add_boundary_segments(segments,bounds)
        print("time to add boundary segments", time.time()-startTimer)
        
        
        
        ax = None
        if plot:
            fig = plt.figure()
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.scatter(params.highPriorityStart[0], params.highPriorityStart[1], c='r',zorder=10000)
            ax.scatter(params.highPriorityEnd[0], params.highPriorityEnd[1], c='r',zorder=10000)
            # ax.set_xlim([-1000,params.bounds[0]+1000])
            # ax.set_ylim([-1000,params.bounds[1]+1000])
            for seg in segments:
                plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 'g--')
            plt.show()

        return segments,ax

    def get_voronoi_ridge_segements(self,radarList,bounds,plot=False):
        startTimer = time.time()
        points = np.array([[radar.position[0], radar.position[1]] for radar in radarList])
        vor = Voronoi(points)
        print("time for scipy voronoi", time.time()-startTimer)


        startTimer = time.time()
        segments = []
        ptp_bound = vor.points.ptp(axis=0)
        center = vor.points.mean(axis=0)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                intersections = self.find_intersection_with_boundary(vor.vertices[simplex[0]], vor.vertices[simplex[1]], bounds)
                if len(intersections) >0:
                    if len(intersections) == 1:
                        if np.any(vor.vertices[simplex[0]] <0) or np.any(vor.vertices[simplex[0]] > bounds[0]):
                            segments.append([intersections[0], vor.vertices[simplex[1]]]) 
                        elif np.any(vor.vertices[simplex[1]] <0) or np.any(vor.vertices[simplex[1]] > bounds[0]):
                            segments.append([intersections[0], vor.vertices[simplex[0]]])
                        else:
                            segments.append([vor.vertices[simplex[0]], intersections[0]])
                    else:
                        segments.append(intersections)
                else:
                    segments.append(vor.vertices[simplex])
                # segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if (vor.furthest_site):
                    direction = -direction
                aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
                far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor
                
                intersections = self.find_intersection_with_boundary(vor.vertices[i], far_point, bounds)

                if len(intersections)>0:
                    segments.append([vor.vertices[i], intersections[0]])

        print("time to find segments", time.time()-startTimer)

        segments = np.array(segments)
        startTimer = time.time()
        segments = self.remove_unfeasible_segments(segments,radarList)
        print("time to remove unfeasible segments", time.time()-startTimer)
        startTimer = time.time()
        segments = self.add_boundary_segments(segments,bounds)
        print("time to add boundary segments", time.time()-startTimer)
        
        
        
        ax = None
        if plot:
            fig = plt.figure()
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.scatter(params.highPriorityStart[0], params.highPriorityStart[1], c='r',zorder=10000)
            ax.scatter(params.highPriorityEnd[0], params.highPriorityEnd[1], c='r',zorder=10000)
            ax.set_xlim([-1000,params.bounds[0]+1000])
            ax.set_ylim([-1000,params.bounds[1]+1000])
            for seg in segments:
                plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 'g--')

        return segments,ax
    
    def get_weighted_adjacency_matrix(self,segments):
        nodes = dict()
        currentNodeNumber = 0
        nodes[currentNodeNumber] = [0,0]
        goalNode = -1
        for seg in segments:
            for point in seg:
                nodeExists = False
                for key in nodes.keys():
                    if np.all(np.isclose(nodes[key], point)):
                        nodeExists = True
                        break
                if not nodeExists:
                    currentNodeNumber += 1
                    nodes[currentNodeNumber] = point
                    if np.all(np.isclose(point, params.highPriorityEnd,atol=1e-5)):
                        goalNode = currentNodeNumber
        adjacencyMatrix = np.zeros((len(nodes),len(nodes)))
        for seg in segments:
            node1 = None
            node2 = None
            for key in nodes.keys():
                if np.all(np.isclose(nodes[key], seg[0],atol=1e-5)):
                    node1 = key
                if np.all(np.isclose(nodes[key], seg[1],atol=1e-5)):
                    node2 = key
            adjacencyMatrix[node1,node2] = np.linalg.norm(seg[0]-seg[1])
            adjacencyMatrix[node2,node1] = np.linalg.norm(seg[0]-seg[1])
        
        return adjacencyMatrix,nodes,goalNode
    
    def fill_in_path(self,path,spacing = 500):

        new_path = []
        for i in range(len(path)-1):
            num_points = int(np.linalg.norm(path[i]-path[i+1])/spacing)
            points = np.linspace(path[i], path[i+1], num_points)
            for point in points:
                new_path.append(point)
        new_path.append(path[-1])
        return np.array(new_path)
    
    def move_first_control_point_so_spline_passes_through_start(self, controlPoints,knotPoints, start,startVelocity):
        dt = knotPoints[3]-knotPoints[0]
        A = np.array([[1/6,0,2/3,0],[0,1/6,0,2/3],[-3/(2*dt),0,0,0],[0,-3/(2*dt),0,0]])
        c3x = controlPoints[2,0]
        c3y = controlPoints[2,1]

        b = np.array([[start[0] - (1/6)*c3x],[start[1]- (1/6)*c3y],[startVelocity[0]-3/(2*dt)*c3x],[startVelocity[1]-3/(2*dt)*c3y]])

        x = np.linalg.solve(A,b)
        controlPoints[0:2,0:2] = x.reshape((2,2))

        return controlPoints

    def move_last_control_point_so_spline_passes_through_end(self, controlPoints,knotPoints, end,endVelocity):
        dt = knotPoints[3]-knotPoints[0]
        A = np.array([[2/3,0,1/6,0],[0,2/3,0,1/6],[0,0,3/(2*dt),0],[0,0,0,3/(2*dt)]])
        cn_minus_2_x = controlPoints[-3,0]
        cn_minus_2_y = controlPoints[-3,1]

        b = np.array([[end[0] - (1/6)*cn_minus_2_x],[end[1]- (1/6)*cn_minus_2_y],[endVelocity[0]+3/(2*dt)*cn_minus_2_x],[endVelocity[1]+3/(2*dt)*cn_minus_2_y]])

        x = np.linalg.solve(A,b)
        controlPoints[-2:,-2:] = x.reshape((2,2))

        return controlPoints

    def find_derivative_control_points(self, controlPoints, knotPoints,splineOrder):
        dt = knotPoints[splineOrder] - knotPoints[0]

        previousControlPoints = controlPoints

        newControlPoints = np.array([splineOrder*(previousControlPoints[i+1] - previousControlPoints[i]) / dt for i in range(len(previousControlPoints)-1)])

        return newControlPoints,knotPoints[1:-1],splineOrder-1

    def evaluate_spline(self,evalPoints, controlPoints, knotPoints,splineOrder):
        # knotPoints = np.vstack((knotPoints,knotPoints))
        # evalPoints = np.vstack((evalPoints,evalPoints))
        # out = coef2curve(evalPoints, knotPoints, controlPoints.T, splineOrder)
        # # out = matrix_bspline_evaluation(evalPoints, 1, controlPoints.T, knotPoints)
        # scaleFactor = knotPoints[-splineOrder-1]/(len(knotPoints)-2*splineOrder-1)
        # return np.array([matrix_bspline_evaluation(point, scaleFactor, controlPoints.T, knotPoints) for point in evalPoints]).squeeze()
        return matrix_bspline_evaluation_for_dataset(controlPoints.T, knotPoints, params.numSamplesPerInterval)
        # return self.spline_seg(controlPoints, knotPoints)(evalPoints)

    def evaluate_spline_at_time_t(self,t, controlPoints, knotPoints,splineOrder):
        # knotPoints = np.vstack((knotPoints,knotPoints))
        # evalPoints = np.vstack((evalPoints,evalPoints))
        # out = coef2curve(evalPoints, knotPoints, controlPoints.T, splineOrder)
        # # out = matrix_bspline_evaluation(evalPoints, 1, controlPoints.T, knotPoints)
        scaleFactor = knotPoints[-splineOrder-1]/(len(knotPoints)-2*splineOrder-1)
        return np.array([matrix_bspline_evaluation(point, scaleFactor, controlPoints.T, knotPoints) for point in t]).squeeze()
        # return matrix_bspline_evaluation_for_dataset(controlPoints.T, knotPoints, params.numSamplesPerInterval)
        # return self.spline_seg(controlPoints, knotPoints)(evalPoints)

    
    def evaluate_spline_derivative(self, t, controlPoints, knotPoints,splineOrder, derivativeOrder):
        # f = lambda x: self.evaluate_spline(x,controlPoints, knotPoints, splineOrder)
        # for i in range(derivativeOrder):
        #     f = jacfwd(f)
        # A = f(jnp.array(t))
        # # A = jacfwd(self.evaluate_spline)(jnp.array(t),controlPoints, knotPoints, splineOrder)
        # return A.transpose(0,2,1).reshape(len(t)**2,2)[::len(t)+1]
        # # return np.array([A[i,:,i] for i in range(len(A))])
        # # return A[A!=0].reshape(len(t),2)
        # for i in range(derivativeOrder):
        #     controlPoints, knotPoints, splineOrder = self.find_derivative_control_points(controlPoints, knotPoints,splineOrder)
        
        # return self.evaluate_spline(t, controlPoints, knotPoints, splineOrder)

        scaleFactor = knotPoints[-splineOrder-1]/(len(knotPoints)-2*splineOrder-1)
        # return np.array([derivative_matrix_bspline_evaluation(time, derivativeOrder, scaleFactor, controlPoints.T, knotPoints, clamped = False) for time in t]).squeeze()
        # return self.spline_seg(controlPoints, knotPoints).derivative(derivativeOrder)(t)
        # return matrix_bspline_derivative_evaluation_for_dataset(controlPoints.T, knotPoints, 3, derivativeOrder)
        return matrix_bspline_derivative_evaluation_for_dataset(derivativeOrder, scaleFactor, controlPoints.T, knotPoints, params.numSamplesPerInterval)
        
        

    def get_initial_guess_voronoi(self,radarList,bounds,radarParams = None, radarParamsCov = None, plot=False,ax=None):


#################### this code is for unwieghted voronoi
        if not self.useWeightedVoronoi:
            startTime = time.time()
            segments,ax = self.get_voronoi_ridge_segements(radarList,bounds,plot=plot)
            # weights = np.array([radar.outputPower*radar.transmitGain*radar.recieveGain for radar in radarList])
            # segments,ax = self.get_weighted_voronoi_ridge_segements(radarList,weights,bounds,plot=plot)
            print("Time to get voronoi segments", time.time()-startTime)

            startTime = time.time()
            adjMatrix,nodes, goalNode = self.get_weighted_adjacency_matrix(segments)
            print("Time to get adjacency matrix", time.time()-startTime)


            
            startTime = time.time()
            g = ig.Graph.Weighted_Adjacency(adjMatrix, mode="undirected")
            print("Time to create graph", time.time()-startTime)
            startTime = time.time()
            path = g.get_shortest_paths(0,to=goalNode,weights=g.es["weight"])
            print("Time to find shortest path", time.time()-startTime)

            startTime= time.time()
            path = np.array([nodes[node] for node in path[0]])
            path = self.fill_in_path(path)
######################
        else:
            if not self.uncertainRadar:
                print(type(radarList))
                path = compute_path_weighted_voronoi(radarList,plot,ax)
            else:
                # dataIndex = 639
                # # dataIndex = 200
                # radarParams = np.load("saved_data/currentData/estimated_params/"+str(dataIndex)+".npy")
                # radarParamsCov = np.load("saved_data/currentData/estimated_params_cov/"+str(dataIndex)+".npy")
                startTime = time.time()
                path = uncertainVoronoiPathIntialization.find_initial_trajectory_uncertain_radar(radarParams,radarParamsCov,spacing=50,ax=ax)
                print("Time to find initial trajectory", time.time()-startTime)
        
        
        
        

        startTime = time.time()
        controlPoints, knotPoints = self.fit_spline_to_path(path,params.numControlPoints)
        print("Time to fit spline to path", time.time()-startTime)
        
        startTime = time.time()
        knotPoints = create_unclamped_knot_points(0,1, params.numControlPoints,params.splineOrder)
        knotPoints,tf = self.assure_velocity_constraint(radarList, controlPoints.reshape((-1,)), knotPoints,params.numControlPoints)
        print("tf", tf)
        print("Time to assure velocity constraint", time.time()-startTime)

        controlPoints = self.move_first_control_point_so_spline_passes_through_start(controlPoints,knotPoints,params.highPriorityStart,[10,10])
        controlPoints = self.move_last_control_point_so_spline_passes_through_end(controlPoints,knotPoints,params.highPriorityEnd,[10,10])

        spline = self.spline_seg(controlPoints, knotPoints)

        



        if plot:
            tmpSpline = self.spline_seg(controlPoints, knotPoints)
            self.plot_spline(tmpSpline,ax)
            # self.plot_spline_from_control_points(controlPoints, knotPoints,ax)

            # ax.plot(path[:,0], path[:,1])
            # for key in nodes.keys():
            #     ax.scatter(nodes[key][0], nodes[key][1], c='b',zorder=1000000)
            #     ax.text(nodes[key][0], nodes[key][1], str(key),c='c',zorder=1000000)


            # g.vs["label"] = [str(key) for key in nodes.keys()]
            # coords = np.array([nodes[key] for key in nodes.keys()])
            # layout = ig.Layout(coords=coords)

            # fig,ax2 = plt.subplots()
            # ig.plot(g,layout= layout,target=ax2)
            # self.plot_constraints(spline, tuple(radarList))
            self.plot_constraints(spline, tuple(radarList),radarParams,radarParamsCov)
            # plt.show()
        
        return controlPoints,tf
        
        
        
        



            
        
        


        
    
    def plot_spline(self, spline,ax):
        controlPoints = spline.c
        tf = spline.t[-params.splineOrder-1]
        t = np.linspace(0, tf, 1000)
        pos = spline(t)
        ax.plot(pos[:,0], pos[:,1])
        # ax.plot(controlPoints[:,0], controlPoints[:,1], 'k--',marker='o',alpha=.5)

    def plot_spline_from_control_points(self, controlPoints, knotPoints,ax):
        t = np.linspace(0, knotPoints[-params.splineOrder-1], 1000)
        spline = self.spline_seg(controlPoints, knotPoints)
        pos = spline(t)
        
        # pos = self.evaluate_spline(t, controlPoints, knotPoints,params.splineOrder)

        ax.plot(pos[:,0], pos[:,1])
        # ax.plot(controlPoints[:,0], controlPoints[:,1], 'k--',marker='o',alpha=.5)
    
    def plot_constraints(self, spline, radarList,radarParams = None, radarParamsCov=None, numConstraintSamples=100,ax= None):
        tf = spline.t[-1]
        t = np.linspace(0, tf, params.numConstraintSamples)
        # t = np.linspace(0, tf, numConstraintSamples)
        controlPoints = spline.c
        knotPoints = spline.t
        if not self.uncertainRadar:
            pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,numConstraintSamples)
        else:
            pd, u, v, pos = self.spline_constraints_uncertain_radar(radarParams,radarParamsCov, controlPoints, knotPoints,numConstraintSamples)
            # print()
        maxpdIndex = np.argmax(pd)
        closestRadarIndex = np.argmin(np.linalg.norm(pos[maxpdIndex]-np.array([radar.position for radar in radarList]),axis=1)) 
        if ax is None:
            fig,ax = plt.subplots()
        c = ax.scatter(pos[:,0], pos[:,1], c=pd)
        fig.colorbar(c, ax=ax)
        # c = ax.scatter(pos[:,0], pos[:,1],s=5)

        
def main():
    radarList = create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm)
    dataIndex = 639
    # dataIndex = 400
    radarParams = np.load("saved_data/currentData/estimated_params/"+str(dataIndex)+".npy")
    print("radarParams", radarParams)
    radarParamsCov = np.load("saved_data/currentData/estimated_params_cov/"+str(dataIndex)+".npy")
    hpp = HighPriorityPathPlannerDeterministic(tuple(radarList))
    pdMap = ProbabilityOfDetectionMap(params.X_test,tuple(radarList))
    fig,ax = plt.subplots()
    if hpp.uncertainRadar:
        Z = uncertainVoronoiPathIntialization.safe_corridors_uncertain_radar(params.X_test,params.probabilityOfDetectionThreshold, params.thresholdConfidence, radarParams, radarParamsCov,params.radarRecieveGain,0, params.radarWavelength,params.radarWavelengthPriorVariance, params.agentRadarCrossSection, params.radarPulseWidth,params.radarPulseWidthPriorVariance, params.radarSystemTemperature,params.radarSystemTemperaturePriorVariance, params.radarProbabilityOfFalseAlarm,params.radarProbabilityOfFalseAlarmPriorVariance) 
    else:
        Z = pdMap.groundTruthpdMap
    c = ax.pcolormesh(params.X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),params.X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints))
    # hpp.plan_deterministic_path(tuple(radarList),plot=True,ax=ax)
    startTime = time.time()
    hpp.plan_uncertain_path(tuple(radarList),radarParams,radarParamsCov,plot=True,ax=ax)
    print("path planning time", time.time()-startTime)

    # _,_ = hpp.get_initial_guess_voronoi(radarList,params.bounds,radarParams,radarParamsCov,plot=True,ax=ax)
    

    if ax is None:
        fig,ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # c = pdMap.plot_mean(ax,plotGroundTruth=True)
    fig.colorbar(c, ax=ax)
    hpp.plot_spline(hpp.spline,ax)
    hpp.plot_constraints(hpp.spline, tuple(radarList),radarParams,radarParamsCov)
    plt.show()
    
    
if __name__ == "__main__":
    main()
        
