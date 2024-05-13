import numpy as np
from scipy.constants import k as boltzman
import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import params

from pyoptsparse import Optimization, OPT

from scipy import interpolate

from main_helper import create_radar_list

from probabilityOfDetectionMap import ProbabilityOfDetectionMap


from PythonRobotics.PathPlanning.RRTStar import rrt_star

from scipy.interpolate import make_lsq_spline,make_smoothing_spline, splrep, splev, splprep
from scipy.interpolate import BSpline



class HighPriorityPathPlanner:
    def __init__(self) -> None:
        pass
    

    
    def signal_to_noise_ration(self, effectiveRadarPower, radarRecieverGain, wavelength, radarCrossSection, radarPulseWidth, distance, radarSystemTemperature):
        #antennea gain not i decibels
        return (effectiveRadarPower*radarRecieverGain*wavelength**2*radarCrossSection*radarPulseWidth)/((4*np.pi)**3*distance**4*boltzman*radarSystemTemperature)

    def compute_probability_of_detection_at_xy(self, position, radarParams):
        radarXY = radarParams[0:2]
        distance = np.linalg.norm(radarXY-position)
        snr = self.signal_to_noise_ration(radarParams[2], params.radarRecieveGainPriorMean, params.radarWavelengthPriorMean, params.agentRadarCrossSection, params.radarPulseWidth, distance, params.radarSystemTemperaturePriorMean)
        if snr<0:
            print("STOP")
        return np.array([self.probability_of_detection(params.radarProbabilityOfFalseAlarmPriorMean, snr)])

    def probability_of_detection(self, probabilityOfFalseAlarm, snr):
        return np.exp((np.log(probabilityOfFalseAlarm))/(snr + 1))


    def ground_truth_probability_of_detection(self, X_test, trueRadarParametersList):
        probabilityOfNoDetection = np.ones(len(X_test))
        

        for i,position in enumerate(X_test):
            pd_list = []
            pd_cov_list = []
            for j, radar in enumerate(trueRadarParametersList):
                pdi = self.compute_probability_of_detection_at_xy(position, (radar.position[0], radar.position[1], params.radarOutputPower*params.radarTransmitGain))
                probabilityOfNoDetection[i] *= (1-pdi[0])
        
        
        return 1-probabilityOfNoDetection   


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
            

    def create_evenly_spaced_control_points(self, start, stop, numControlPoints):
        xPoints = np.linspace(start[0] + .1, stop[0], numControlPoints, endpoint=False) 
        yPoints = np.linspace(start[1]+.1, stop[1], numControlPoints, endpoint=False) 
        points = np.hstack((xPoints.reshape((len(xPoints),1)), yPoints.reshape((len(xPoints),1))))
        return points


    def spline_constraints(self, radarList, controlPoints, knotPoints,numConstraintSamples):
        spline = self.spline_seg(controlPoints, knotPoints)
        tf = knotPoints[-1]
        t = np.linspace(0,tf,numConstraintSamples)
        u,v = self.get_turn_rate_and_velocity(t, spline)

        pos = spline(t)
        pd = self.ground_truth_probability_of_detection(spline(t), radarList)
        # return np.max(pdMean), u, v, pos
        return pd, u, v, pos



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

    def create_knot_points(self, t0, tf, numControlPoints):
        #the number of control points
        l = numControlPoints

        #create evenly spaced knot points
        t = np.linspace(t0, tf, l - 2, endpoint=True)

        #add repeated knot points at begining and end
        t = np.append([t0, t0, t0], t)
        t = np.append(t, [tf, tf, tf])
        return t
    
    def create_control_points(self, optimizedControlPoints, currPose,endPose, splineOrder, knotPoints):
        controlPoints = np.zeros((params.numControlPoints,2))
        controlPoints[0,:] = currPose[0:2]
        controlPoints[1:-1,:] = optimizedControlPoints.reshape((params.numControlPoints-2,2))

        # controlPoints[1,0] = np.cos(currPose[2]) * self.currentVelocity * knotPoints[splineOrder + 1] / splineOrder + currPose[0]
        # controlPoints[1,1] = np.sin(currPose[2]) * self.currentVelocity * knotPoints[splineOrder + 1] / splineOrder + currPose[1]


        controlPoints[-1,:] = endPose 
        return controlPoints
        

    
    def plan_deterministic_path(self, radar_list):
        initialControlPoints, tfIntial,ax = self.find_initial_guess_rrt_star(radarList,plot=True)
        initialControlPoints = initialControlPoints[1:-1,:]
        def objective_function(xDict):
            tf = xDict['tf']
            knotPoints = self.create_knot_points(0, tf, params.numControlPoints)
            controlPoints = self.create_control_points(xDict['control_points'], params.highPriorityStart, params.highPriorityEnd, params.splineOrder, knotPoints)
            funcs = {}
            pd, u, v, pos = self.spline_constraints(radar_list, controlPoints, knotPoints,params.numConstraintSamples)
            funcs['obj'] = tf 
            funcs['turn_rate'] = u 
            funcs['velocity'] = v 
            funcs['position'] = pos
            funcs['pd'] = pd
            return funcs, False
        
        straitLineDist = np.linalg.norm(np.array(params.highPriorityEnd) - np.array(params.highPriorityStart))
            

        optProb = Optimization("low priority path", objective_function)
        optProb.addVarGroup(name = "control_points", nVars = 2*(params.numControlPoints-2), varType = 'c', value = initialControlPoints.reshape((2*(params.numControlPoints-2))), lower = 0, upper=params.bounds[1])
        # optProb.addVarGroup(name = "tf", nVars = 1, varType = 'c', value = straitLineDist/params.agentSpeed, lower = 0, upper=params.pathLengthMultiplier * straitLineDist/params.agentSpeed)
        optProb.addVarGroup(name = "tf", nVars = 1, varType = 'c', value = tfIntial, lower = 0, upper=params.pathLengthMultiplier * straitLineDist/params.agentSpeed)
        optProb.addConGroup("turn_rate", params.numConstraintSamples, lower=-params.maxTurnRate, upper=params.maxTurnRate, scale=1.0 / params.maxTurnRate)
        optProb.addConGroup("velocity", params.numConstraintSamples, lower=-params.velocityBounds[0], upper=params.velocityBounds[1], scale=1.0 / params.velocityBounds[1])
        optProb.addConGroup("pd", params.numConstraintSamples, lower=0, upper=params.probabilityOfDetectionThreshold, scale=1.0)
        optProb.addObj("obj")
        opt = OPT("ipopt")
        opt.options['print_level'] = 5
        opt.options['tol'] = 1e-8
        sol = opt(optProb, sens = 'FD')
        print(sol)
        knotPoints = self.create_knot_points(0, sol.xStar['tf'], params.numControlPoints)
        controlPoints = self.create_control_points(sol.xStar['control_points'], params.highPriorityStart, params.highPriorityEnd, params.splineOrder, knotPoints)
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

        tck_x = splrep(t,path[:,0],k=params.splineOrder,t=knots)
        control_points_x = tck_x[1]
        control_points_x = control_points_x[control_points_x != 0]
        control_points_x[0] = params.highPriorityStart[0]
        control_points_x[-1] = params.highPriorityEnd[0]
        spline_x = BSpline(tck_x[0], control_points_x, tck_x[2], extrapolate=False)
        print("x knots", tck_x[0])

        tck_y = splrep(t,path[:,1],k=params.splineOrder,t=knots)
        control_points_y = tck_y[1]
        control_points_y = control_points_y[control_points_y != 0]
        control_points_y[0] = params.highPriorityStart[1]
        control_points_y[-1] = params.highPriorityEnd[1]
        print("len control points", len(control_points_y))
        spline_y = BSpline(tck_y[0], control_points_y, tck_y[2], extrapolate=False)
        combined_control_points = np.hstack((control_points_x.reshape((len(control_points_x),1)), control_points_y.reshape((len(control_points_y),1))))

        combined_knot_points = tck_x[0]
        return combined_control_points,combined_knot_points
    
    def assure_pd_less_than_threshold(self, radarList, controlPoints, knotPoints):
        pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,params.numConstraintSamples)
        num_control_points = len(controlPoints)
        while np.max(pd) > params.probabilityOfDetectionThreshold:
            print("pd", np.max(pd))
            num_control_points += 1
            combined_control_points, combined_knot_points = self.fit_spline_to_path(pos,num_control_points)
            pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)

        return combined_control_points,combined_knot_points
        
    
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
            max_iter=1000)
        path = rrt.planning(animation=plot)
        path = np.flip(np.array(path),axis=0)

        num_control_points = params.numControlPoints
        combined_control_points, combined_knot_points = self.fit_spline_to_path(path,num_control_points)

        #make sure initial pd is less than threshold
        # combined_control_points,combined_knot_points = self.assure_pd_less_than_threshold(radarList, combined_control_points, combined_knot_points)

        # spline.derivative(1)(0)
        tf = 1
        pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)
        print("pd", np.max(pd))
        while np.max(v) > params.velocityBounds[1]:
            # print(np.max(v))
            tf += 1
            combined_knot_points = self.create_knot_points(0, tf, num_control_points)
            pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)
        
        

        print("combined knot points", combined_knot_points)
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
                # plt.show()
        return combined_control_points, tf,ax


        
    
    def plot_spline(self, spline,ax):
        tf = spline.t[-1]
        t = np.linspace(0, tf, 200)
        pos = spline(t)
        ax.plot(pos[:,0], pos[:,1])
    
    def plot_constraints(self, spline, radarList,numConstraintSamples=100):
        tf = spline.t[-1]
        # t = np.linspace(0, tf, params.numConstraintSamples)
        t = np.linspace(0, tf, numConstraintSamples)
        controlPoints = spline.c
        knotPoints = spline.t
        pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,numConstraintSamples)

        fig,axes = plt.subplots(3)
        axes[0].plot(t, pd)
        axes[1].plot(t, u)
        axes[2].plot(t, v)
        
    
    
if __name__ == "__main__":
    hpp = HighPriorityPathPlanner()

    radarList = create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm)
    # hpp.find_initial_guess_rrt_star(radarList)
    pdMap = ProbabilityOfDetectionMap(params.X_test,radarList)
    ax = hpp.plan_deterministic_path(radarList)
    
    # fig,ax = plt.subplots
    # ax.set_aspect('equal')
    c = pdMap.plot_mean(ax,plotGroundTruth=True)
    # fig.colorbar(c, ax=ax)
    hpp.plot_spline(hpp.spline,ax)
    hpp.plot_constraints(hpp.spline, radarList)
    plt.show()
        