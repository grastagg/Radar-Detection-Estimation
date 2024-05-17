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

from scipy.spatial import Voronoi, voronoi_plot_2d

import igraph as ig
import time


from bspline.bsplines import BsplineEvaluation


class HighPriorityPathPlannerDeterministic:
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


    def get_turn_rate_and_velocity(self, numPointsPerInterval, spl):
        # out_d1 = spl.derivative(1)(t)
        # out_d2 = spl.derivative(2)(t)
        # out_d1 = spl.get_spline_derivative_data(len(t),1)
        # out_d2 = spl.get_spline_derivative_data(len(t),2)
        # plt.figure()
        # spl.get_

        
        out_d1,t = spl.get_spline_derivative_data(numPointsPerInterval,1)
        # print("test",out_d1[11])
        out_d2,t = spl.get_spline_derivative_data(numPointsPerInterval,2)
        
        x1_dot = out_d1[:,0]
        x2_dot = out_d1[:,1]
        x1_ddot = out_d2[:,0]
        x2_ddot = out_d2[:,1]
        # x1_dot = out_d1[0,:]
        # x2_dot = out_d1[1,:]
        # x1_ddot = out_d2[0,:]
        # x2_ddot = out_d2[1,:]
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


    def spline_constraints(self, radarList, controlPoints, knotPoints,numConstraintSamplesPerInterval):
        spline = self.spline_seg(controlPoints, knotPoints)
        # tf = knotPoints[-1]
        # t = np.linspace(0,tf,numConstraintSamples)
        u,v = self.get_turn_rate_and_velocity(numConstraintSamplesPerInterval, spline)

        pos,t = spline.get_spline_data(numConstraintSamplesPerInterval)
        # pos = spline(t)
        pd = self.ground_truth_probability_of_detection(pos, radarList)
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
        # spline = interpolate.BSpline(t, control_points, 3)
        spline = BsplineEvaluation(control_points,order= 3,scale_factor=t[-1],clamped=True)

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
        

    
    def plan_deterministic_path(self, radar_list,plot=False):
        startTimer = time.time()
        # initialControlPoints, tfIntial,ax = self.find_initial_guess_rrt_star(radarList,plot=plot)
        initialControlPoints, tfIntial,ax = self.get_initial_guess_voronoi(radarList,params.bounds,plot=plot)
        print("Time to find initial guess", time.time()-startTimer)
        initialControlPoints = initialControlPoints[1:-1,:]


        def objective_function(xDict):
            tf = xDict['tf']
            knotPoints = self.create_knot_points(0, tf, params.numControlPoints)
            controlPoints = self.create_control_points(xDict['control_points'], params.highPriorityStart, params.highPriorityEnd, params.splineOrder, knotPoints)
            funcs = {}
            # pd, u, v, pos = self.spline_constraints(radar_list, controlPoints, knotPoints,params.numConstraintSamples)
            pd, u, v, pos = self.spline_constraints(radar_list, controlPoints, knotPoints,2)
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
        opt.options['print_level'] = 0
        opt.options['max_iter'] = 2000
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

        tck_y = splrep(t,path[:,1],k=params.splineOrder,t=knots)
        control_points_y = tck_y[1]
        control_points_y = control_points_y[control_points_y != 0]
        control_points_y[0] = params.highPriorityStart[1]
        control_points_y[-1] = params.highPriorityEnd[1]
        combined_control_points = np.hstack((control_points_x.reshape((len(control_points_x),1)), control_points_y.reshape((len(control_points_y),1))))

        combined_knot_points = tck_x[0]
        return combined_control_points,combined_knot_points
    
    def assure_pd_less_than_threshold(self, radarList, controlPoints, knotPoints):
        # pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,params.numConstraintSamples)
        pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,)
        num_control_points = len(controlPoints)
        while np.max(pd) > params.probabilityOfDetectionThreshold:
            num_control_points += 1
            combined_control_points, combined_knot_points = self.fit_spline_to_path(pos,num_control_points)
            pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)

        return combined_control_points,combined_knot_points
    
    def assure_velocity_constraint(self, radarList, controlPoints, knotPoints,num_control_points):
        pd, u, v, pos = self.spline_constraints(radarList, controlPoints, knotPoints,params.numConstraintSamplesPerInterval)
        tf=np.linalg.norm(controlPoints[0]-controlPoints[-1])/params.agentSpeed
        print("tf",tf)
        while np.max(v) > params.velocityBounds[1]:
            print("np.max(v)",np.max(v))
            print("tf",tf)
            # print(np.max(v))
            tf += 3
            combined_knot_points = self.create_knot_points(0, tf, num_control_points)
            pd, u, v, pos = self.spline_constraints(radarList, controlPoints, combined_knot_points,params.numConstraintSamplesPerInterval)
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

        #make sure initial pd is less than threshold
        # combined_control_points,combined_knot_points = self.assure_pd_less_than_threshold(radarList, combined_control_points, combined_knot_points)

        # spline.derivative(1)(0)
        # tf = 1
        # pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)
        # while np.max(v) > params.velocityBounds[1]:
        #     # print(np.max(v))
        #     tf += 1
        #     combined_knot_points = self.create_knot_points(0, tf, num_control_points)
        #     pd, u, v, pos = self.spline_constraints(radarList, combined_control_points, combined_knot_points,params.numConstraintSamples)
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
                # plt.show()
        return combined_control_points, tf,ax
    
    # def dist_of_point_to_line_segment(self, x1, y1, x2, y2, x3, y3): # x3,y3 is the point
    #     #https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    #     px = x2-x1
    #     py = y2-y1

    #     norm = px*px + py*py

    #     u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    #     if u > 1:
    #         u = 1
    #     elif u < 0:
    #         u = 0

    #     x = x1 + u * px
    #     y = y1 + u * py

    #     dx = x - x3
    #     dy = y - y3

    #     # Note: If the actual distance does not matter,
    #     # if you only want to compare what this function
    #     # returns to other results of this function, you
    #     # can just return the squared distance instead
    #     # (i.e. remove the sqrt) to gain a little performance

    #     dist = (dx*dx + dy*dy)

    #     return dist
    
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
        C = np.array([0,0])
        D = np.array([0,bounds[1]])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            return intersection
        C = np.array([0,0])
        D = np.array([bounds[0],0])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            return intersection
        C = np.array([0,bounds[1]])
        D = np.array([bounds[0],bounds[1]])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            return intersection
        C = np.array([bounds[0],0])
        D = np.array([bounds[0],bounds[1]])
        intersection = self.intersection_of_two_line_segments(A,B,C,D)
        if intersection is not None:
            return intersection
        return None
    
    # def find_line_segments_from_point_to_closest
    def add_boundary_segments(self,segments,bounds):
        tolerance = 1e-5
        segments_to_add = []
        if np.any(np.isclose(segments[:,:,0],0,atol=tolerance)):
            y_coords = np.sort(segments[np.isclose(segments[:,:,0],0,atol=tolerance)][:,1])
            # temp_segments = []
            for i in range(len(y_coords)+1):
                if i == 0:
                    segments_to_add.append([[0,0],[0,y_coords[i]]])
                elif i == len(y_coords):
                    segments_to_add.append([[0,y_coords[i-1]],[0,bounds[1]]])
                else:
                    segments_to_add.append([[0,y_coords[i-1]],[0,y_coords[i]]])    
        else:
            segments_to_add.append([[0,0],[0,bounds[1]]])
        if np.any(np.isclose(segments[:,:,0], bounds[0],atol=tolerance)):
            y_coords = np.sort(segments[np.isclose(segments[:,:,0], bounds[0],atol=tolerance)][:,1])
            for i in range(len(y_coords)+1):
                if i == 0:
                    segments_to_add.append([[bounds[0],0],[bounds[0],y_coords[i]]])
                elif i == len(y_coords):
                    segments_to_add.append([[bounds[0],y_coords[i-1]],[bounds[0],bounds[1]]])
                else:
                    segments_to_add.append([[bounds[0],y_coords[i-1]],[bounds[0],y_coords[i]]])    
        else:
            segments_to_add.append([[bounds[0],0],[bounds[0],bounds[1]]])
        if np.any(np.isclose(segments[:,:,1],0,atol=1e-5)):
            x_coords = np.sort(segments[np.isclose(segments[:,:,1],0,atol=1e-5)][:,0])
            for i in range(len(x_coords)+1):
                if i == 0:
                    segments_to_add.append([[0,0],[x_coords[i],0]])
                elif i == len(x_coords):
                    segments_to_add.append([[x_coords[i-1],0],[bounds[0],0]])
                else:
                    segments_to_add.append([[x_coords[i-1],0],[x_coords[i],0]])
        else:
            segments_to_add.append([[0,0],[bounds[0],0]])
        if np.any(np.isclose(segments[:,:,1], bounds[1],atol=tolerance)):
            x_coords = np.sort(segments[np.isclose(segments[:,:,1], bounds[1],atol=tolerance)][:,0])
            for i in range(len(x_coords)+1):
                if i == 0:
                    segments_to_add.append([[0,bounds[1]],[x_coords[i],bounds[1]]])
                elif i == len(x_coords):
                    segments_to_add.append([[x_coords[i-1],bounds[1]],[bounds[0],bounds[1]]])
                else:
                    segments_to_add.append([[x_coords[i-1],bounds[1]],[x_coords[i],bounds[1]]])
        else:
            segments_to_add.append([[0,bounds[1]],[bounds[0],bounds[1]]])
                    
        segments = np.vstack((segments,segments_to_add))
        return segments
        
        
    # def remove_segments_too_close_to_point(self, point, segments, minDist):
        



    def get_voronoi_ridge_segements(self,radarList,bounds,plot=False):
        points = np.array([[radar.position[0], radar.position[1]] for radar in radarList])
        vor = Voronoi(points)

        segments = []
        ptp_bound = vor.points.ptp(axis=0)
        center = vor.points.mean(axis=0)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                intersection = self.find_intersection_with_boundary(vor.vertices[simplex[0]], vor.vertices[simplex[1]], bounds)
                if intersection is not None:
                    if np.any(vor.vertices[simplex[0]] <0) or np.any(vor.vertices[simplex[0]] > bounds[0]):
                        segments.append([intersection, vor.vertices[simplex[1]]]) 
                    else:
                        segments.append([vor.vertices[simplex[0]], intersection])
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
                
                far_point = self.find_intersection_with_boundary(vor.vertices[i], far_point, bounds)

                if far_point is not None:
                    segments.append([vor.vertices[i], far_point])


        segments = np.array(segments)
        segments = self.add_boundary_segments(segments,bounds)
        
        
        
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

    def get_initial_guess_voronoi(self,radarList,bounds,plot=False):
        startTime = time.time()
        segments,ax = self.get_voronoi_ridge_segements(radarList,bounds,plot=plot)
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
        controlPoints, knotPoints = self.fit_spline_to_path(path,params.numControlPoints)
        print("Time to fit spline to path", time.time()-startTime)

        startTime = time.time()
        knotPoints,tf = self.assure_velocity_constraint(radarList, controlPoints, knotPoints,params.numControlPoints)
        print("Time to assure velocity constraint", time.time()-startTime)


        if plot:
            tmpSpline = self.spline_seg(controlPoints, knotPoints)
            self.plot_spline(tmpSpline,ax)

            ax.plot(path[:,0], path[:,1], 'r--')
            for key in nodes.keys():
                ax.scatter(nodes[key][0], nodes[key][1], c='b',zorder=1000000)
                ax.text(nodes[key][0], nodes[key][1], str(key),c='c',zorder=1000000)


            g.vs["label"] = [str(key) for key in nodes.keys()]
            coords = np.array([nodes[key] for key in nodes.keys()])
            layout = ig.Layout(coords=coords)

            fig,ax2 = plt.subplots()
            ig.plot(g,layout= layout,target=ax2)
        
        return controlPoints,tf,ax
        
        
        
        



            
        
        


        
    
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
    hpp = HighPriorityPathPlannerDeterministic()

    radarList = create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPower, params.radarTransmitGain, params.radarRecieveGain, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm)
    # hpp.find_initial_guess_rrt_star(radarList,plot=True)
    pdMap = ProbabilityOfDetectionMap(params.X_test,radarList)
    startTime = time.time()
    ax = hpp.plan_deterministic_path(radarList,plot=True)
    print("path planning time", time.time()-startTime)
    # _,_,ax = hpp.get_initial_guess_voronoi(radarList,params.bounds,plot=True)
    
    # fig,ax = plt.subplots
    # ax.set_aspect('equal')
    if ax is None:
        fig,ax = plt.subplots()
    c = pdMap.plot_mean(ax,plotGroundTruth=True)
    # fig.colorbar(c, ax=ax)
    hpp.plot_spline(hpp.spline,ax)
    hpp.plot_constraints(hpp.spline, radarList)
    plt.show()
        