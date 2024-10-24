# import numpy as np
import getpass
from time import time
import numpy
from scipy.special import erfinv
from scipy import interpolate
from scipy.stats.qmc import LatinHypercube, scale
from pyoptsparse import Optimization, OPT, IPOPT
from matplotlib.patches import Circle
import jax.numpy as np

# import numpy as np
from jax import grad, jacfwd
import jax
import time
from functools import partial

from statistics import NormalDist
import matplotlib.pyplot as plt
import timeit


from angleUnwrapper import AngleUnwrapper
from highPriorityHelperFunctions import (
    probability_radar_at_point_given_path_history,
    path_safety_prob,
)
from measurement_model import (
    measurement_jacobian_jax,
    measurement_jacobian,
    measurement_model,
)


from lowPriorityPathPlannerHelperFunctions import (
    compute_objective_jax,
    compute_objective_vec,
)


@jax.jit
def find_closest_emitter(pos, estimatedRadarParams):
    closetEmittorIndex = np.argmin(
        np.linalg.norm(np.array(estimatedRadarParams)[:, 0:2] - np.array(pos), axis=1)
    )
    return closetEmittorIndex


@jax.jit
def next_measurement_covariance(
    pos,
    estimatedRadarParams,
    estimatedRadarCovariance,
    measurementCov,
    radarMeasurementCoeff,
):
    x = pos[0]
    y = pos[1]
    estimatedRadarParams = estimatedRadarParams
    estimatedRadarCovariance = estimatedRadarCovariance
    x_em = estimatedRadarParams[0]
    y_em = estimatedRadarParams[1]
    erp = estimatedRadarParams[2]
    H = measurement_jacobian_jax(x_em, y_em, erp, x, y, radarMeasurementCoeff)
    R = measurementCov
    K = (
        estimatedRadarCovariance
        @ H.T
        @ np.linalg.inv(H @ estimatedRadarCovariance @ H.T + R)
    )
    nextCovariance = (np.eye(3) - K @ H) @ estimatedRadarCovariance
    # currentCovDet =np.linalg.det(estimatedRadarCovariance)
    return nextCovariance


@jax.jit
def next_measurement_covariance_determinant(
    pos, estimatedRadarParams_list, estimatedRadarCovariance_list, measurementCov
):
    x = pos[0]
    y = pos[1]
    out = 0
    currCovSum = 0
    for i, estimatedRadarParams in enumerate(estimatedRadarParams_list):
        estimatedRadarCovariance = estimatedRadarCovariance_list[i]
        x_em = estimatedRadarParams[0]
        y_em = estimatedRadarParams[1]
        erp = estimatedRadarParams[2]
        H = measurement_jacobian_jax(x_em, y_em, erp, x, y)
        R = measurementCov
        K = (
            estimatedRadarCovariance
            @ H.T
            @ np.linalg.inv(H @ estimatedRadarCovariance @ H.T + R)
        )
        nextCovariance = (np.eye(3) - K @ H) @ estimatedRadarCovariance
        currentCovDet = np.linalg.det(estimatedRadarCovariance)
        currCovSum += currentCovDet
        out += currentCovDet - np.linalg.det(nextCovariance)

    # nextCovariance = estimatedRadarCovariance - estimatedRadarCovariance@H.T@np.linalg.inv(H@estimatedRadarCovariance@H.T+R)@H@estimatedRadarCovariance
    # return 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(estimatedRadarCovariance)) - 1/2*np.log((2*np.pi*np.exp(1))**3*np.linalg.det(nextCovariance))
    # return out/currCovSum
    return out


@jax.jit
def distance_from_line(x0, y0, x1, y1, x2, y2):
    return np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )


@jax.jit
def distance_from_goal(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


partial(jax.jit, static_argnums=(2, 3))


def get_agent_future_path_waypoint(waypoint, pos, agentSpeed, agentPathHistorydt):
    dist = np.linalg.norm(waypoint - pos)
    time = dist / agentSpeed
    num_future_points = np.floor(time / agentPathHistorydt).astype(int)
    points = np.linspace(pos, waypoint, num_future_points)
    return points


@jax.jit
def kernel_seperation(allAgentPathHistory, next_measurement, lengthScale):
    return -np.sum(
        np.exp(
            -np.linalg.norm(allAgentPathHistory - next_measurement, axis=1)
            / lengthScale
        )
    )


@jax.jit
def find_closest_emitter(pos, estimatedRadarParams):
    return np.argmin(np.linalg.norm(estimatedRadarParams[:, 0:2] - pos, axis=1))


@jax.jit
def objective_function_at_pos_new(
    pos, estimatedRadarParams, estimatedRadarParamsCov, measuremetLocations
):
    # # covScale = 3e15
    # covScale = 1
    # distFromStraitScale = np.sqrt(params.bounds[0]**2+bounds[1]**2)
    # coeffSeperation = .6
    # coeffCovariance = .4
    # coeffDistanceFromStrait = 1
    # lengthScale = params.lengthScale
    x0 = pos[0]
    y0 = pos[1]
    x1 = params.highPriorityStart[0]
    y1 = params.highPriorityStart[1]
    x2 = params.highPriorityEnd[0]
    y2 = params.highPriorityEnd[1]

    distanceFromStraitLinePath = (
        -distance_from_line(x0, y0, x1, y1, x2, y2) / params.distFromStraitScale
    )

    # dist = np.linalg.norm(pos - currentPosition) / 20000
    # objectivFunctionVal = alpha * self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * dist

    # minDistFromOtherMeasurements = np.average(np.linalg.norm(measuremetLocations-pos, axis=1))/self.minSeperationScale
    kernelSeperation = (
        -np.sum(
            np.exp(
                -np.linalg.norm(measuremetLocations - pos, axis=1) / params.lengthScale
            )
        )
        / params.seperationScale
    )
    nexMeasCov = (
        next_measurement_covariance_determinant(
            pos, estimatedRadarParams, estimatedRadarParamsCov
        )
        / params.nextCovarianceScale
    )

    objectiveFunctionVal = (
        params.nextCovarianceWeight * nexMeasCov
        + params.seperationWeight * kernelSeperation
        + params.distFromStraitWeight * distanceFromStraitLinePath
    )

    return objectiveFunctionVal


class SplinePathPlanningLowPriority:
    def __init__(self, params):
        self.params = params

        self.numMeasurements = 0
        self.bestMesurementPlot = None
        self.splinePath = None
        self.previousEmittorParamsList = []
        self.splinePathPlot = None
        self.splineControlPointsPlot = None
        self.currentSplineTime = 0
        self.currentVelocity = (params.velocityBounds[0] + params.velocityBounds[1]) / 2
        self.first = False

        self.useSpline = False
        # self.bestMeasurementLoc = None
        self.bestMeasurementLocList = None
        self.kp = 1

        self.bestMeasurementLocPlotList = []
        self.firstPlot = True

        # desired heading list to unwrap heading
        self.numHeadingsToSave = 10
        # self.savedHeadings = []
        # self.savedDesiredHeadings = []

        self.angleUnwrapperDesiredAngle = AngleUnwrapper(
            window_size=self.numHeadingsToSave
        )
        self.angleUnwrapperCurrentAngle = AngleUnwrapper(
            window_size=self.numHeadingsToSave
        )

        # self.minSeperationScale = np.sqrt(2)*params.bounds[0]
        # self.covScale = 3e20
        # self.distFromStraitScale = 5000

        self.bestMeasurementLocSeperation = 5000.0
        self.plotTest = False

        self.numOptStartLocations = 5

        xStart = np.linspace(
            params.bounds[0] / (self.numOptStartLocations + 1),
            params.bounds[0] - params.bounds[0] / (self.numOptStartLocations + 1),
            self.numOptStartLocations,
        )
        yStart = np.linspace(
            params.bounds[1] / (self.numOptStartLocations + 1),
            params.bounds[1] - params.bounds[1] / (self.numOptStartLocations + 1),
            self.numOptStartLocations,
        )
        self.optStartLocationsX, self.optStartLocationsY = np.meshgrid(xStart, yStart)

        self.figNum = 0
        self.timeSinceLastOpt = self.params.pathOptTime + 1

        self.initialHeadings = np.linspace(
            -numpy.pi, numpy.pi, self.numOptStartLocations
        )

        lowerBound = -numpy.pi
        upperBound = numpy.pi

        # self.initialHeadingList = scale(LatinHypercube(params.numAgents).random(self.numOptStartLocations-2),lowerBound,upperBound)
        # self.initialHeadingList = numpy.append(self.initialHeadingList, numpy.zeros((1,params.numAgents)),axis=0)

        self.initialWaypointList = scale(
            LatinHypercube(2 * params.numAgents, seed=params.rng).random(
                self.numOptStartLocations
            ),
            0,
            params.bounds[1],
        )
        # print("self.initialWaypointList",self.initialWaypointList)
        # fig,ax = plt.subplots()
        # ax.set_xlim([0,params.bounds[0]])
        # ax.set_ylim([0,params.bounds[1]])
        # # ax.scatter(self.initialWaypointList[0::3],self.initialWaypointList[1::3])
        # for i in range(self.numOptStartLocations):
        #     ax.scatter(self.initialWaypointList[i,0::2],self.initialWaypointList[i,1::2],label=str(i))
        # ax.legend()

        # plt.show()

        self.timeSinceLastOpt = self.params.pathOptTime + 1
        self.gridSearch = True
        self.objectiveFunctionPlotIndex = 0

    def get_control(self, dt, currentPose, low_priority_agent_index):
        u = np.pi / 4
        v = (self.params.velocityBounds[0] + self.params.velocityBounds[1]) / 2
        if self.useSpline:
            if self.splinePath is not None:
                u, v = self.get_turn_rate_and_velocity(
                    np.array([self.currentSplineTime]), self.splinePath
                )
                u = u[0]
                v = v[0]
                self.currentSplineTime += dt
            self.currentVelocity = v
        else:
            if self.bestMeasurementLocList is not None:
                u, v = self.get_turn_rate_and_velocity_waypoint(
                    self.bestMeasurementLocList[low_priority_agent_index],
                    currentPose,
                    low_priority_agent_index,
                )

        return u, v

    def get_turn_rate_and_velocity_waypoint(
        self, bestMeasurementLoc, currentPose, lowPriorityAgentIndex
    ):
        desiredHeading = numpy.arctan2(
            bestMeasurementLoc[1] - currentPose[1],
            bestMeasurementLoc[0] - currentPose[0],
        )
        # self.angleUnwrapperDesiredAngle.add_angle(desiredHeading)
        # self.angleUnwrapperCurrentAngle.add_angle(currentHeading)
        # desiredHeading = self.angleUnwrapperDesiredAngle.unwrap_angles()[-1]
        # currentHeading = self.angleUnwrapperCurrentAngle.unwrap_angles()[-1]
        # if lowPriorityAgentIndex == 0:
        #     print("desiredHeading",desiredHeading)
        # print("currentHeading",currentHeading)
        v = (self.params.velocityBounds[0] + self.params.velocityBounds[1]) / 2
        # u = self.kp * (desiredHeading - currentPose[2])
        # u = self.kp * (desiredHeading - currentHeading)
        u = desiredHeading
        return u, v

    def newParamsDifferent(self, estimatedRadarParamsList):
        for i in range(len(estimatedRadarParamsList)):
            for j in range(len(estimatedRadarParamsList[i])):
                if (
                    estimatedRadarParamsList[i][j]
                    != self.previousEmittorParamsList[i][j]
                ):
                    return True

        return False

    def get_distance_agent_to_waypoints(self, agentList, waypoints):
        distances = numpy.zeros((len(agentList),))
        for i, agent in enumerate(agentList):
            distances[i] = numpy.linalg.norm(
                numpy.array(agent.position[0:2]) - waypoints[i]
            )
        return distances

    def update_path(
        self,
        estimatedRadarParamsList,
        estimatedRadarParamsCovList,
        probabilityOfDetectionMap,
        agentList,
        numMeasurements,
        measurementLocations,
        dt,
        allAgentesCurrentPos,
    ):
        if self.bestMeasurementLocList is not None:
            distances = self.get_distance_agent_to_waypoints(
                agentList, self.bestMeasurementLocList
            )
            if numpy.any(distances < 50):
                self.bestMeasurementLocList = (
                    self.optimize_next_best_measurement_waypoint(
                        agentList,
                        estimatedRadarParamsList,
                        estimatedRadarParamsCovList,
                        probabilityOfDetectionMap,
                        measurementLocations,
                        allAgentesCurrentPos,
                    )
                )
                self.timeSinceLastOpt = 0
                # self.bestMeasurementLocList = self.optimize_next_best_measurement_distance_constrained(agentList, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, measurementLocations, allAgentesCurrentPos)
            elif self.timeSinceLastOpt > self.params.pathOptTime:
                self.bestMeasurementLocList = (
                    self.optimize_next_best_measurement_waypoint(
                        agentList,
                        estimatedRadarParamsList,
                        estimatedRadarParamsCovList,
                        probabilityOfDetectionMap,
                        measurementLocations,
                        allAgentesCurrentPos,
                    )
                )
                self.timeSinceLastOpt = 0
        else:
            if len(estimatedRadarParamsList) > 0:
                # self.bestMeasurementLocList = [[agent.position[0]-1000,agent.position[1]] for agent in agentList]
                # self.bestMeasurementLocList = self.optimize_next_best_measurement_distance_constrained(agentList, estimatedRadarParamsList, estimatedRadarParamsCovList, probabilityOfDetectionMap, measurementLocations, allAgentesCurrentPos)
                self.bestMeasurementLocList = (
                    self.optimize_next_best_measurement_waypoint(
                        agentList,
                        estimatedRadarParamsList,
                        estimatedRadarParamsCovList,
                        probabilityOfDetectionMap,
                        measurementLocations,
                        allAgentesCurrentPos,
                    )
                )
                self.timeSinceLastOpt = 0
        self.timeSinceLastOpt += dt

    def chance_constraint(self, rho, delta, mean, var):
        # return (mean - rho) > (-erfinv(2*delta-1)*np.sqrt(2*var))
        # return (mean - rho) > (erfinv(-2*delta+1)*np.sqrt(2*var))
        return (NormalDist(mu=mean, sigma=np.sqrt(var)).cdf(rho)) > delta
        # return (mean - rho) > (erfinv(-2*delta+1)*np.sqrt(2*var))

    def chance_constraints_at_points(self, X_test, probabilityOfDetectionMap):
        constraintMet = np.zeros((len(X_test), 1))

        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        rho = self.params.probabilityOfDetectionThreshold
        delta = self.params.thresholdConfidence
        for i, mean in enumerate(pdMean):
            var = pdVar[i]
            constraintMet[i] = self.chance_constraint(rho, delta, mean, var)
        return constraintMet

    def objective_function(
        self,
        X_test,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        probabilityOfDetectionMap,
        currentPosition,
        pathHistory,
    ):
        alpha = self.params.lowPrioritySafetyBestMeasurementTradeoff
        objectivFunctionVal = numpy.zeros((len(X_test), 1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        for i, pos in enumerate(X_test):
            objectivFunctionVal[i] = objective_function_at_pos_new(
                pos, estimatedRadarParams, estimatedRadarParamsCov, pathHistory
            )
        return objectivFunctionVal

    def probability_of_detection_at_points(
        self,
        X_test,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        probabilityOfDetectionMap,
    ):
        objectivFunctionVal = np.zeros((len(X_test), 1))
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar(X_test, estimatedRadarParams, estimatedRadarParamsCov)
        pdMean = probabilityOfDetectionMap.pdMap
        pdVar = probabilityOfDetectionMap.pdCovMap
        for i, mean in enumerate(pdMean):
            var = pdVar[i]
            objectivFunctionVal[i] = mean
        return objectivFunctionVal

    def objective_function_at_pos(
        self,
        pos,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        probabilityOfDetectionMap,
        currentPosition,
    ):
        # alpha =self.params.lowPrioritySafetyBestMeasurementTradeoff
        alpha = self.params.lowPriorityDistanceBestMeasurementTradeoff
        # pdMean, pdVar = probabilityOfDetectionMap.compute_probability_of_detection_at_points_multiple_radar([pos], estimatedRadarParams, estimatedRadarParamsCov, False)
        # var = pdVar[i]
        # objectivFunctionVal = alpha * self.next_measurement_covariance_determinant(pos, estimatedRadarParams, estimatedRadarParamsCov) - (1-alpha) * pdMean
        dist = np.linalg.norm(pos - currentPosition) / 20000
        objectivFunctionVal = (
            alpha
            * self.next_measurement_covariance_determinant(
                pos, estimatedRadarParams, estimatedRadarParamsCov
            )
            - (1 - alpha) * dist
        )
        return objectivFunctionVal

    def agent_seperation_constraint(self, pos, agentBestMeausrementLocList):
        return (
            np.linalg.norm(agentBestMeausrementLocList - pos)
            - self.bestMeasurementLocSeperation
        )

    def agent_speeration_constraint_jacobian(self, pos, agentBestMeausrementLocList):
        return (
            (agentBestMeausrementLocList - pos)
            / numpy.linalg.norm(agentBestMeausrementLocList - pos)
        ).reshape((1, 2))

    ############# NEW TEST PATH PLANNER ####################

    # def find_closest_emitter(self, pos, estimatedRadarParams):
    #     closetEmittorIndex = np.argmin(np.linalg.norm(np.array(estimatedRadarParams)[:,0:2]-np.array(pos), axis=1))
    #     # if len(estimatedRadarParams) == 2:
    #     #     print()
    #     # # closetEmittorIndex = np.argmin(np.linalg.norm(estimatedRadarParams[:,0:2]-pos, axis=1))
    #     # # print("closetEmittorIndex TEST",closetEmittorIndex)
    #     # closetEmittorIndex = 0
    #     # for i in range(len(estimatedRadarParams)):
    #     #     if np.linalg.norm(estimatedRadarParams[i][0:2]-pos) < np.linalg.norm(estimatedRadarParams[closetEmittorIndex][0:2]-pos):
    #     #         closetEmittorIndex = i
    #     # print("closetEmittorIndex",closetEmittorIndex)
    #     return closetEmittorIndex

    def find_most_uncertain_emitter(self, estimatedRadarParamsCov):
        mostUncertainIndex = 0
        for i in range(len(estimatedRadarParamsCov)):
            if np.linalg.det(estimatedRadarParamsCov[i]) > np.linalg.det(
                estimatedRadarParamsCov[mostUncertainIndex]
            ):
                mostUncertainIndex = i
        return mostUncertainIndex

    def find_agent_optimization_order(
        self, agentList, estimatedRadarParams, estimatedRadarParamsCov
    ):
        agentOrder = []
        mostUncertainIndex = self.find_most_uncertain_emitter(estimatedRadarParamsCov)

        distToMostUncertain = numpy.zeros((len(agentList),))
        for i, agent in enumerate(agentList):
            distToMostUncertain[i] = numpy.linalg.norm(
                agent.position[0:2] - estimatedRadarParams[mostUncertainIndex][0:2]
            )

        agentOrder = numpy.argsort(distToMostUncertain, axis=0)
        return agentOrder

    def get_agent_future_path(self, heading, agent, time):
        num_future_points = time // params.agentPathHistorydt
        first_point = np.array(agent.position[0:2])
        last_point = np.array(
            agent.position[0:2]
        ) + time * self.params.agentSpeed * np.array([np.cos(heading), np.sin(heading)])
        points = np.linspace(first_point, last_point, num_future_points)
        # print("points",points)
        return points

    def objective_function_for_best_measurement_dist_constrained(
        self,
        headings,
        agentList,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        allAgentPathHistory,
        velocity,
        pathTime,
        agentOrder,
    ):
        x1 = self.params.highPriorityStart[0]
        y1 = self.params.highPriorityStart[1]
        x2 = self.params.highPriorityEnd[0]
        y2 = self.params.highPriorityEnd[1]
        estimatedRadarParamsCov_temp = estimatedRadarParamsCov.copy()
        distanceFromStraitLinePathObj = 0
        kernelSeperationObj = 0
        allAgentPathHistory_temp = allAgentPathHistory.copy()

        probRadarAtLocationObjective = 0

        for k in range(len(headings)):
            next_measurement = np.array(
                agentList[agentOrder[k]].position[0:2]
            ) + velocity * pathTime * np.array(
                [np.cos(headings[agentOrder[k]]), np.sin(headings[agentOrder[k]])]
            )
            if len(estimatedRadarParams) > 0:
                closest_emitter_index = find_closest_emitter(
                    np.array(next_measurement), np.array(estimatedRadarParams)
                )
                # closest_emitter_index = find_closest_emitter(next_measurement, estimatedRadarParams)
                # estimatedRadarParamsCov_temp[closest_emitter_index] = self.next_measurement_covariance(next_measurement, estimatedRadarParams, estimatedRadarParamsCov_temp, closest_emitter_index)
                estimatedRadarParamsCov_temp[closest_emitter_index] = (
                    next_measurement_covariance(
                        next_measurement,
                        estimatedRadarParams[closest_emitter_index],
                        estimatedRadarParamsCov_temp[closest_emitter_index],
                    )
                )

            x0 = next_measurement[0]
            y0 = next_measurement[1]
            distanceFromStraitLinePathObj += (
                distance_from_line(x0, y0, x1, y1, x2, y2) / params.distFromStraitScale
            )
            kernelSeperationObj += kernel_seperation(
                allAgentPathHistory_temp, next_measurement
            )
            # probRadarAtLocationObjective += probability_radar_at_point_given_path_history(next_measurement, allAgentPathHistory_temp,self.params.radarTransmitGain,self.params.radarOutputPower,self.params.agentELINTAnteneaGain,self.params.radarWavelength,self.params.radarSystemTemperature,self.params.radarProbabilityOfFalseAlarm,params.radarPulseWidth)

            # kernelSeperationObj += -np.sum(np.exp(-np.linalg.norm(allAgentPathHistory-next_measurement, axis=1)/params.lengthScale))
            futurePath = self.get_agent_future_path(
                headings[agentOrder[k]], agentList[agentOrder[k]], pathTime
            )
            allAgentPathHistory_temp = np.vstack((allAgentPathHistory_temp, futurePath))
        obj_cov = 0
        for i in range(len(estimatedRadarParams)):
            obj_cov += np.linalg.det(estimatedRadarParamsCov_temp[i])
        (
            returnself.params.nextCovarianceWeight
            * obj_cov
            / params.nextCovarianceScale
            - self.params.seperationWeight
            * kernelSeperationObj
            / params.seperationScale
            + self.params.distFromStraitWeight
            * distanceFromStraitLinePathObj
            / params.distFromStraitScale
        )
        # returnself.params.nextCovarianceWeight * obj_cov/params.nextCovarianceScale +self.params.seperationWeight*probRadarAtLocationObjective/params.seperationScale +self.params.distFromStraitWeight * distanceFromStraitLinePathObj/params.distFromStraitScale

    def waypoint_position_constraint(
        self, heading, allAgentCurrentPos, velocity, pathTime
    ):
        return (
            np.hstack(
                (
                    np.cos(heading).reshape((len(heading), 1)),
                    np.sin(heading).reshape((len(heading), 1)),
                )
            )
            * velocity
            * pathTime
            + allAgentCurrentPos
        ).reshape((2 * len(heading),))

    def waypoint_position_constraint_jac(
        self, heading, allAgentCurrentPos, velocity, pathTime
    ):
        out = numpy.hstack(
            (
                numpy.array([-numpy.sin(heading[0]), numpy.cos(heading[0])])
                * velocity
                * pathTime,
                numpy.zeros((2 * (len(heading) - 1),)),
            )
        ).reshape((2 * len(heading), 1))
        for i in range(1, heading.shape[0]):
            col = (
                numpy.array([-numpy.sin(heading[i]), numpy.cos(heading[i])])
                * velocity
                * pathTime
            )
            for k in range(i):
                col = numpy.hstack((numpy.zeros((2,)), col))
            for k in range(i, heading.shape[0] - 1):
                col = numpy.hstack((col, numpy.zeros((2,))))
            col = col.reshape((2 * len(heading), 1))
            out = numpy.hstack((out, col))

        return out

    # def get_agent_future_path_waypoint(self,waypoint, agent):
    #     pos = np.array(agent.position[0:2])
    #     dist = np.linalg.norm(waypoint-pos)
    #     time = dist/params.agentSpeed
    #     num_future_points = time//params.agentPathHistorydt
    #     points = np.linspace(pos, waypoint, num_future_points.astype(int))
    #     # print("points",points)
    #     return points

    # def find_closest_emitter(self, pos, estimatedRadarParams):
    #     closetEmittorIndex = np.argmin(np.linalg.norm(np.array(estimatedRadarParams)[:,0:2]-np.array(pos), axis=1))
    #     return closetEmittorIndex

    def objective_function_for_best_measurement_waypoints(
        self,
        waypoints,
        agentList,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        allAgentPathHistory,
        velocity,
        pathTime,
        agentOrder,
        test=False,
    ):
        x1 = self.params.highPriorityStart[0]
        y1 = self.params.highPriorityStart[1]
        x2 = self.params.highPriorityEnd[0]
        y2 = self.params.highPriorityEnd[1]
        estimatedRadarParamsCov_temp = estimatedRadarParamsCov.copy()
        distanceFromStraitLinePathObj = 0
        exploreObj = 0
        covObj = 0
        allAgentPathHistory_temp = allAgentPathHistory.copy()

        if self.plotTest:
            tempOptLocations = []
            allAgentPathHistory_tempPlot = allAgentPathHistory_temp.copy()

        for k in range(len(agentList)):
            index = agentOrder[k]
            next_measurement = waypoints[2 * index : 2 * index + 2]
            # if len(estimatedRadarParams)>0:
            closest_emitter_index = find_closest_emitter(
                np.array(next_measurement), np.array(estimatedRadarParams)
            )
            # test= next_measurement_covariance(next_measurement, estimatedRadarParams[closest_emitter_index], estimatedRadarParamsCov_temp[closest_emitter_index])
            # estimatedRadarParamsCov_temp[closest_emitter_index]=test
            covDetSum = 0
            countCov = 0
            for q in range(len(estimatedRadarParams)):
                if len(estimatedRadarParams[q]) > 0:
                    nextMeasCov = next_measurement_covariance(
                        next_measurement,
                        estimatedRadarParams[q],
                        estimatedRadarParamsCov_temp[q],
                        self.params.measurementCov,
                        self.params.radarMeasurementCoeff,
                    )
                    covDetSum += np.linalg.det(nextMeasCov)
                    countCov += 1

            estimatedRadarParamsCov_temp[closest_emitter_index] = (
                next_measurement_covariance(
                    next_measurement,
                    estimatedRadarParams[closest_emitter_index],
                    estimatedRadarParamsCov_temp[closest_emitter_index],
                    self.params.measurementCov,
                    self.params.radarMeasurementCoeff,
                )
            )

            # covObj += np.linalg.det(estimatedRadarParamsCov_temp[closest_emitter_index])
            covObj += covDetSum / countCov

            x0 = next_measurement[0]
            y0 = next_measurement[1]
            # distanceFromStraitLinePathObj += distance_from_line(x0,y0,x1,y1,x2,y2)
            distanceFromStraitLinePathObj += distance_from_goal(x0, y0, x2, y2)

            futurePath = get_agent_future_path_waypoint(
                next_measurement,
                np.array(agentList[index].position[0:2]),
                self.params.agentSpeed,
                self.params.agentPathHistorydt,
            )

            # explorationObj = probability_radar_at_point_given_path_history(futurePath, allAgentPathHistory_temp,self.params.radarTransmitGain,self.params.radarOutputPower,self.params.agentELINTAnteneaGain,self.params.radarWavelength,self.params.radarSystemTemperature,self.params.radarProbabilityOfFalseAlarm,params.radarPulseWidth)
            explore = path_safety_prob(
                allAgentPathHistory_temp,
                futurePath,
                self.params.radarTransmitGain,
                self.params.radarOutputPower,
                self.params.agentELINTAnteneaGain,
                self.params.radarWavelength,
                self.params.radarSystemTemperature,
                self.params.radarProbabilityOfFalseAlarm,
                self.params.radarPulseWidth,
            )
            exploreMean = np.mean(explore)
            exploreObj += exploreMean
            if test:
                print("agent index", index)
                print("exploreMean", exploreMean)
                print(
                    "cov",
                    np.linalg.det(estimatedRadarParamsCov_temp[closest_emitter_index]),
                )
                print("dist", distance_from_line(x0, y0, x1, y1, x2, y2))

            if self.plotTest:
                numTestPoints = 25
                testX = numpy.linspace(0, self.params.bounds[0], numTestPoints)
                testY = numpy.linspace(0, self.params.bounds[1], numTestPoints)
                testX, testY = numpy.meshgrid(testX, testY)
                objF = numpy.zeros((numTestPoints, numTestPoints))

                for i in range(numTestPoints):
                    # print("i",i)
                    for j in range(numTestPoints):
                        closest_emitter_index = find_closest_emitter(
                            np.array([testX[i, j], testY[i, j]]),
                            np.array(estimatedRadarParams),
                        )
                        cov = next_measurement_covariance(
                            np.array([testX[i, j], testY[i, j]]),
                            estimatedRadarParams[closest_emitter_index],
                            estimatedRadarParamsCov_temp[closest_emitter_index],
                            self.params.measurementCov,
                            self.params.radarMeasurementCoeff,
                        )
                        # dist = distance_from_line(testX[i,j],testY[i,j],x1,y1,x2,y2)
                        dist = distance_from_goal(testX[i, j], testY[i, j], x2, y2)
                        futurePath_temp = get_agent_future_path_waypoint(
                            np.array([testX[i, j], testY[i, j]]),
                            np.array(agentList[index].position[0:2]),
                            self.params.agentSpeed,
                            self.params.agentPathHistorydt,
                        )
                        explore = path_safety_prob(
                            allAgentPathHistory_tempPlot,
                            futurePath_temp,
                            self.params.radarTransmitGain,
                            self.params.radarOutputPower,
                            self.params.agentELINTAnteneaGain,
                            self.params.radarWavelength,
                            self.params.radarSystemTemperature,
                            self.params.radarProbabilityOfFalseAlarm,
                            self.params.radarPulseWidth,
                        )
                        # explore = path_safety_prob(allAgentPathHistory_temp,futurePath_temp,params.radarTransmitGain,params.radarOutputPower,self.params.agentELINTAnteneaGain,self.params.radarWavelength,self.params.radarSystemTemperature,params.radarProbabilityOfFalseAlarm,self.params.radarPulseWidth)
                        exploreMean = np.mean(explore)
                        obj = (
                            self.params.nextCovarianceWeight
                            * np.linalg.det(cov)
                            / self.params.nextCovarianceScale
                            - self.params.seperationWeight * exploreMean / 0.5
                            + self.params.distFromStraitWeight
                            * dist
                            / self.params.distFromStraitScale
                        )
                        objF[i, j] = obj

                tempOptIndex = numpy.nanargmin(objF)
                tempOptLocations.append(testX.flat[tempOptIndex])
                tempOptLocations.append(testY.flat[tempOptIndex])
                print("opt obj", objF.flat[tempOptIndex])

                futurePathTempPlot = get_agent_future_path_waypoint(
                    np.array([testX.flat[tempOptIndex], testY.flat[tempOptIndex]]),
                    np.array(agentList[index].position[0:2]),
                    self.params.agentSpeed,
                    self.params.agentPathHistorydt,
                )
                allAgentPathHistory_tempPlot = numpy.vstack(
                    (allAgentPathHistory_tempPlot, futurePathTempPlot)
                )

                fig, ax = plt.subplots()
                ax.set_title("agent " + str(agentOrder[k]))
                c = ax.pcolormesh(testX, testY, objF)
                ax.set_aspect("equal")
                fig.colorbar(c, ax=ax)
                ax.scatter(waypoints[0::2], waypoints[1::2])

                ax.scatter(
                    allAgentPathHistory_tempPlot[:, 0],
                    allAgentPathHistory_tempPlot[:, 1],
                )
                ax.scatter(tempOptLocations[-2], tempOptLocations[-1], marker="x")

            allAgentPathHistory_temp = np.vstack((allAgentPathHistory_temp, futurePath))

        objectiveFunc = (
            self.params.nextCovarianceWeight * covObj / self.params.nextCovarianceScale
            - self.params.seperationWeight * exploreObj / 0.5
            + self.params.distFromStraitWeight
            * distanceFromStraitLinePathObj
            / self.params.distFromStraitScale
        )
        if self.plotTest:
            # plt.show()
            self.plotTest = False
            return tempOptLocations, objectiveFunc

        # obj_cov = 0
        # for i in range(len(estimatedRadarParams)):

        #     obj_cov += np.linalg.det(estimatedRadarParamsCov_temp[i])
        return objectiveFunc

    def find_initial_headings(self, agentList, estimatedParams_list):
        initialHeadings = numpy.linspace(0, 2 * np.pi, len(agentList))
        for i in range(len(agentList)):
            closesEmmiterIndex = find_closest_emitter(
                np.array(agentList[i].position[0:2]), np.array(estimatedParams_list)
            )
            initialHeadings[i] = np.arctan2(
                -agentList[i].position[1] + estimatedParams_list[closesEmmiterIndex][1],
                -agentList[i].position[0] + estimatedParams_list[closesEmmiterIndex][0],
            )
        return initialHeadings

    ######TEST waypoint optimization#######
    def distance_constraint(self, waypoints, allAgentCurrentPos):
        waypoints = waypoints.reshape((len(allAgentCurrentPos), 2))
        return np.linalg.norm(waypoints - allAgentCurrentPos, axis=1)

    def find_initial_waypoints(self, agentList, estimatedParams_list):
        initialWaypoints = numpy.zeros((len(agentList) * 2,))
        for i in range(len(agentList)):
            closesEmmiterIndex = self.find_closest_emitter(
                agentList[i].position[0:2], estimatedParams_list
            )
            initialWaypoints[2 * i] = estimatedParams_list[closesEmmiterIndex][0]
            initialWaypoints[2 * i + 1] = estimatedParams_list[closesEmmiterIndex][1]
        return initialWaypoints

    def optimize_next_best_measurement_waypoint(
        self,
        agentList,
        estimatedParams_list,
        estimatedRadarCovariance_list,
        probabilityOfDetectionMap,
        measurementLocations,
        allAgentesCurrentPos,
    ):
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        # allAgentPathHistory = np.concatenate([agent.pathHistory for agent in agentList])
        allAgentPathHistory = np.array(allAgentPathHistory)

        if len(estimatedParams_list) > 0:
            agentOrder = self.find_agent_optimization_order(
                agentList, estimatedParams_list, estimatedRadarCovariance_list
            )
        else:
            agentOrder = range(len(agentList))
        print("agentOrder", agentOrder)
        # agentOrder = range(len(agentList))

        if self.gridSearch:
            optWaypoints = self.optimize_waypoints_discrete_search(
                agentList,
                estimatedParams_list,
                estimatedRadarCovariance_list,
                allAgentPathHistory,
                agentOrder,
            )
            bestMeasurementLocList = []
            for k in range(len(agentList)):
                index = agentOrder[k]
                bestMeasurementLocList.append(optWaypoints[2 * index : 2 * index + 2])
            return bestMeasurementLocList

        # initialWaypoints = self.find_initial_waypoints(agentList, estimatedParams_list)
        initialWaypoints = self.initialWaypointList[0]
        # initialHeadingsList = []

        self.plotTest = False
        tempObjectiveFuncScale = np.abs(
            self.objective_function_for_best_measurement_waypoints(
                initialWaypoints,
                agentList,
                estimatedParams_list,
                estimatedRadarCovariance_list,
                allAgentPathHistory,
                self.params.agentSpeed,
                self.params.pathOptTime,
                agentOrder,
            )
        )

        # run once to compile
        # tempgrad = grad(self.objective_function_for_best_measurement_waypoints)(initialWaypoints, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime, agentOrder)

        # numTest = 100
        # objtime = timeit.timeit(lambda: self.objective_function_for_best_measurement_dist_constrained(initialHeadings, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime, agentOrder), number=numTest)
        # gradTime = timeit.timeit(lambda: grad(self.objective_function_for_best_measurement_dist_constrained)(initialHeadings, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime, agentOrder), number=numTest)

        def objective_function(xdict):
            waypoints = xdict["waypoints"]
            funcs = {}
            funcs["obj"] = (
                self.objective_function_for_best_measurement_waypoints(
                    waypoints,
                    agentList,
                    estimatedParams_list,
                    estimatedRadarCovariance_list,
                    allAgentPathHistory,
                    self.params.agentSpeed,
                    self.params.pathOptTime,
                    agentOrder,
                )
                / tempObjectiveFuncScale
            )
            # funcs['distances'] = numpy.array(self.distance_constraint(waypoints, allAgentesCurrentPos))
            return funcs, False

        def sens(xDict, funcs):
            waypoints = xDict["waypoints"]
            funcsSens = {}
            funcsSens["obj"] = {
                "waypoints": grad(
                    self.objective_function_for_best_measurement_waypoints
                )(
                    waypoints,
                    agentList,
                    estimatedParams_list,
                    estimatedRadarCovariance_list,
                    allAgentPathHistory,
                    self.params.agentSpeed,
                    self.params.pathOptTime,
                    agentOrder,
                )
                / tempObjectiveFuncScale
            }
            # funcsSens['distances'] = {"waypoints" : jacfwd(self.distance_constraint,argnums=0)(waypoints, allAgentesCurrentPos).reshape((len(agentList),len(agentList)*2))}
            return funcsSens, False

        optWaypoints = None
        bestOptVal = np.inf
        totalTime = 0
        numOptimization = 0
        for initialWaypoint in self.initialWaypointList:
            start = time.time()
            waypoints, fStar = self.run_optimization_waypoints(
                objective_function, sens, agentList, initialWaypoint
            )

            print("fStar", fStar)
            numOptimization += 1
            totalTime += time.time() - start
            if fStar is not None:
                if fStar < bestOptVal:
                    print("new best", fStar)
                    bestOptVal = fStar
                    optWaypoints = waypoints
        print("totalTime", totalTime)
        print("averageTime", totalTime / numOptimization)
        print("number of optimizations", numOptimization)

        ##########TEST##########
        # self.plotTest =True
        print("OPT TEST")
        # tempOptLocations,optObjectiveFunctionVal = self.objective_function_for_best_measurement_waypoints(optWaypoints, agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime, agentOrder,test=True)
        print("optimal waypoints", optWaypoints)

        start = time.time()
        tempOptLocations = self.optimize_waypoints_discrete_search(
            agentList,
            estimatedParams_list,
            estimatedRadarCovariance_list,
            allAgentPathHistory,
            agentOrder,
        )
        print("discrete search time", time.time() - start)
        self.plotTest = False
        print("tempOptLocations", tempOptLocations)
        tempOptLocations = numpy.array(tempOptLocations).squeeze()
        tempObjectiveFunctionVal = (
            self.objective_function_for_best_measurement_waypoints(
                tempOptLocations,
                agentList,
                estimatedParams_list,
                estimatedRadarCovariance_list,
                allAgentPathHistory,
                self.params.agentSpeed,
                self.params.pathOptTime,
                agentOrder,
            )
        )
        optObjectiveFunctionVal = (
            self.objective_function_for_best_measurement_waypoints(
                optWaypoints,
                agentList,
                estimatedParams_list,
                estimatedRadarCovariance_list,
                allAgentPathHistory,
                self.params.agentSpeed,
                self.params.pathOptTime,
                agentOrder,
            )
        )
        print("optimal objective function value", optObjectiveFunctionVal)
        print("tempObjectiveFunctionVal", tempObjectiveFunctionVal)
        fig, ax = plt.subplots()
        ax.scatter(
            tempOptLocations[0::2], tempOptLocations[1::2], label="discrete search"
        )
        ax.scatter(optWaypoints[0::2], optWaypoints[1::2], label="optimal")
        ax.legend()
        plt.show()

        bestMeasurementLocList = []
        for k in range(len(agentList)):
            bestMeasurementLocList.append(optWaypoints[2 * k : 2 * k + 2])
        return bestMeasurementLocList

    def optimize_waypoints_discrete_search(
        self,
        agentList,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        allAgentPathHistory,
        agentOrder,
    ):
        x1 = self.params.highPriorityStart[0]
        y1 = self.params.highPriorityStart[1]
        x2 = self.params.highPriorityEnd[0]
        y2 = self.params.highPriorityEnd[1]
        estimatedRadarParamsCov_temp = estimatedRadarParamsCov.copy()
        allAgentPathHistory_temp = allAgentPathHistory.copy()

        showPlot = False
        if showPlot:
            fig, axs = plt.subplots(1, len(agentList))

        numTestPoints = 50
        testX = numpy.linspace(0, self.params.bounds[0], numTestPoints)
        testY = numpy.linspace(0, self.params.bounds[1], numTestPoints)
        testX, testY = numpy.meshgrid(testX, testY)
        pos = np.vstack((testX.ravel(), testY.ravel())).T
        tempOptLocations = []

        start = time.time()
        for k in range(len(agentList)):
            index = agentOrder[k]

            objF = compute_objective_vec(
                pos,
                np.array(estimatedRadarParams),
                np.array(estimatedRadarParamsCov_temp),
                x1,
                y1,
                x2,
                y2,
                allAgentPathHistory_temp,
                agentList[index].position,
                self.params.agentSpeed,
                self.params.agentPathHistorydt,
                self.params.measurementCov,
                self.params.radarMeasurementCoeff,
                self.params.radarTransmitGain,
                self.params.radarOutputPower,
                self.params.agentELINTAnteneaGain,
                self.params.radarWavelength,
                self.params.radarSystemTemperature,
                self.params.radarProbabilityOfFalseAlarm,
                self.params.radarPulseWidth,
                self.params.nextCovarianceWeight,
                self.params.nextCovarianceScale,
                self.params.seperationWeight,
                self.params.seperationScale,
                self.params.distFromStraitWeight,
                self.params.distFromStraitScale,
            )

            tempOptIndex = numpy.nanargmin(objF)
            tempOptLocations.append(testX.flat[tempOptIndex])
            tempOptLocations.append(testY.flat[tempOptIndex])
            closestEmittorOpt = find_closest_emitter(
                np.array([testX.flat[tempOptIndex], testY.flat[tempOptIndex]]),
                np.array(estimatedRadarParams),
            )
            cov = next_measurement_covariance(
                np.array([testX.flat[tempOptIndex], testY.flat[tempOptIndex]]),
                estimatedRadarParams[closestEmittorOpt],
                estimatedRadarParamsCov_temp[closestEmittorOpt],
                self.params.measurementCov,
                self.params.radarMeasurementCoeff,
            )
            estimatedRadarParamsCov_temp[closestEmittorOpt] = cov

            if showPlot:
                ax = axs[k]
                ax.set_title("agent " + str(agentOrder[k]))
                ax.set_aspect("equal")
                c = ax.pcolormesh(testX, testY, objF.reshape(testX.shape))
                # fig.colorbar(c, ax=ax)
                ax.scatter(
                    allAgentPathHistory_temp[:, 0], allAgentPathHistory_temp[:, 1]
                )
                ax.scatter(tempOptLocations[-2], tempOptLocations[-1], marker="x")

                ax.set_aspect("equal")
                ax.scatter(
                    allAgentPathHistory_temp[:, 0], allAgentPathHistory_temp[:, 1]
                )
                ax.scatter(tempOptLocations[-2], tempOptLocations[-1], marker="x")
                ax.scatter(agentList[index].position[0], agentList[index].position[1])

            futurePath = get_agent_future_path_waypoint(
                np.array([testX.flat[tempOptIndex], testY.flat[tempOptIndex]]),
                np.array(agentList[index].position[0:2]),
                self.params.agentSpeed,
                self.params.agentPathHistorydt,
            )

            allAgentPathHistory_temp = np.vstack((allAgentPathHistory_temp, futurePath))
        print("discrete search time", time.time() - start)

        if showPlot:
            ax.scatter(
                np.array(estimatedRadarParams)[:, 0],
                np.array(estimatedRadarParams)[:, 1],
                color="red",
            )
            file_name = (
                "/home/ggs24/repos/magiccvs/radar_detection_estimation/images/objective_function/"
                + str(self.objectiveFunctionPlotIndex)
                + ".png"
            )
            plt.savefig(file_name)
            self.objectiveFunctionPlotIndex += 1

        return tempOptLocations

    def run_optimization_waypoints(
        self, objective_function, sens, agentList, initialWaypoints
    ):
        optProb = Optimization("find best measurement location", objective_function)
        optProb.addVarGroup(
            name="waypoints",
            nVars=2 * len(agentList),
            varType="c",
            value=initialWaypoints,
            lower=0,
            upper=self.params.bounds[1],
        )
        # optProb.addConGroup("distances", len(agentList), lower = 0, upper =self.params.agentSpeed*params.pathOptTime)
        optProb.addObj("obj")
        opt = OPT("ipopt")
        # opt.options['hsllib'] = '/home/grant/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        username = getpass.getuser()
        opt.options["hsllib"] = (
            "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
        )
        opt.options["linear_solver"] = "ma97"
        opt.options["print_level"] = 5

        # opt.options['derivative_test'] = 'first-order'
        # opt.options['derivative_test_perturbation'] = 1e-5
        opt.options["max_iter"] = 50
        opt.options["tol"] = 1e-5
        sol = opt(optProb, sens=sens)
        # sol = opt(optProb, sens = 'fd')

        optHeadings = sol.xStar["waypoints"]

        # if sol.optInform['value'] == 0 or sol.optInform['value'] == 1:
        print("OPTIMIZATION SUCCESSFUL")
        return optHeadings, sol.fStar
        # else:
        #     print("OPTIMIZATION FAILED")
        #     return None,None

    ######TEST waypoint optimization#######

    def optimize_next_best_measurement_distance_constrained(
        self,
        agentList,
        estimatedParams_list,
        estimatedRadarCovariance_list,
        probabilityOfDetectionMap,
        measurementLocations,
        allAgentesCurrentPos,
    ):
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        # allAgentPathHistory = np.concatenate([agent.pathHistory for agent in agentList])
        allAgentPathHistory = np.array(allAgentPathHistory)

        if len(estimatedParams_list) > 0:
            agentOrder = self.find_agent_optimization_order(
                agentList, estimatedParams_list, estimatedRadarCovariance_list
            )
        else:
            agentOrder = range(len(agentList))

        intialHeadingsTemp = self.find_initial_headings(agentList, estimatedParams_list)
        initialHeadingsListTemp = self.initialHeadingList.copy()

        initialHeadingsListTemp = numpy.append(
            initialHeadingsListTemp,
            intialHeadingsTemp.reshape((1, params.numAgents)),
            axis=0,
        )
        initialHeadings = self.initialHeadingList[0]

        tempObjectiveFuncScale = np.abs(
            self.objective_function_for_best_measurement_dist_constrained(
                initialHeadings,
                agentList,
                estimatedParams_list,
                estimatedRadarCovariance_list,
                allAgentPathHistory,
                self.params.agentSpeed,
                self.params.pathOptTime,
                agentOrder,
            )
        )

        # run once to compile
        grad(self.objective_function_for_best_measurement_dist_constrained)(
            initialHeadings,
            agentList,
            estimatedParams_list,
            estimatedRadarCovariance_list,
            allAgentPathHistory,
            self.params.agentSpeed,
            self.params.pathOptTime,
            agentOrder,
        )

        def objective_function(xdict):
            headings = xdict["headings"]
            funcs = {}
            funcs["obj"] = (
                self.objective_function_for_best_measurement_dist_constrained(
                    headings,
                    agentList,
                    estimatedParams_list,
                    estimatedRadarCovariance_list,
                    allAgentPathHistory,
                    self.params.agentSpeed,
                    self.params.pathOptTime,
                    agentOrder,
                )
                / tempObjectiveFuncScale
            )
            pos_con = self.waypoint_position_constraint(
                headings,
                allAgentesCurrentPos,
                self.params.agentSpeed,
                self.params.pathOptTime,
            )
            funcs["pos_con"] = pos_con
            # print("funcs", funcs)
            return funcs, False

        def sens(xDict, funcs):
            # print("in sens")
            # print("xDict",xDict)
            # print("funcs",funcs)
            headings = xDict["headings"]
            funcsSens = {}
            funcsSens["obj"] = {
                "headings": grad(
                    self.objective_function_for_best_measurement_dist_constrained
                )(
                    headings,
                    agentList,
                    estimatedParams_list,
                    estimatedRadarCovariance_list,
                    allAgentPathHistory,
                    self.params.agentSpeed,
                    self.params.pathOptTime,
                    agentOrder,
                )
                / tempObjectiveFuncScale
            }
            # funcsSens['pos_con'] = {"headings" : jacfwd(lambda x: self.waypoint_position_constraint(x,allAgentesCurrentPos,self.params.agentSpeed,self.params.pathOptTime))(headings).reshape((len(agentList)*2, len(agentList)))}
            funcsSens["pos_con"] = {
                "headings": self.waypoint_position_constraint_jac(
                    headings,
                    allAgentesCurrentPos,
                    self.params.agentSpeed,
                    self.params.pathOptTime,
                )
            }
            # print("funcsSens['obj']",funcsSens['obj'])
            # print("funcsSens['pos_con']",funcsSens['pos_con'])
            return funcsSens, False

        optHeadings = None
        bestOptVal = np.inf
        totalTime = 0
        numOptimization = 0
        # for itialHeadings in self.initialHeadingList:
        for itialHeadings in initialHeadingsListTemp:
            start = time.time()
            constraints = self.waypoint_position_constraint(
                itialHeadings,
                allAgentesCurrentPos,
                self.params.agentSpeed,
                self.params.pathOptTime,
            )
            if np.any(constraints < 0):
                print("initial heading out of bounds")
                continue
            elif np.any(constraints > self.params.bounds[1]):
                print("initial heading out of bounds")
                continue
            else:
                headings, fStar = self.run_optimization(
                    objective_function, sens, agentList, itialHeadings
                )
                numOptimization += 1
                totalTime += time.time() - start
                if fStar is not None:
                    if fStar < bestOptVal:
                        bestOptVal = fStar
                        optHeadings = headings
        print("totalTime", totalTime)
        print("averageTime", totalTime / numOptimization)
        print("number of optimizations", numOptimization)

        # fig,ax = plt.subplots()
        # headingPlot1 = numpy.linspace(-1*numpy.pi,1*numpy.pi,50)
        # headingPlot2 = numpy.linspace(-1*numpy.pi,1*numpy.pi,50)
        # [X,Y] = numpy.meshgrid(headingPlot1,headingPlot2)
        # Z = numpy.zeros_like(X)
        # for i in range(len(headingPlot1)):
        #     for j in range(len(headingPlot2)):
        #         Z[j,i] = -self.objective_function_for_best_measurement_dist_constrained(numpy.array([X[j,i],Y[j,i]]), agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime,agentOrder)/tempObjectiveFuncScale
        # ax.pcolormesh(X,Y,Z)
        # objPlot = numpy.zeros_like(headingPlot)
        # for k in range(len(headingPlot)):
        #     objPlot[k] = self.objective_function_for_best_measurement_dist_constrained(np.array([headingPlot[k]]), agentList, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime)/tempObjectiveFuncScale
        # ax.plot(headingPlot, objPlot)
        # ax.scatter(optHeadings, sol.fStar)

        # image = np.array(fig.canvas.renderer.buffer_rgba())
        # plt.imsave("images/headingPlot/"+str(self.figNum)+".png", image)
        # plt.show()
        # ax.scatter(optHeadings[0],optHeadings[1])
        # fig.savefig("images/headingPlot/"+str(self.figNum)+".png")
        # self.figNum += 1

        bestMeasurementLocList = []
        for k in range(len(agentList)):
            bestMeasurementLocList.append(
                np.array(agentList[k].position[0:2])
                + self.params.pathOptTime
                * self.params.agentSpeed
                * np.array([np.cos(optHeadings[k]), np.sin(optHeadings[k])])
            )
        return bestMeasurementLocList

    def run_optimization(self, objective_function, sens, agentList, initialHeadings):
        optProb = Optimization("find best measurement location", objective_function)
        optProb.addVarGroup(
            name="headings",
            nVars=len(agentList),
            varType="c",
            value=initialHeadings,
            lower=-2 * np.pi,
            upper=2 * np.pi,
        )
        optProb.addConGroup(
            "pos_con", 2 * len(agentList), lower=0, upper=params.bounds[1]
        )
        optProb.addObj("obj")
        opt = OPT("ipopt")
        # opt.options['hsllib'] = '/home/grant/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        username = getpass.getuser()
        opt.options["hsllib"] = (
            "/home/" + username + "/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
        )
        # opt.options['hsllib'] = '/home/ggs24/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
        opt.options["linear_solver"] = "ma97"
        opt.options["print_level"] = 5

        # opt.options['derivative_test'] = 'first-order'
        # opt.options['derivative_test_perturbation'] = 1e-5
        opt.options["max_iter"] = 200
        opt.options["tol"] = 1e-8
        sol = opt(optProb, sens=sens)
        # sol = opt(optProb, sens = 'fd')
        print(sol)

        optHeadings = sol.xStar["headings"]

        if sol.optInform["value"] == 0 or sol.optInform["value"] == 1:
            print("OPTIMIZATION SUCCESSFUL")
            return optHeadings, sol.fStar
        else:
            print("OPTIMIZATION FAILED")
            return None, None

    ############# NEW TEST PATH PLANNER ####################

    def optimize_next_best_measurement(
        self,
        agentList,
        estimatedParams_list,
        estimatedRadarCovariance_list,
        probabilityOfDetectionMap,
        measurementLocations,
    ):
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        # allAgentPathHistory = np.concatenate([agent.pathHistory for agent in agentList])
        allAgentPathHistory = np.array(allAgentPathHistory)

        bestMeasurementLocList = []
        tempEstimatedParamsCovList = estimatedRadarCovariance_list.copy()
        for k in range(len(agentList)):

            def objective_function(xdict):
                pos = xdict["pos"]
                # obj = -next_measurement_covariance_determinant(pos, estimatedParams_list, estimatedRadarCovariance_list)
                # obj = -objective_function_at_pos_new(pos, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory)
                obj = -objective_function_at_pos_new(
                    pos,
                    estimatedParams_list,
                    tempEstimatedParamsCovList,
                    allAgentPathHistory,
                )
                funcs = {}
                funcs["obj"] = obj
                if k > 0:
                    funcs["sep"] = self.agent_seperation_constraint(
                        np.array(pos), np.array(bestMeasurementLocList)
                    ).squeeze()
                fail = False
                return funcs, fail

            minObjectiveFunctionVal = np.inf
            minObjFuncValLoc = None

            def sens(xDict, funcs):
                pos = xDict["pos"]
                funcsSens = {}
                # funcsSens['obj'] = {"pos" : -grad(objective_function_at_pos_new)(pos, estimatedParams_list, estimatedRadarCovariance_list, np.array(measurementLocations))}
                # funcsSens['obj'] = {"pos" : -grad(objective_function_at_pos_new)(pos, estimatedParams_list, estimatedRadarCovariance_list, allAgentPathHistory)}
                funcsSens["obj"] = {
                    "pos": -grad(objective_function_at_pos_new)(
                        pos,
                        estimatedParams_list,
                        tempEstimatedParamsCovList,
                        allAgentPathHistory,
                    )
                }

                if k > 0:
                    # funcsSens['seperation'] = {"pos" : grad(self.agent_seperation_constraint)( np.array(pos), np.array(bestMeasurementLocList))}
                    funcsSens["sep"] = {
                        "pos": self.agent_speeration_constraint_jacobian(
                            numpy.array(pos), numpy.array(bestMeasurementLocList)
                        )
                    }
                    # funcsSens['seperation'] = {"pos" : [0.0,0.0]}
                return funcsSens, False

            start = time.time()
            for i in range(self.numOptStartLocations):
                for j in range(self.numOptStartLocations):
                    x_start = self.optStartLocationsX[i][j]
                    y_start = self.optStartLocationsY[i][j]
                    if k > 0:
                        print()

                    optProb = Optimization(
                        "find best measurement location" + str(k), objective_function
                    )
                    optProb.addVarGroup(
                        name="pos",
                        nVars=2,
                        varType="c",
                        value=[x_start, y_start],
                        lower=0,
                        upper=params.bounds[1],
                    )
                    optProb.addObj("obj")
                    if k > 0:
                        optProb.addConGroup("sep", k, lower=0, upper=10000000)
                    opt = OPT("ipopt")
                    opt.options["hsllib"] = (
                        "/home/grant/packages/ThirdParty-HSL/.libs/libcoinhsl.so"
                    )
                    # opt.options['hsllib'] = '/home/ggs24/packages/ThirdParty-HSL/.libs/libcoinhsl.so'
                    opt.options["linear_solver"] = "ma97"
                    opt.options["print_level"] = 0

                    # opt.options['derivative_test'] = 'first-order'
                    opt.options["max_iter"] = 2000
                    opt.options["tol"] = 1e-3
                    sol = opt(optProb, sens=sens)
                    if sol.optInform["value"] == 0:
                        print("OPTIMIZATION SUCCESSFUL")
                    else:
                        print("OPTIMIZATION FAILED")
                        print(sol.optInform)
                        print(sol)

                    print("VAL:", sol.fStar)
                    if sol.fStar < minObjectiveFunctionVal and (
                        sol.optInform["value"] == 0 or sol.optInform["value"] == 1
                    ):
                        print("TEST")
                        minObjectiveFunctionVal = sol.fStar
                        minObjFuncValLoc = sol.xStar["pos"]

            if minObjFuncValLoc is None:
                print("NO OPTIMAL SOLUTION FOUND")
                minObjFuncValLoc = self.bestMeasurementLocList[k]

            bestMeasurementLocList.append(minObjFuncValLoc)
            futurePath = self.get_future_path(
                agentList[k].position, minObjFuncValLoc, self.params.agentSpeed
            )
            tempEstimatedParamsCovList = self.get_future_covariance(
                estimatedParams_list, tempEstimatedParamsCovList, minObjFuncValLoc
            )

            allAgentPathHistory = np.vstack((allAgentPathHistory, futurePath))
            print("BEST VAL:", minObjectiveFunctionVal)
            print("BEST Loc:", bestMeasurementLocList[-1])
            print("time for optimization", time.time() - start)

        # return minObjFuncValLoc
        return bestMeasurementLocList

    def get_future_path(self, currentPose, bestMeasurementLoc, speed):
        dist = np.linalg.norm(currentPose[0:2] - bestMeasurementLoc)
        time = dist / speed
        numPoints = int(time / params.agentPathHistorydt)
        x = np.linspace(currentPose[0], bestMeasurementLoc[0], numPoints)
        y = np.linspace(currentPose[1], bestMeasurementLoc[1], numPoints)
        return np.hstack((x.reshape((len(x), 1)), y.reshape((len(y), 1))))

    def get_future_covariance(
        self, estimatedRadarParamsList, estimatedRadarParamsCovList, bestMeasurementLoc
    ):
        closetEmittorIndex = 0
        for i in range(len(estimatedRadarParamsList)):
            if np.linalg.norm(
                estimatedRadarParamsList[i][0:2] - bestMeasurementLoc
            ) < np.linalg.norm(
                estimatedRadarParamsList[closetEmittorIndex][0:2] - bestMeasurementLoc
            ):
                closetEmittorIndex = i

        x_em = estimatedRadarParamsList[closetEmittorIndex][0]
        y_em = estimatedRadarParamsList[closetEmittorIndex][1]
        erp = estimatedRadarParamsList[closetEmittorIndex][2]
        x = bestMeasurementLoc[0]
        y = bestMeasurementLoc[1]

        H = self.params.measurement_jacobian_jax(x_em, y_em, erp, x, y)
        R = self.params.measurementCov
        K = (
            estimatedRadarParamsCovList[closetEmittorIndex]
            @ H.T
            @ np.linalg.inv(
                H @ estimatedRadarParamsCovList[closetEmittorIndex] @ H.T + R
            )
        )
        nextCovariance = (np.eye(3) - K @ H) @ estimatedRadarParamsCovList[
            closetEmittorIndex
        ]

        estimatedRadarParamsCovList[closetEmittorIndex] = nextCovariance
        return estimatedRadarParamsCovList

    def plot_spline(self, ax, spl, num_points):
        t0 = spl.t[0]
        tf = spl.t[-1]
        t = np.linspace(t0, tf, num_points, endpoint=True)
        x = spl(t)[:, 0]
        y = spl(t)[:, 1]
        if self.splinePathPlot is not None:
            for line in self.splinePathPlot:
                line.remove()
            # for line in self.splineControlPointsPlot:
            #     line.remove()
        self.splinePathPlot = ax.plot(x, y, color="black")
        # plt.scatter(x,y,c = spline_color)
        control_points = spl.c
        # self.splineControlPointsPlot = ax.plot(control_points[:, 0], control_points[:, 1], 'k--', label='Control polygon', marker='o', zorder = 100000)

    def plot_objective_and_constraint(
        self,
        ax,
        X_test,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        probabilityOfDetectionMap,
        currPos,
        numMeasurements,
        agentList,
        agentPlotIndex,
    ):
        # objectiveFunctionVal, constraintMet = objective_function(X_test, estimatedRadarParams, estimatedRadarParamsCov, probabilityOfDetectionMap)
        allAgentPathHistory = []
        for agent in agentList:
            allAgentPathHistory += agent.pathHistory
        allAgentPathHistory = np.array(allAgentPathHistory)

        tempEstimatedParamsCovList = estimatedRadarParamsCov.copy()
        for i in range(agentPlotIndex):
            # futurePath = self.get_future_path(agentList[i].position, self.bestMeasurementLocList[i],self.params.agentSpeed)
            tempEstimatedParamsCovList = self.get_future_covariance(
                estimatedRadarParams,
                tempEstimatedParamsCovList,
                self.bestMeasurementLocList[i],
            )
            # allAgentPathHistory = np.vstack((allAgentPathHistory, futurePath))
            # allAgentPathHistory = np.vstack((allAgentPathHistory, futurePath))

        objectiveFunctionVal = self.objective_function(
            X_test,
            estimatedRadarParams,
            tempEstimatedParamsCovList,
            probabilityOfDetectionMap,
            currPos[0:2],
            allAgentPathHistory,
        )
        # objectiveFunctionVal = self.objective_function(X_test, estimatedRadarParams, tempEstimatedParamsCovList, probabilityOfDetectionMap, currPos[0:2], allAgentPathHistory)

        # objectiveFunctionVal = np.zeros((len(X_test)))

        # for i, pos in enumerate(X_test):
        #     heading = [np.arctan2(pos[1]-currPos[1], pos[0]-currPos[0])]
        #     objectiveFunctionVal[i] = self.objective_function_for_best_measurement_dist_constrained(heading, agentList, estimatedRadarParams, tempEstimatedParamsCovList, allAgentPathHistory,self.params.agentSpeed,self.params.pathOptTime)

        # objectiveFunctionVal[constraintMet ==0] = -1

        c = ax.pcolormesh(
            X_test[:, 0].reshape((params.numTestPoints, params.numTestPoints)),
            X_test[:, 1].reshape((params.numTestPoints, params.numTestPoints)),
            objectiveFunctionVal.reshape((params.numTestPoints, params.numTestPoints)),
        )

        if self.splinePath is not None:
            self.plot_spline(ax, self.splinePath, 100)

        # if self.bestMeasurementLoc is not None:
        if self.bestMeasurementLocList is not None:
            for i in range(params.numAgents):
                if self.firstPlot:
                    self.bestMeasurementLocPlotList.append(
                        Circle(
                            (
                                self.bestMeasurementLocList[i][0],
                                self.bestMeasurementLocList[i][1],
                            ),
                            radius=200,
                            fill=True,
                            color="r",
                            zorder=100000000,
                        )
                    )
                    ax.add_patch(self.bestMeasurementLocPlotList[i])
                else:
                    self.bestMeasurementLocPlotList[i].center = (
                        self.bestMeasurementLocList[i][0],
                        self.bestMeasurementLocList[i][1],
                    )
            self.firstPlot = False

        return c

    def plot_chance_constraints(
        self,
        ax,
        X_test,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        probabilityOfDetectionMap,
        currPos,
        numMeasurements,
    ):
        constraintMet = self.chance_constraints_at_points(
            X_test, probabilityOfDetectionMap
        )
        groundTruthConstraint = probabilityOfDetectionMap.groundTruthpdMap

        c1 = ax.contour(
            X_test[:, 0].reshape((params.numTestPoints, params.numTestPoints)),
            X_test[:, 1].reshape((params.numTestPoints, params.numTestPoints)),
            groundTruthConstraint.reshape((params.numTestPoints, params.numTestPoints)),
            levels=[params.probabilityOfDetectionThreshold],
        )

        c2 = ax.contour(
            X_test[:, 0].reshape((params.numTestPoints, params.numTestPoints)),
            X_test[:, 1].reshape((params.numTestPoints, params.numTestPoints)),
            probabilityOfDetectionMap.pdMap.reshape(
                (params.numTestPoints, params.numTestPoints)
            ),
            levels=[params.probabilityOfDetectionThreshold],
            cmap="hsv",
        )

        # c4 = ax.pcolormesh(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), probabilityOfDetectionMap.pdCovMap.reshape((params.numTestPoints,params.numTestPoints)))

        c3 = ax.contour(
            X_test[:, 0].reshape((params.numTestPoints, params.numTestPoints)),
            X_test[:, 1].reshape((params.numTestPoints, params.numTestPoints)),
            constraintMet.reshape((params.numTestPoints, params.numTestPoints)),
            levels=[-1, 0, 1],
            cmap="coolwarm",
        )
        # c3 = ax.contourf(X_test[:,0].reshape((params.numTestPoints,params.numTestPoints)), X_test[:,1].reshape((params.numTestPoints,params.numTestPoints)), constraintMet.reshape((params.numTestPoints,params.numTestPoints)), levels = [-1,0,1], cmap = 'coolwarm')
        return [c1, c2, c3]


if __name__ == "__main__":
    pathPlanner = SplinePathPlanningLowPriority()
