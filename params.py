import numpy as np
from scipy.constants import c
import jax.numpy as jnp
import jax

# from scipy.constants import boltzman
from scipy.constants import k as boltzman
import os
import getpass

import inspect
from utils import create_test_points


# np.random.seed(1241)

randomSeed = 10211230
# rng = jax.random.defualt_rng(randomSeed)
rng = np.random.default_rng(randomSeed)
# np.random.seed(randomSeed)


def db_to_amplitude(db):
    return 10 ** (db / 10)


def frequency_to_wavelength(freq):
    return c / freq


# TODO: To make simulation more realistic compute the max range the radar can detect a UAV and the max range the UAV can detect the radar or define them probabalistically


# simulation parameters
# bounds = (18000.0, 18000.0) #meters
bounds = (22000.0, 22000.0)  # meters
numTestPoints = 500

# numTestPoints = 1
# numTestPoints =30
simulationEndTime = 600
# simulationEndTime = 10
simulationTimestep = 0.1
plotTimeStep = 0.5

# high priority path plannings
highPriorityStart = (0, 0)
highPriorityEnd = (bounds[0], bounds[1])
highPriorityStraitLineA = highPriorityStart[1] - highPriorityEnd[1]
highPriorityStraitLineB = highPriorityEnd[0] - highPriorityStart[0]
highPriorityStraitLineC = (
    highPriorityStart[0] * highPriorityEnd[1]
    - highPriorityEnd[0] * highPriorityStart[1]
)
probabilityOfDetectionThreshold = 0.15
thresholdConfidence = 0.70
highPriorityAgentRadarCrossSection = 0.1


# radar parameters
# radarPositions = [(6000,10000),(16000, 10000), (18000,3000),(2000,3000)]
# radarPhases = [0,np.pi, np.pi/2, np.pi/3, .123]
# radarAngularRates = [3,2.5,3.5,4]
def find_radius_from_radar_pd(
    radarOuputPower,
    radarTransmitGain,
    radarRecieveGain,
    radarWavelength,
    radarPulseWidth,
    radarProbabilityOfFalseAlarm,
    radarSystemTemperature,
    agentRadarCrossSection,
    pd,
):
    erp = radarOuputPower * radarTransmitGain
    G_r = radarRecieveGain
    lamb = radarWavelength
    sigma = agentRadarCrossSection
    tua = radarPulseWidth
    Pfa = radarProbabilityOfFalseAlarm
    Ts = radarSystemTemperature
    k = boltzman
    R = (
        ((erp * G_r * lamb**2 * sigma * tua) / ((np.log(Pfa) / np.log(pd)) - 1))
        * (1 / ((4 * np.pi) ** 3 * k * Ts))
    ) ** 0.25
    return R


def get_random_parameters(mean, range, numSamples):
    return rng.uniform(mean - range, mean + range, numSamples)


numRadar = 13
# numRadar = 9

radarOutputPower = 10000.0
# radarOutputPower = 5000
radarOutputPowerRange = 10000.0
radarOutputPowerList = get_random_parameters(
    radarOutputPower, radarOutputPowerRange, numRadar
)

radarTransmitGaindb = 10.0
radarTransmitGainRange = 10.0
radarTransmitGainList = get_random_parameters(
    radarTransmitGaindb, radarTransmitGainRange, numRadar
)
# radarTransmitGaindb = 16
radarRecieveGaindb = 10.0
radarRecieveGainRange = 0.0
radarRecieveGainList = get_random_parameters(
    radarRecieveGaindb, radarRecieveGainRange, numRadar
)

# radarRecieveGaindb = 16
radarTransmitGain = db_to_amplitude(radarTransmitGaindb)
radarRecieveGain = db_to_amplitude(radarRecieveGaindb)

radarFrequency = 3.0e9
radarWavelength = frequency_to_wavelength(radarFrequency)

radarProbabilityOfFalseAlarm = 1.0e-6
radarPulseWidth = 1.1e-5

radarSystemTemperature = 745.4148


radarPositions = []
radarPhases = []
radarAngularRates = []


# safeRadius = find_radius_from_radar_pd(radarOutputPower, radarTransmitGain, radarRecieveGain, radarWavelength, radarPulseWidth, radarProbabilityOfFalseAlarm, radarSystemTemperature, highPriorityAgentRadarCrossSection, probabilityOfDetectionThreshold)

minInterRadarDistList = 1.5 * np.array(
    [
        find_radius_from_radar_pd(
            radarOutputPowerList[i],
            radarTransmitGainList[i],
            radarRecieveGainList[i],
            radarWavelength,
            radarPulseWidth,
            radarProbabilityOfFalseAlarm,
            radarSystemTemperature,
            highPriorityAgentRadarCrossSection,
            probabilityOfDetectionThreshold,
        )
        for i in range(numRadar)
    ]
)
minRadarDistFromStartList = 2.2 * (minInterRadarDistList / 2)


radarWeights = np.sqrt(
    np.sqrt(
        np.array(
            [
                outputPower * transmitGain * recieveGain
                for outputPower, transmitGain, recieveGain in zip(
                    radarOutputPowerList, radarTransmitGainList, radarRecieveGainList
                )
            ]
        )
    )
)


# minRadarDistFromStart = 7000
# minInterRadarDist = 6000
# minInterRadarDist = 4000
def find_min_dist_to_other_radar(potentialRadarPosition, currentRadarPositions):
    mindist = 1000000
    minIndex = -1
    for i, tempRadar in enumerate(currentRadarPositions):
        dist = np.linalg.norm(tempRadar - potentialRadarPosition)
        if dist < mindist:
            mindist = dist
            minIndex = i
    return mindist, minIndex


def find_min_weighted_dist_to_other_radar(
    potentialRadarPosition, currentRadarPositions, radarWeights
):
    mindist = np.inf
    minIndex = -1
    for i, tempRadar in enumerate(currentRadarPositions):
        dist = np.linalg.norm(tempRadar - potentialRadarPosition) / radarWeights[i]
        if dist < mindist:
            mindist = dist
            minIndex = i
    if minIndex == -1:
        return np.inf, -1
    mindist = np.linalg.norm(currentRadarPositions[minIndex] - potentialRadarPosition)
    return mindist, minIndex


def find_radar_position(currentRadarPositions):
    potentialRadarPosition = rng.uniform(0, bounds[0], 2)
    # distToStart = np.linalg.norm(potentialRadarPosition)
    distToStart = np.linalg.norm(potentialRadarPosition - highPriorityStart)
    distToEnd = np.linalg.norm(potentialRadarPosition - highPriorityEnd)
    minDistToOtherRadar, minRadarIndex = find_min_weighted_dist_to_other_radar(
        potentialRadarPosition, currentRadarPositions, radarWeights
    )

    radarIndex = len(radarPositions)

    minInterRadarDist = (
        minInterRadarDistList[radarIndex] + minInterRadarDistList[minRadarIndex]
    )

    minRadarDistFromStart = minRadarDistFromStartList[radarIndex]
    notFound = (
        distToStart < minRadarDistFromStart
        or minDistToOtherRadar < minInterRadarDist
        or distToEnd < minRadarDistFromStart
    )

    maxTries = 10000
    numTries = 0

    while notFound and numTries < maxTries:
        potentialRadarPosition = rng.uniform(0, bounds[0], 2)
        distToStart = np.linalg.norm(potentialRadarPosition)
        distToEnd = np.linalg.norm(potentialRadarPosition - highPriorityEnd)
        minDistToOtherRadar, radarIndex = find_min_weighted_dist_to_other_radar(
            potentialRadarPosition, currentRadarPositions, radarWeights
        )
        minInterRadarDist = (
            minInterRadarDistList[radarIndex] + minInterRadarDistList[minRadarIndex]
        )
        # notfound = disttostart < minradardistfromstart or mindisttootherradar < mininterradardist
        notFound = (
            distToStart < minRadarDistFromStart
            or minDistToOtherRadar < minInterRadarDist
            or distToEnd < minRadarDistFromStart
        )
        numTries += 1

    return potentialRadarPosition, notFound


radarPositionsFound = True
for i in range(numRadar):
    radarPos, notFound = find_radar_position(radarPositions)
    if notFound:
        radarPositionsFound = False
        break
    else:
        radarPositions.append(radarPos)
        radarPhases.append(rng.uniform(0, np.pi, 1)[0])
        radarAngularRates.append(rng.uniform(2, 4, 1)[0])

safePdDists = []

for i in range(numRadar):
    safePdDists.append(
        find_radius_from_radar_pd(
            radarOutputPowerList[i],
            radarTransmitGainList[i],
            radarRecieveGainList[i],
            radarWavelength,
            radarPulseWidth,
            radarProbabilityOfFalseAlarm,
            radarSystemTemperature,
            highPriorityAgentRadarCrossSection,
            probabilityOfDetectionThreshold,
        )
    )

safePdDists = np.array(safePdDists)

# agent parameters
# agentInitialStates = [[100,100,np.pi/4]]
# agentInitialStates = [[100,100,np.pi/4],[200,200,np.pi/3]]
# agentInitialStates = [[100, 100, np.pi / 4], [200, 200, np.pi / 2], [150, 150, 0]]
agentInitialStates = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

#     [0, 0, np.pi / 2],
#     [0, 0, 0],
# ]
numAgents = 3
# get list of zeors for initial states
agentInitialStates = []
agentStartAngle = 0
agentEndAngle = np.pi / 2
agentHeadings = np.linspace(agentStartAngle, agentEndAngle, numAgents)
agentHeadings = np.array([np.pi / 4])
for i in range(numAgents):
    agentInitialStates.append([0, 0, agentHeadings[i]])
print("agentInitialStates", agentInitialStates)
agentSensingRange = 5000
agentPowerMeasurementStdDev = 0.0001
agentAngleMeasurementStdDev = 2 * np.pi / 180
agentELINTAnteneaGaindb = 1
agentELINTAnteneaGain = db_to_amplitude(agentELINTAnteneaGaindb)
agentELINTSystemLoss = 1
radarMeasurementCoeff = (agentELINTAnteneaGain * radarWavelength**2) / (
    (4 * np.pi) ** 2 * agentELINTSystemLoss
)
radarMeasurementCoeffDB = 10 * np.log10(radarMeasurementCoeff)
agentRadarCrossSection = 0.1

agentPathHistorydt = 5

measurementCov = np.array(
    [[agentAngleMeasurementStdDev**2, 0], [0, agentPowerMeasurementStdDev**2]]
)


# unknown parameters
radarRecieveGainPriorMean = radarRecieveGain
radarRecieveGainPriorVariance = 0

radarProbabilityOfFalseAlarmPriorMean = radarProbabilityOfFalseAlarm
radarProbabilityOfFalseAlarmPriorVariance = 0

radarPulseWidthPriorMean = radarPulseWidth
# radarPulseWidthPriorVariance = 1e-6
radarPulseWidthPriorVariance = 0

radarWavelengthPriorMean = radarWavelength
# radarWavelengthPriorVariance = 1e-2
radarWavelengthPriorVariance = 0

radarSystemTemperaturePriorMean = radarSystemTemperature
# radarSystemTemperaturePriorVariance = 10
radarSystemTemperaturePriorVariance = 0

# plotting
plotObjectiveFunction = False
plotChanceConstraints = False
plotPd = True
plotPdCov = False
plotBestMeasurement = False


# path planning
pathLengthMultiplier = 2
numObjectiveFunctionSamples = 20
lowPrioritySafetyBestMeasurementTradeoff = 0.1
lowPriorityDistanceBestMeasurementTradeoff = 0.6
agentSpeed = 134
# agentSpeed = 50
numControlPoints = 40
# numControlPoints = 20
maxTurnRate = 1
velocityBounds = [100, 134]
numSamplesPerInterval = 3
numConstraintSamples = numSamplesPerInterval * (numControlPoints - 2) - 2
# numConstraintSamples = 22
splineOrder = 3

useDistToGoal = True

if useDistToGoal:
    distFromStraitScale = np.sqrt(bounds[0] ** 2 + bounds[1] ** 2)
else:
    distFromStraitScale = np.sqrt(bounds[0] ** 2 + bounds[1] ** 2) / 2
    distFromStraitWeight = 0.025

seperationScale = 0.5
# seperationWeight = 0.3


nextCovarianceScale = 1e14
# nextCovarianceWeight = 0.3

seperationWeight = 0.5
distFromStraitWeight = 0.0
nextCovarianceWeight = 0.5


# pathOptTime = 50
pathOptTime = 20
lengthScale = 300


X_test = create_test_points(numTestPoints, bounds)

# def measurement_model_db(xem, yem, erp_db, x, y):
#     return np.array([[np.arctan2(yem-y, xem-x)], [erp_db + radarMeasurementCoeffDB - 10*np.log10((xem-x)**2 + (yem-y)**2)]])

# def measurement_jacobian_db(xem, yem, erp_db, x, y):
#     d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
#     d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
#     d_h1_d_p_emmitter = 0

#     d_h2_d_x_emmitter = -(20*(xem-x))/(np.log(10)*((xem-x)**2+(yem-y)**2))
#     d_h2_d_y_emmitter = -(20*(yem-y))/(np.log(10)*((yem-y)**2+(xem-x)**2))
#     d_h2_d_p_emmitter = 1

#     return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])


c = (
    radarRecieveGain * radarWavelength**2 * agentRadarCrossSection * radarPulseWidth
) / ((4 * np.pi) ** 3 * boltzman * radarSystemTemperature)
erp = radarOutputPower * radarTransmitGain


saveDataToFile = True

# what type of path planning to use
lowPriorityPathPlanner = "lawnmower"
# pathPlanner = "optimization"

username = getpass.getuser()
# dataFile = (
#     "/home/"
#     + username
#     + "/repos/magiccvs/radar_detection_estimation/saved_data/mc_runs/"
#     + str(randomSeed)
#     + "/"
#     + lowPriorityPathPlanner
#     + "/"
# )
# dataFile = (
#     "/home/"
#     + username
#     + "/repos/magiccvs/radar_detection_estimation/saved_data/ratioData/expCov"
#     + str(sepUncertaintyRatio)
#     + "/expDist"
#     + str(sepDistRatio)
#     + "/"
#     + str(randomSeed)
#     + "/"
#     + lowPriorityPathPlanner
#     + "/"
# )
agentColors = ["b", "g", "c", "m", "y", "k"]


# copy params.py to data folder for reference
