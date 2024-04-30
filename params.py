import numpy as np
from scipy.constants import c
import jax.numpy as jnp
import jax

np.random.seed(191234)
        
def db_to_amplitude(db):
    return 10**(db/10)

def frequency_to_wavelength(freq):
    return c/freq
        
#TODO: To make simulation more realistic compute the max range the radar can detect a UAV and the max range the UAV can detect the radar or define them probabalistically

    

#simulation parameters
bounds = (18000.0, 18000.0) #meters
numTestPoints = 30
# numTestPoints =100 
simulationEndTime = 900
simulationTimestep = 0.1
plotTimeStep = 0.5

#radar parameters
radarPositions = [(6000,10000),(16000, 10000), (18000,3000),(2000,3000)]
radarPhases = [0,np.pi, np.pi/2, np.pi/3, .123]
radarAngularRates = [3,2.5,3.5,4]
radarPositions = []
radarPhases = []
radarAngularRates = []
numRadar = 8

minRadarDistFromStart = 7000
minInterRadarDist = 5000
def find_min_dist_to_other_radar(potentialRadarPosition, currentRadarPositions):
    mindist = 1000000
    for tempRadar in currentRadarPositions:
        dist = np.linalg.norm(tempRadar - potentialRadarPosition)
        if dist < mindist:
            mindist = dist
    return mindist

def find_radar_position(currentRadarPositions):
    potentialRadarPosition = np.random.uniform(0,bounds[0]-2000,2) 
    distToStart = np.linalg.norm(potentialRadarPosition)
    minDistToOtherRadar = find_min_dist_to_other_radar(potentialRadarPosition, currentRadarPositions)
    notFound = distToStart < minRadarDistFromStart or minDistToOtherRadar < minInterRadarDist

    while notFound:
        potentialRadarPosition = np.random.uniform(0,bounds[0]-2000,2) 
        distToStart = np.linalg.norm(potentialRadarPosition)
        minDistToOtherRadar = find_min_dist_to_other_radar(potentialRadarPosition, currentRadarPositions)
        notFound = distToStart < minRadarDistFromStart or minDistToOtherRadar < minInterRadarDist
    return potentialRadarPosition


for i in range(numRadar):
            
    radarPositions.append(find_radar_position(radarPositions))
    radarPhases.append(np.random.uniform(0,np.pi,1)[0])
    radarAngularRates.append(np.random.uniform(2,4,1)[0])
# radarPositions = [(3000,10000),(8000, 10000)]
radarOutputPower = 10000
# radarOutputPower = 2500 
radarTransmitGaindb = 10
# radarTransmitGaindb = 16
radarRecieveGaindb = 10
# radarRecieveGaindb = 16
radarTransmitGain = db_to_amplitude(radarTransmitGaindb)
radarRecieveGain = db_to_amplitude(radarRecieveGaindb)
radarFrequency = 3e9
radarWavelength = frequency_to_wavelength(radarFrequency)
# print("radarWavelength",radarWavelength)
radarProbabilityOfFalseAlarm = 1e-6
radarPulseWidth = 1.1e-5
radarSystemTemperature = 745.4148
print("Actual ERP", radarOutputPower * radarTransmitGain)
print("Actual ERP in DB", 10*np.log10(radarOutputPower * radarTransmitGain))


#agent parameters
# agentInitialStates = [[100,100,np.pi/4]]
agentInitialStates = [[100,100,np.pi/4],[200,200,np.pi/3]]
# agentInitialStates = [[100,100,np.pi/4],[200,200,np.pi/3],[150,150,np.pi/5]]
numAgents = len(agentInitialStates)
# agentSensingRange = 10000
agentSensingRange = 5000
agentPowerMeasurementStdDev = .001
agentAngleMeasurementStdDev = (3*np.pi/180)
agentELINTAnteneaGaindb = 18
agentELINTAnteneaGain = db_to_amplitude(agentELINTAnteneaGaindb)
print(agentELINTAnteneaGain)
agentELINTSystemLoss = 1
radarMeasurementCoeff = (agentELINTAnteneaGain * radarWavelength**2)/((4*np.pi)**2 * agentELINTSystemLoss)
radarMeasurementCoeffDB = 10*np.log10(radarMeasurementCoeff)
agentRadarCrossSection = .1

agentPathHistorydt = 2

measurementCov = np.array([[agentAngleMeasurementStdDev**2,0],[0,agentPowerMeasurementStdDev**2]])



#unknown parameters
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

#plotting
plotObjectiveFunction = True 
plotChanceConstraints =False 
plotPd = False 
plotPdCov = False 
plotBestMeasurement = False 

#chance constraints
probabilityOfDetectionThreshold = 0.1
thresholdConfidence = 0.95


#path planning
pathLengthMultiplier = 1.2
numObjectiveFunctionSamples = 20
lowPrioritySafetyBestMeasurementTradeoff = .1
lowPriorityDistanceBestMeasurementTradeoff = .6
agentSpeed = 134
numControlPoints = 8
maxTurnRate = 1
velocityBounds = [100,134]
numConstraintSamples = 20
splineOrder = 3

distFromStraitScale = np.sqrt(bounds[0]**2 + bounds[1]**2)/2
# distFromStraitWeight = 1
# distFromStraitWeight = .3
distFromStraitWeight = 0

seperationScale = 1
seperationWeight = .2
# seperationWeight = 0

# nextCovarianceScale = 1e19
nextCovarianceScale = 1e20
nextCovarianceWeight = .5
# nextCovarianceWeight = 0

pathOptTime = 30
lengthScale = 300


#high priority path plannings
highPriorityStart = (100,100)
highPriorityEnd = (17900,17900)
highPriorityStraitLineA = highPriorityStart[1] - highPriorityEnd[1]
highPriorityStraitLineB = highPriorityEnd[0] - highPriorityStart[0]
highPriorityStraitLineC = highPriorityStart[0]*highPriorityEnd[1] - highPriorityEnd[0] * highPriorityStart[1]


def create_test_points(numTestPoints, bounds):
    x_test = np.linspace(0,bounds[0],numTestPoints)
    y_test = np.linspace(0,bounds[1],numTestPoints)

    X_test = []
    
    for i in range(numTestPoints):
        for j in range(numTestPoints):
            X_test.append(np.array([x_test[i],y_test[j]]))
    return X_test
    
    
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

def measurement_model(xem, yem, erp, x, y):
    return np.array([[np.arctan2(yem-y, xem-x)], [(erp*radarMeasurementCoeff)/((xem-x)**2 + (yem-y)**2)]])

def measurement_jacobian(xem, yem, erp, x, y):
    d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_p_emmitter = 0

    d_h2_d_x_emmitter = -(2*erp*radarMeasurementCoeff*(xem-x))/((xem-x)**2+(yem-y)**2)**2 
    d_h2_d_y_emmitter = -(2*erp*radarMeasurementCoeff*(yem-y))/((yem-y)**2+(xem-x)**2)**2 
    d_h2_d_p_emmitter = radarMeasurementCoeff/((yem-y)**2+(xem-x)**2)

    return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])
@jax.jit
def measurement_jacobian_jax(xem, yem, erp, x, y):
    d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_p_emmitter = 0

    d_h2_d_x_emmitter = -(2*erp*radarMeasurementCoeff*(xem-x))/((xem-x)**2+(yem-y)**2)**2 
    d_h2_d_y_emmitter = -(2*erp*radarMeasurementCoeff*(yem-y))/((yem-y)**2+(xem-x)**2)**2 
    d_h2_d_p_emmitter = radarMeasurementCoeff/((yem-y)**2+(xem-x)**2)

    return jnp.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    
