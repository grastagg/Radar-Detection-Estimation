import numpy as np
from scipy.constants import c
import jax.numpy as jnp
import jax
# from scipy.constants import boltzman
from scipy.constants import k as boltzman
import os
import getpass

# np.random.seed(1241)

# randomSeed = 91231
randomSeed = 1111
# randomSeed = 11024122
# randomSeed = 1102042
np.random.seed(randomSeed)

        
def db_to_amplitude(db):
    return 10**(db/10)

def frequency_to_wavelength(freq):
    return c/freq
        
#TODO: To make simulation more realistic compute the max range the radar can detect a UAV and the max range the UAV can detect the radar or define them probabalistically

    

#simulation parameters
# bounds = (18000.0, 18000.0) #meters
bounds = (22000.0, 22000.0) #meters
numTestPoints = 500

# numTestPoints = 1
# numTestPoints =30
simulationEndTime = 10000
simulationTimestep = 0.1
plotTimeStep = 0.5

#high priority path plannings
highPriorityStart = (0,0)
highPriorityEnd = (bounds[0],bounds[1])
highPriorityStraitLineA = highPriorityStart[1] - highPriorityEnd[1]
highPriorityStraitLineB = highPriorityEnd[0] - highPriorityStart[0]
highPriorityStraitLineC = highPriorityStart[0]*highPriorityEnd[1] - highPriorityEnd[0] * highPriorityStart[1]
probabilityOfDetectionThreshold = 0.15
thresholdConfidence = 0.70
highPriorityAgentRadarCrossSection = .1

#radar parameters
# radarPositions = [(6000,10000),(16000, 10000), (18000,3000),(2000,3000)]
# radarPhases = [0,np.pi, np.pi/2, np.pi/3, .123]
# radarAngularRates = [3,2.5,3.5,4]
def find_radius_from_radar_pd(radarOuputPower, radarTransmitGain, radarRecieveGain, radarWavelength, radarPulseWidth, radarProbabilityOfFalseAlarm, radarSystemTemperature, agentRadarCrossSection, pd):
    erp = radarOuputPower*radarTransmitGain
    G_r = radarRecieveGain
    lamb = radarWavelength
    sigma = agentRadarCrossSection
    tua = radarPulseWidth
    Pfa = radarProbabilityOfFalseAlarm
    Ts = radarSystemTemperature
    k = boltzman
    R = (((erp*G_r*lamb**2*sigma*tua)/((np.log(Pfa)/np.log(pd))-1))*(1/((4*np.pi)**3*k*Ts)))**.25
    return R

def get_random_parameters(mean, range, numSamples):
    return np.random.uniform(mean-range, mean+range, numSamples)


numRadar = 13

radarOutputPower = 10000.0
# radarOutputPower = 5000
radarOutputPowerRange =10000.0
radarOutputPowerList = get_random_parameters(radarOutputPower, radarOutputPowerRange, numRadar)

radarTransmitGaindb = 10.0
radarTransmitGainRange = 10.0
radarTransmitGainList = get_random_parameters(radarTransmitGaindb, radarTransmitGainRange, numRadar)
# radarTransmitGaindb = 16
radarRecieveGaindb = 10.0
radarRecieveGainRange = 0.0
radarRecieveGainList = get_random_parameters(radarRecieveGaindb, radarRecieveGainRange, numRadar)

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

minInterRadarDistList = 1.1*np.array([find_radius_from_radar_pd(radarOutputPowerList[i], radarTransmitGainList[i], radarRecieveGainList[i], radarWavelength, radarPulseWidth, radarProbabilityOfFalseAlarm, radarSystemTemperature, highPriorityAgentRadarCrossSection, probabilityOfDetectionThreshold) for i in range(numRadar)])
minRadarDistFromStartList = 1.9*(minInterRadarDistList/2) 


radarWeights = np.sqrt(np.sqrt(np.array([outputPower*transmitGain*recieveGain for outputPower,transmitGain,recieveGain in zip(radarOutputPowerList,radarTransmitGainList,radarRecieveGainList)])))


# minRadarDistFromStart = 7000
# minInterRadarDist = 6000
# minInterRadarDist = 4000
def find_min_dist_to_other_radar(potentialRadarPosition, currentRadarPositions):
    mindist = 1000000
    minIndex = -1
    for i,tempRadar in enumerate(currentRadarPositions):
        dist = np.linalg.norm(tempRadar - potentialRadarPosition)
        if dist < mindist:
            mindist = dist
            minIndex = i
    return mindist,minIndex

def find_min_weighted_dist_to_other_radar(potentialRadarPosition, currentRadarPositions, radarWeights):
    mindist = np.inf
    minIndex = -1
    for i,tempRadar in enumerate(currentRadarPositions):
        dist = np.linalg.norm(tempRadar - potentialRadarPosition)/radarWeights[i]
        if dist < mindist:
            mindist = dist
            minIndex = i
    if minIndex == -1:
        return np.inf, -1
    mindist = np.linalg.norm(currentRadarPositions[minIndex] - potentialRadarPosition)
    return mindist,minIndex

def find_radar_position(currentRadarPositions):
    potentialRadarPosition = np.random.uniform(0,bounds[0]-2000,2) 
    # distToStart = np.linalg.norm(potentialRadarPosition)
    distToStart = np.linalg.norm(potentialRadarPosition-highPriorityStart)
    distToEnd = np.linalg.norm(potentialRadarPosition-highPriorityEnd)
    minDistToOtherRadar,minRadarIndex = find_min_weighted_dist_to_other_radar(potentialRadarPosition, currentRadarPositions,radarWeights)

    radarIndex = len(radarPositions)

    minInterRadarDist = minInterRadarDistList[radarIndex] + minInterRadarDistList[minRadarIndex]

    minRadarDistFromStart = minRadarDistFromStartList[radarIndex]
    notFound = distToStart < minRadarDistFromStart or minDistToOtherRadar < minInterRadarDist or distToEnd < minRadarDistFromStart

    while notFound:
        potentialRadarPosition = np.random.uniform(0,bounds[0],2) 
        distToStart = np.linalg.norm(potentialRadarPosition)
        distToEnd = np.linalg.norm(potentialRadarPosition-highPriorityEnd)
        minDistToOtherRadar,radarIndex = find_min_weighted_dist_to_other_radar(potentialRadarPosition, currentRadarPositions, radarWeights)
        minInterRadarDist = minInterRadarDistList[radarIndex] + minInterRadarDistList[minRadarIndex]
        # notFound = distToStart < minRadarDistFromStart or minDistToOtherRadar < minInterRadarDist
        notFound = distToStart < minRadarDistFromStart or minDistToOtherRadar < minInterRadarDist or distToEnd < minRadarDistFromStart
    return potentialRadarPosition


for i in range(numRadar):
            
    radarPositions.append(find_radar_position(radarPositions))
    radarPhases.append(np.random.uniform(0,np.pi,1)[0])
    radarAngularRates.append(np.random.uniform(2,4,1)[0])

safePdDists = []

for i in range(numRadar):
    safePdDists.append(find_radius_from_radar_pd(radarOutputPowerList[i], radarTransmitGainList[i], radarRecieveGainList[i], radarWavelength, radarPulseWidth, radarProbabilityOfFalseAlarm, radarSystemTemperature, highPriorityAgentRadarCrossSection, probabilityOfDetectionThreshold))

safePdDists = np.array(safePdDists)

#agent parameters
# agentInitialStates = [[100,100,np.pi/4]]
# agentInitialStates = [[100,100,np.pi/4],[200,200,np.pi/3]]
agentInitialStates = [[100,100,np.pi/4],[200,200,np.pi/3],[150,150,np.pi/5]]
numAgents = len(agentInitialStates)
# agentSensingRange = 10000
agentSensingRange = 5000
agentPowerMeasurementStdDev = .0001
agentAngleMeasurementStdDev = (2*np.pi/180)
agentELINTAnteneaGaindb = 1
agentELINTAnteneaGain = db_to_amplitude(agentELINTAnteneaGaindb)
agentELINTSystemLoss = 1
radarMeasurementCoeff = (agentELINTAnteneaGain * radarWavelength**2)/((4*np.pi)**2 * agentELINTSystemLoss)
radarMeasurementCoeffDB = 10*np.log10(radarMeasurementCoeff)
agentRadarCrossSection = .1

agentPathHistorydt = 5

measurementCov = np.array([[agentAngleMeasurementStdDev**2,0],[0,agentPowerMeasurementStdDev**2]])



#unknown parameters
radarRecieveGainPriorMean= radarRecieveGain
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
plotObjectiveFunction = False 
plotChanceConstraints =False 
plotPd = True 
plotPdCov = False 
plotBestMeasurement = False 



#path planning
pathLengthMultiplier = 2
numObjectiveFunctionSamples = 20
lowPrioritySafetyBestMeasurementTradeoff = .1
lowPriorityDistanceBestMeasurementTradeoff = .6
agentSpeed = 134
numControlPoints = 40
# numControlPoints = 20
maxTurnRate = 1
velocityBounds = [100,134]
numSamplesPerInterval = 3
numConstraintSamples = numSamplesPerInterval*(numControlPoints-2)-2
# numConstraintSamples = 22
splineOrder = 3

distFromStraitScale = np.sqrt(bounds[0]**2 + bounds[1]**2)/2
# distFromStraitWeight = 1
distFromStraitWeight = .1
distFromStraitWeight = 0

seperationScale = 1
seperationWeight = .3
# seperationWeight = 0

# nextCovarianceScale = 1e16
nextCovarianceScale = 1e14
# nextCovarianceScale = 1e15
# nextCovarianceScale = 1e22
nextCovarianceWeight = .3
# nextCovarianceWeight = 0

# pathOptTime = 50
pathOptTime = 20
lengthScale = 300




def create_test_points(numTestPoints, bounds):
    x_test = np.linspace(0,bounds[0],numTestPoints)
    y_test = np.linspace(0,bounds[1],numTestPoints)

    X_test = []
    
    for i in range(numTestPoints):
        for j in range(numTestPoints):
            X_test.append(np.array([x_test[i],y_test[j]]))
    return np.array(X_test)
    
    
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

    

c = (radarRecieveGain*radarWavelength**2*agentRadarCrossSection*radarPulseWidth)/((4*np.pi)**3*boltzman*radarSystemTemperature)
erp = radarOutputPower*radarTransmitGain


def create_data_file(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        # os.mkdir(filepath+"estimated_params/")
        # os.mkdir(filepath+"estimated_params_cov/")
        os.mkdir(filepath+"/high_priority_path/")
        for i in range(numRadar):
            os.mkdir(filepath+"radar_"+str(i)+"/")
            os.mkdir(filepath+"radar_"+str(i)+"/estimated_params/")
            os.mkdir(filepath+"radar_"+str(i)+"/estimated_params_cov/")
    else:
        print("Directory already exists")
        return
        # cont = input("overwrite? y/n")
        # if cont == "y":
        #     return
        # else:
        #     exit()
            
    
    
saveDataToFile = True

username = getpass.getuser()
dataFile = "/home/"+ username+"/repos/magiccvs/radar_detection_estimation/saved_data/"+str(randomSeed)+"/"
agentColors = ['b','g','c','m','y','k']


#copy params.py to data folder for reference
def copy_params():
    os.system("cp params.py "+dataFile +"params.py ")



