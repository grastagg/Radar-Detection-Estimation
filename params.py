import numpy as np
from radar import RadarCircularPattern
from agent import Agent
from scipy.constants import c

def create_radar_list(radarPositions, radarPhases, radarAngularRates, radarOutputPower, radarTransmitGain, radarRecieveGain, radarWavelength, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    radarList = []
    
    for i in range(len(radarPositions)):
        radarList.append(RadarCircularPattern(position=radarPositions[i], phase=radarPhases[i], angular_rate=radarAngularRates[i], outputPower=radarOutputPower, transmitGain = radarTransmitGain, recieveGain = radarRecieveGain, wavelength = radarWavelength, pulseWidth = radarPulseWidth, systemTemperature = radarSystemTemperature, probabilityOfFalseAlarm=radarProbabilityOfFalseAlarm))
        
    return radarList


def create_agent_list(agentInitialStates, numRadar, agentSensingRange, agentPowerMeasurementStdDev, agentAngleMeasurementStdDev, agentELINTAnteneaGain, agentELINTSystemLoss, radarWavelength, radarCrossSection):
    agentList =[]
    for i in range(len(agentInitialStates)):
        agentList.append(Agent(initialPosition=agentInitialStates[i], numRadar = numRadar, sensingRange = agentSensingRange, powerMeasurementStdDev=agentPowerMeasurementStdDev,angleMeasurementStdDev=agentAngleMeasurementStdDev, elintAntenneaGain=agentELINTAnteneaGain, elintSystemLoss=agentELINTSystemLoss, emittorWavelength=radarWavelength, radarCrossSection = radarCrossSection))
    
    return agentList
        
def db_to_amplitude(db):
    return 10**(db/10)

def frequency_to_wavelength(freq):
    return c/freq
        
#TODO: To make simulation more realistic compute the max range the radar can detect a UAV and the max range the UAV can detect the radar or define them probabalistically
        

#simulation parameters
bounds = (20000, 20000) #meters
numTestPoints = 60
simulationEndTime = 90
simulationTimestep = 0.1
plotTimeStep = 0.5

#radar parameters
# radarPositions = [(10000,10000), (6000,12000),(16000, 10000)]
radarPositions = [(6000,10000),(16000, 10000)]
# radarPositions = [(6000,10000),(10000,10000),(16000, 10000)]
# radarPositions = [(10000,10000)]
radarPhases = [0,np.pi, np.pi/2, np.pi/3]
radarAngularRates = [3,2.5,3.5,4]
radarOutputPower = 10000
radarTransmitGaindb = 18
radarRecieveGaindb = 18
radarTransmitGain = db_to_amplitude(radarTransmitGaindb)
radarRecieveGain = db_to_amplitude(radarRecieveGaindb)
radarFrequency = 3e9
radarWavelength = frequency_to_wavelength(radarFrequency)
# print("radarWavelength",radarWavelength)
radarProbabilityOfFalseAlarm = 1e-6
radarPulseWidth = 1.1e-5
radarSystemTemperature = 745.4148
radarList = create_radar_list(radarPositions, radarPhases, radarAngularRates, radarOutputPower, radarTransmitGain, radarRecieveGain, radarWavelength, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)
print("Actual ERP", radarOutputPower * radarTransmitGain)
print("Actual ERP in DB", 10*np.log10(radarOutputPower * radarTransmitGain))


#agent parameters
# agentInitialStates = [[10,10,0]]
# agentInitialStates = [[10000,4010,0],[6000,6000,-np.pi/4]]
# agentInitialStates = [[10000,4010,0]]
# agentInitialStates = [[5000,5000,0],[5000,15000,0]]
agentInitialStates = [[5000,5000,0]]
agentSensingRange = 10000
agentPowerMeasurementStdDev = .001
agentAngleMeasurementStdDev = (3*np.pi/180)
agentELINTAnteneaGaindb = 18
agentELINTAnteneaGain = db_to_amplitude(agentELINTAnteneaGaindb)
print(agentELINTAnteneaGain)
agentELINTSystemLoss = 1
radarMeasurementCoeff = (agentELINTAnteneaGain * radarWavelength**2)/((4*np.pi)**2 * agentELINTSystemLoss)
radarMeasurementCoeffDB = 10*np.log10(radarMeasurementCoeff)
agentRadarCrossSection = .1
agentList = create_agent_list(agentInitialStates, len(radarList), agentSensingRange, agentPowerMeasurementStdDev, agentAngleMeasurementStdDev, agentELINTAnteneaGain, agentELINTSystemLoss, radarWavelength, agentRadarCrossSection)

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
plotPd = True 
plotPdCov = True
plotBestMeasurement = True 

#chance constraints
probabilityOfDetectionThreshold = 0.5
thresholdConfidence = 0.5


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

    
