import numpy as np
from radar import RadarCircularPattern
from agent import Agent

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
        
        
        
        

#simulation parameters
bounds = (1200, 1200)
numTestPoints = 60
simulationEndTime = 40
simulationTimestep = .1

#radar parameters
radarPositions = [(600,600)]
# radarPositions = [(200,800), (500,600), (800,800), (1100,600)]
radarPhases = [0,np.pi, np.pi/2, np.pi/3]
radarAngularRates = [3,2.5,3.5,4]
radarOutputPower = 100000
radarTransmitGain = 1
radarRecieveGain = 1
radarWavelength = 0.003
radarProbabilityOfFalseAlarm = 1e-6
radarPulseWidth = 1.1e-5
radarSystemTemperature = 745.4148
radarList = create_radar_list(radarPositions, radarPhases, radarAngularRates, radarOutputPower, radarTransmitGain, radarRecieveGain, radarWavelength, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm)


#agent parameters
# agentInitialStates = [[10,10,0]]
agentInitialStates = [[600,10,0]]
agentSensingRange = 900
agentPowerMeasurementStdDev = 0.001
agentAngleMeasurementStdDev = (2*np.pi/180)
agentELINTAnteneaGain = 1
agentELINTSystemLoss = 1
radarMeasurementCoeff = (agentELINTAnteneaGain * radarWavelength**2)/((4*np.pi)**2 * agentELINTSystemLoss)
agentRadarCrossSection = 1
agentList = create_agent_list(agentInitialStates, len(radarList), agentSensingRange, agentPowerMeasurementStdDev, agentAngleMeasurementStdDev, agentELINTAnteneaGain, agentELINTSystemLoss, radarWavelength, agentRadarCrossSection)


#unknown parameters
radarRecieveGainPriorMean = 1
radarRecieveGainPriorVariance = .1

radarProbabilityOfFalseAlarmPriorMean = radarProbabilityOfFalseAlarm
radarPulseWidthPriorMean = radarPulseWidth
radarWavelengthPriorMean = radarWavelength
radarSystemTemperaturePriorMean = radarSystemTemperature


def create_test_points(numTestPoints, bounds):
    x_test = np.linspace(0,bounds[0],numTestPoints)
    y_test = np.linspace(0,bounds[1],numTestPoints)

    X_test = []
    
    for i in range(numTestPoints):
        for j in range(numTestPoints):
            X_test.append(np.array([x_test[i],y_test[j]]))
    return X_test
    