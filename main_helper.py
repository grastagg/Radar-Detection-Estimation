import params
from agent import Agent
from radar import RadarCircularPattern



def create_radar_list(radarPositions, radarPhases, radarAngularRates, radarOutputPowers, radarTransmitGains, radarRecieveGains, radarWavelength, radarPulseWidth, radarSystemTemperature, radarProbabilityOfFalseAlarm):
    radarList = []
    
    for i in range(len(radarPositions)):
        radarList.append(RadarCircularPattern(position=radarPositions[i], phase=radarPhases[i], angular_rate=radarAngularRates[i], outputPower=radarOutputPowers[i], transmitGain = radarTransmitGains[i], recieveGain = radarRecieveGains[i], wavelength = radarWavelength, pulseWidth = radarPulseWidth, systemTemperature = radarSystemTemperature, probabilityOfFalseAlarm=radarProbabilityOfFalseAlarm))
        
    return radarList


def create_agent_list(agentInitialStates, numRadar, agentSensingRange, agentPowerMeasurementStdDev, agentAngleMeasurementStdDev, agentELINTAnteneaGain, agentELINTSystemLoss, radarWavelength, radarCrossSection):
    agentList =[]
    for i in range(len(agentInitialStates)):
        agentList.append(Agent(initialPosition=agentInitialStates[i], numRadar = numRadar, sensingRange = agentSensingRange, powerMeasurementStdDev=agentPowerMeasurementStdDev,angleMeasurementStdDev=agentAngleMeasurementStdDev, elintAntenneaGain=agentELINTAnteneaGain, elintSystemLoss=agentELINTSystemLoss, emittorWavelength=radarWavelength, radarCrossSection = radarCrossSection, agentId = i))
    
    return agentList



    
