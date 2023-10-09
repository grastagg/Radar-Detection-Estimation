import numpy as np



class ProbabilityOfDetectionMap():
    def __init__(self):
        pass

    
    
    def probability_of_detection(probability_of_false_alarm, snr):
        return np.exp((np.ln(probability_of_false_alarm))/(snr + 1))
    
    
    