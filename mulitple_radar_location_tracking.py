import numpy as np

class multipleRadarLocationEstimator:
    def __init__(self):
        self.ekf_list = []
        
        self.unused_measurement_aoa_values = []
        self.unused_measurement_locations = []
        
    
    
    
    def incorperate_new_measurement(self, aoa_value, measurement_location, measurement_var):
        pass