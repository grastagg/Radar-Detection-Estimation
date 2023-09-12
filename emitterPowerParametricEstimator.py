import numpy as np



class EmitterPowerParametricEstimator:
    def __init__(self):
        self.estimated_radar_power = None
        
    def basis_function(self, radial_distance):
        return 1/radial_distance**2

    def create_basis_function_matrix(self, measurment_locations, center):
        A = np.zeros((len(measurment_locations),1))
        for i,meas_loc in enumerate(measurment_locations):
            A[i] = self.basis_function(np.linalg.norm(meas_loc - center))
        return A

    
    def fit_parametric_estimator(self, measurment_power_values, measurement_locations, estimated_radar_location):
        estimated_radar_location = estimated_radar_location.reshape((1,2))
        A = self.create_basis_function_matrix(measurement_locations, estimated_radar_location)
        self.estimated_radar_power = np.linalg.lstsq(A, measurment_power_values, rcond=None)[0]

        # print("estimated radar power", self.estimated_radar_power)
    
    def prediction(self, prediction_locations, estimated_radar_location):
        A = self.create_basis_function_matrix(prediction_locations, estimated_radar_location)
        return A*self.estimated_radar_power
        
        