import numpy as np
from scipy.optimize import least_squares

from sklearn.linear_model import RANSACRegressor
from sklearn import base

class NonlinearEstimator(base):
    def __init__(self,position):
        self.estimated_emmiter_location = None

    def measurement_residual(self, emmiter_location, aoa_measurements, measurement_locations):
        return np.array([self.measurement_model(emmiter_location[0], emmiter_location[1], loc[0], loc[1]) for loc in measurement_locations]).reshape((len(aoa_measurements),)) - np.array(aoa_measurements).reshape((len(aoa_measurements),))

    def fit(self, X, y):
        x0 = np.array([500,500])
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(X,y))
        self.estimated_location = sol.x
        return self

    def score(self, X,y):
        return np.norm(self.measurement_residual(self.estimated_emmiter_location, y, X))
        

    def predict(self, X):
        return [self.measurement_model(self.estimated_emmiter_location[0], self.estimated_emmiter_location[1], loc[0], loc[1]) for loc in X]

    def measurement_model(self, xem, yem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)]])
    
    def stack_measurement_jacobian(self, emitter_location, aoa_measurements, measurement_locations):
        return np.array([self.measurement_jacobian(emitter_location[0], emitter_location[1], loc[0], loc[1]) for loc in measurement_locations])

    def measurement_jacobian(self, xem, yem, x, y):
        d_h_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        return np.array([d_h_d_x_emmitter, d_h_d_y_emmitter])
    
    # def get_params(self, deep):
    #     return {"position": self.estimated_emmiter_location}
    

class MultipleRadarLocationEstimator:
    def __init__(self):
        self.ekf_list = []


        self.measurement_locations = []
        self.aoa_measurement_values = []
        # self.regressionModel = NonlinearEstimator()
        self.ransacRegressor = RANSACRegressor(NonlinearEstimator(0), min_samples=2)
    


    
    def add_measurement(self, measurement_location, aoa_value, measurement_var):
        self.measurement_locations.append(measurement_location)
        self.aoa_measurement_values.append(aoa_value)
        if len(self.aoa_measurement_values) > 2:
            self.ransacRegressor.fit(np.array(self.measurement_locations), np.array(self.aoa_measurement_values).reshape((-1,1)))
            print(self.ransacRegressor.get_params())
            print()
