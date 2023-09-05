import numpy as np
from scipy.optimize import least_squares

from sklearn.linear_model import RANSACRegressor
# from sklearn import base

class NonlinearEstimator():
    def __init__(self):
        self.estimated_emmiter_location =  None

    def measurement_residual(self, emmiter_location, measurement_locations, aoa_measurements):
        return np.array([self.measurement_model(emmiter_location[0], emmiter_location[1], loc[0], loc[1]) for loc in measurement_locations]).reshape((len(aoa_measurements),)) - np.array(aoa_measurements).reshape((len(aoa_measurements),))

    def fit(self, X, y):
        x0 = np.array([500,500])
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(X,y))
        self.estimated_emmiter_location = sol.x

    def score(self, X,y):
        return np.linalg.norm(self.measurement_residual(self.estimated_emmiter_location, X, y))
        

    def predict(self, X):
        return np.array([self.measurement_model(self.estimated_emmiter_location[0], self.estimated_emmiter_location[1], loc[0], loc[1]) for loc in X]).reshape((-1,1))

    def measurement_model(self, xem, yem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)]])
    
    def stack_measurement_jacobian(self, emitter_location,measurement_locations, aoa_measurements):
        return np.array([self.measurement_jacobian(emitter_location[0], emitter_location[1], loc[0], loc[1]) for loc in measurement_locations])

    def measurement_jacobian(self, xem, yem, x, y):
        d_h_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        return np.array([d_h_d_x_emmitter, d_h_d_y_emmitter])

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_estimate_emmitor_location(self):
        return self.estimated_emmiter_location

    def get_params(self, deep=False):
        # return {"position": self.estimated_emmiter_location}
        return {}
    

class MultipleRadarLocationEstimator:
    def __init__(self):
        self.ekf_list = []

        self.estimated_emmiter_location = None


        self.measurement_locations = []
        self.aoa_measurement_values = []
        self.regressionModel = NonlinearEstimator()
        self.ransacRegressor = RANSACRegressor(self.regressionModel, min_samples=2,random_state=0,residual_threshold=.5 )

        self.inlier_mask = None
    


    
    def add_measurement(self, measurement_location, aoa_value, measurement_var):
        self.measurement_locations.append(measurement_location)
        self.aoa_measurement_values.append(aoa_value)
        if len(self.aoa_measurement_values) > 2:
            self.ransacRegressor.fit(np.array(self.measurement_locations), np.array(self.aoa_measurement_values).reshape((-1,1)))
            self.estimated_emmiter_location = self.ransacRegressor.estimator_.get_estimate_emmitor_location()
            print("ransac prediction", self.ransacRegressor.estimator_.get_estimate_emmitor_location())
            self.inlier_mask = self.ransacRegressor.inlier_mask_
            # print("number of inliers", self.ransacRegressor.inlier_mask_)
            
    def plot(self, ax):
        if self.estimated_emmiter_location is not None:
            ax.scatter(self.estimated_emmiter_location[0], self.estimated_emmiter_location[1], marker='x', color = 'b')