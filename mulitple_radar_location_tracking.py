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
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(X,y), bounds=([0,0],[2000,2000]))
        self.estimated_emmiter_location = sol.x

    def score(self, X,y):
        return np.linalg.norm(self.measurement_residual(self.estimated_emmiter_location, X, y))**2
    
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
    def __init__(self, sensing_range, angle_measurement_std_dev):
        self.ekf_list = []
        self.sensing_range = sensing_range
        self.angle_measurement_std_dev = angle_measurement_std_dev

        self.estimated_emmiter_locations = [] 

        self.outlier_indicies = []


        self.measurement_locations = []
        self.aoa_measurement_values = []

        self.inlier_mask = None
        self.group_lists = []
    

    def fit_ransac_model(self, measurement_locations, aoa_values, original_index):
        print("in recursion")
        print("original index", original_index)
        print("measurement_locations",measurement_locations)
        print("aoa_values", aoa_values)
        if len(aoa_values) > 1:
            regressionModel = NonlinearEstimator()
            ransacRegressor = RANSACRegressor(regressionModel, min_samples=2,random_state=0,residual_threshold=.2)
            ransacRegressor.fit(np.array(measurement_locations), np.array(aoa_values).reshape((-1,1)))
            self.estimated_emmiter_locations.append(ransacRegressor.estimator_.get_estimate_emmitor_location())
            # print("ransac prediction", self.ransacRegressor.estimator_.get_estimate_emmitor_location())
            self.inlier_mask = ransacRegressor.inlier_mask_
            print("inliers", self.inlier_mask)
            self.group_lists.append(original_index[ransacRegressor.inlier_mask_== True])
            self.fit_ransac_model(measurement_locations[ransacRegressor.inlier_mask_== False],aoa_values[ransacRegressor.inlier_mask_== False], original_index[ransacRegressor.inlier_mask_== False])
        elif len(aoa_values) >= 1:
            self.outlier_indicies.append(original_index)
        else:
            pass


    
    def add_measurement(self, measurement_location, aoa_value, measurement_var):
        self.estimated_emmiter_locations = [] 
        self.group_lists = []
        self.outlier_indicies = []
        self.measurement_locations.append(measurement_location)
        self.aoa_measurement_values.append(aoa_value)
        if len(self.aoa_measurement_values) > 2:
            self.fit_ransac_model(np.array(self.measurement_locations), np.array(self.aoa_measurement_values), np.arange(len(self.aoa_measurement_values), dtype=int))
            # self.ransacRegressor.fit(np.array(self.measurement_locations), np.array(self.aoa_measurement_values).reshape((-1,1)))
            # self.estimated_emmiter_location = self.ransacRegressor.estimator_.get_estimate_emmitor_location()
            # print("ransac prediction", self.ransacRegressor.estimator_.get_estimate_emmitor_location())
            print("ransac prediction", self.estimated_emmiter_locations)
            print("group lists", self.group_lists)
            # self.inlier_mask = self.ransacRegressor.inlier_mask_
            # print("number of inliers", self.ransacRegressor.inlier_mask_)
        elif len(self.aoa_measurement_values) == 2:
            self.outlier_indicies.append([0,1])
        else:
            self.outlier_indicies.append([0])
            
    def plot(self, ax):
        color_list = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

        for i,angle_indicies in enumerate(self.outlier_indicies):
            self.plot_angle_of_arrival_measurements(ax, 'r', np.array(self.measurement_locations)[angle_indicies], np.array(self.aoa_measurement_values)[angle_indicies])
        if len(self.estimated_emmiter_locations) > 0:
            for i,angle_indicies in enumerate(self.group_lists):
                self.plot_angle_of_arrival_measurements(ax, color_list[i], np.array(self.measurement_locations)[angle_indicies], np.array(self.aoa_measurement_values)[angle_indicies])
            

            for estimated_emmiter_location in self.estimated_emmiter_locations:
                ax.scatter(estimated_emmiter_location[0], estimated_emmiter_location[1], marker='x', c = 'm')
            
    def plot_angle_of_arrival_measurements(self, ax, color, measurement_locations, measurement_angle_of_arrival_values):
        for i,angle in enumerate(measurement_angle_of_arrival_values):
            start_x = measurement_locations[i][0]
            start_y = measurement_locations[i][1]
            end_x = start_x + self.sensing_range * np.cos(angle)
            end_y = start_y + self.sensing_range * np.sin(angle)
            ax.plot([start_x,end_x],[start_y,end_y], c=color)
            end_x = start_x + self.sensing_range * np.cos(angle+self.angle_measurement_std_dev)
            end_y = start_y + self.sensing_range * np.sin(angle+self.angle_measurement_std_dev)
            ax.plot([start_x,end_x],[start_y,end_y],linestyle = '--',c = color)
            end_x = start_x + self.sensing_range * np.cos(angle-self.angle_measurement_std_dev)
            end_y = start_y + self.sensing_range * np.sin(angle-self.angle_measurement_std_dev)
            ax.plot([start_x,end_x],[start_y,end_y],linestyle = '--',c=color)