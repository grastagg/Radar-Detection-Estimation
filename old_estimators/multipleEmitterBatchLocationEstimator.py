import numpy as np
from scipy.optimize import least_squares

from sklearn.linear_model import RANSACRegressor
# from sklearn import base

class NonlinearEstimator():
    def __init__(self, measurement_variance = (4*np.pi/180.0)**2):
        self.estimated_emmiter_location =  None
        self.estimated_emmiter_location_covariances = None
        self.measurement_variance = measurement_variance

    def measurement_residual(self, emmiter_location, measurement_locations, aoa_measurements):
        return np.array([self.measurement_model(emmiter_location[0], emmiter_location[1], loc[0], loc[1]) for loc in measurement_locations]).reshape((len(aoa_measurements),)) - np.array(aoa_measurements).reshape((len(aoa_measurements),))

    def fit(self, X, y):
        x0 = np.array([500,500])
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(X,y), bounds=([0,0],[2000,2000]))
        self.estimated_emmiter_location = sol.x
        # self.estimated_emmiter_location_covariances = self.compute_emmitor_estimate_covariance(X, y, self.measurement_variance)

    def score(self, X,y):
        return np.linalg.norm(self.measurement_residual(self.estimated_emmiter_location, X, y))**2
    
    def predict(self, X):
        return np.array([self.measurement_model(self.estimated_emmiter_location[0], self.estimated_emmiter_location[1], loc[0], loc[1]) for loc in X]).reshape((-1,1))

    def measurement_model(self, xem, yem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)]])
    
    def stack_measurement_jacobian(self, emitter_location,measurement_locations, aoa_measurement_values):
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
    
    def compute_emmitor_estimate_covariance(self, X, y, measurement_variance):
        jacobians = self.stack_measurement_jacobian(self.estimated_emmiter_location, X, y)
        emmiter_location_cov = measurement_variance * np.linalg.inv(jacobians.T@jacobians)
        return emmiter_location_cov

    def get_params(self, deep=False):
        # return {"position": self.estimated_emmiter_location}
        return {}
    

    

class MultipleEmitterBatchLocationEstimator:
    def __init__(self, sensing_range, angle_measurement_std_dev):
        self.ekf_list = []
        self.sensing_range = sensing_range
        self.angle_measurement_std_dev = angle_measurement_std_dev

        self.estimated_emmiter_locations = [] 
        self.estimated_emmiter_location_covariances = [] 

        self.outlier_indicies = np.array([], dtype=int)


        self.measurement_locations = []
        self.aoa_measurement_values = []

        self.inlier_mask = None
        self.group_lists = []

        self.mahalonobis_distance_inlier_threshold = 4
    

    def fit_ransac_model(self, measurement_locations, aoa_values, original_index):
        if len(aoa_values) > 2:
            regressionModel = NonlinearEstimator(self.angle_measurement_std_dev**2)
            ransacRegressor = RANSACRegressor(regressionModel, min_samples=2,random_state=0,residual_threshold=.01)
            ransacRegressor.fit(np.array(measurement_locations), np.array(aoa_values).reshape((-1,1)))
            self.estimated_emmiter_locations.append(ransacRegressor.estimator_.get_estimate_emmitor_location())
            self.group_lists.append(original_index[ransacRegressor.inlier_mask_== True])
            self.estimated_emmiter_location_covariances.append(ransacRegressor.estimator_.compute_emmitor_estimate_covariance(np.array(self.measurement_locations)[self.group_lists[-1]], np.array(self.aoa_measurement_values)[self.group_lists[-1]], self.angle_measurement_std_dev**2))
            for ind in original_index[ransacRegressor.inlier_mask_== True]:
                self.outlier_indicies = np.delete(self.outlier_indicies, np.argwhere(self.outlier_indicies == ind))
                

    def ekf_update(self, aoa_measurement_pos, aoa_measurement_value, aoa_measurement_cov, x_prev, sigma_prev):
        xem = x_prev[0]
        yem = x_prev[1]
        x = aoa_measurement_pos[0]
        y = aoa_measurement_pos[1]

        H = self.measurement_jacobian(xem, yem, x, y)
        K = sigma_prev @ H.T @ np.linalg.inv(H@sigma_prev@H.T + np.array([[aoa_measurement_cov]]))
        xHat = x_prev + (K@(aoa_measurement_value - self.measurement_model(xem, yem, x, y))).reshape((-1,))
        sigmaHat = (np.eye(2) - K@H)@sigma_prev
        return xHat, sigmaHat

    def mahalonobis_distance(self, aoa_value, measurement_location, estimated_emitter_location, emitter_location_covariance):
        z_hat = self.measurement_model(estimated_emitter_location[0], estimated_emitter_location[1], measurement_location[0], measurement_location[1])
        measurement_jacobian = self.measurement_jacobian(estimated_emitter_location[0], estimated_emitter_location[1], measurement_location[0], measurement_location[1])

        z_hat_cov = measurement_jacobian @ emitter_location_covariance @ measurement_jacobian.T
        
        diff_square = min((aoa_value - z_hat)**2, min((aoa_value - (z_hat + 2*np.pi))**2, (aoa_value - (z_hat - 2 * np.pi))**2))
        return np.sqrt(diff_square * (1/ z_hat_cov))[0][0]
    
    def get_linear_model_from_aoa_measurement(self, aoa_value, measurement_location):
        m = np.tan(aoa_value)
        b = measurement_location[1] - m * measurement_location[0]
        return m,b
    
    def mahalonobis_distance_in_x_y_space(self, aoa_value, measurement_location, mean, covariance):
        mean = mean.reshape((2,1))
        m,b = self.get_linear_model_from_aoa_measurement(aoa_value, measurement_location)
        
        A = np.array([m,-1])
        C = np.zeros((3,3))
        inv_cov = np.linalg.inv(covariance)
        C[0:2,0:2] = 2*inv_cov
        C[0:2,2] = A.T
        C[2,0:2] = A
        D = np.zeros((3,1))
        D[0:2,0] = 2*mean.T@inv_cov
        D[2,0] = -b
        # D = np.array([[-mean[0]],[-mean[1]],[-b]])

        
        opt = np.linalg.solve(C, D)
        x = opt[0:2]
        
        mahalonobis_distance_squared = ((x-mean).T@inv_cov@(x-mean))[0][0]
        return np.sqrt(mahalonobis_distance_squared)
    
    def measurement_model(self, xem, yem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)]])

    def measurement_jacobian(self, xem, yem, x, y):
        d_h_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        return np.array([[d_h_d_x_emmitter, d_h_d_y_emmitter]])

    def add_measurement(self, measurement_location, aoa_value, measurement_var):
        # self.estimated_emmiter_locations = [] 
        # self.estimated_emmiter_location_covariances = [] 
        # self.group_lists = []
        # self.outlier_indicies = []
        self.measurement_locations.append(measurement_location)
        self.aoa_measurement_values.append(aoa_value)
        if len(self.aoa_measurement_values) > 2:
            updated_using_ekf = False
            minimum_mal_dist = 10000
            minimum_mal_dist_index = -1
            for i,emitter_location in enumerate(self.estimated_emmiter_locations):
                # measurement_mahalonobis_distance = self.mahalonobis_distance(aoa_value, self.measurement_model(emitter_location[0], emitter_location[1], measurement_location[0], measurement_location[1])[0][0], measurement_var)
                measurement_mahalonobis_distance = self.mahalonobis_distance(aoa_value, measurement_location, emitter_location, self.estimated_emmiter_location_covariances[i])
                measurement_mahalonobis_distance_in_x_y = self.mahalonobis_distance_in_x_y_space(aoa_value, measurement_location, emitter_location, self.estimated_emmiter_location_covariances[i])
                combined_mahalanobis_distance = measurement_mahalonobis_distance_in_x_y + measurement_mahalonobis_distance
                print("measurement_mahalonobis_distance",measurement_mahalonobis_distance)
                print("measurement_mahalonobis_distance_in_x_y",measurement_mahalonobis_distance_in_x_y)
                if measurement_mahalonobis_distance < minimum_mal_dist:
                    minimum_mal_dist  = measurement_mahalonobis_distance
                    minimum_mal_dist_index = i
                # if measurement_mahalonobis_distance_in_x_y < minimum_mal_dist:
                #     minimum_mal_dist  = measurement_mahalonobis_distance_in_x_y
                #     minimum_mal_dist_index = i
                # if combined_mahalanobis_distance < minimum_mal_dist:
                #     minimum_mal_dist  = combined_mahalanobis_distance
                #     minimum_mal_dist_index = i

            if minimum_mal_dist < self.mahalonobis_distance_inlier_threshold:

                self.estimated_emmiter_locations[minimum_mal_dist_index],  self.estimated_emmiter_location_covariances[minimum_mal_dist_index] = self.ekf_update(measurement_location, aoa_value, measurement_var, self.estimated_emmiter_locations[minimum_mal_dist_index], self.estimated_emmiter_location_covariances[minimum_mal_dist_index])
                self.group_lists[minimum_mal_dist_index] = np.append(self.group_lists[minimum_mal_dist_index],(len(self.aoa_measurement_values)-1))

            else:
                self.outlier_indicies = np.append(self.outlier_indicies, len(self.aoa_measurement_values)-1)
                # try:
                self.fit_ransac_model(np.array(self.measurement_locations)[self.outlier_indicies], np.array(self.aoa_measurement_values)[self.outlier_indicies], self.outlier_indicies)
                # except:
                #     print("no model found")
            print("ransac prediction", self.estimated_emmiter_locations)
            print("group lists", self.group_lists)
            print("outliers", self.outlier_indicies)
        elif len(self.aoa_measurement_values) == 2:
            self.outlier_indicies = np.append(self.outlier_indicies, 1)
        else:
            self.outlier_indicies = np.append(self.outlier_indicies, 0)
            
    def plot(self, ax):
        color_list = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

        for i,angle_indicies in enumerate([self.outlier_indicies]):
            self.plot_angle_of_arrival_measurements(ax, 'r', np.array(self.measurement_locations)[angle_indicies], np.array(self.aoa_measurement_values)[angle_indicies])
        if len(self.estimated_emmiter_locations) > 0:
            for i,angle_indicies in enumerate(self.group_lists):
                self.plot_angle_of_arrival_measurements(ax, color_list[i], np.array(self.measurement_locations)[angle_indicies], np.array(self.aoa_measurement_values)[angle_indicies])
            

            for i,estimated_emmiter_location in enumerate(self.estimated_emmiter_locations):
                ax.scatter(estimated_emmiter_location[0], estimated_emmiter_location[1], marker='x', c = 'm')
                c = self.plot_esimate_1_sigma_bounds(ax, estimated_emmiter_location, self.estimated_emmiter_location_covariances[i])
            return c
            
    def plot_angle_of_arrival_measurements(self, ax, color, measurement_locations, measurement_angle_of_arrival_values):
        line_width = 1
        for i,angle in enumerate(measurement_angle_of_arrival_values):
            start_x = measurement_locations[i][0]
            start_y = measurement_locations[i][1]
            end_x = start_x + self.sensing_range * np.cos(angle)
            end_y = start_y + self.sensing_range * np.sin(angle)
            ax.plot([start_x,end_x],[start_y,end_y], c=color, linewidth = line_width)
            end_x = start_x + self.sensing_range * np.cos(angle+self.angle_measurement_std_dev)
            end_y = start_y + self.sensing_range * np.sin(angle+self.angle_measurement_std_dev)
            ax.plot([start_x,end_x],[start_y,end_y],linestyle = '--',c = color, linewidth = line_width)
            end_x = start_x + self.sensing_range * np.cos(angle-self.angle_measurement_std_dev)
            end_y = start_y + self.sensing_range * np.sin(angle-self.angle_measurement_std_dev)
            ax.plot([start_x,end_x],[start_y,end_y],linestyle = '--',c=color, linewidth = line_width)

    def plot_esimate_1_sigma_bounds(self, ax, estimated_emmiter_location, estimated_emmiter_location_cov):
        invCovariance = np.linalg.inv(estimated_emmiter_location_cov)
        
        # x = np.linspace(mean_1-3*sigma_1, mean_1+3*sigma_1, num=100)
        # y = np.linspace(mean_2-3*sigma_2, mean_2+3*sigma_2, num=100)
        x = np.linspace(0, 1200, num=100)
        y = np.linspace(0, 1200, num=100)
        X, Y = np.meshgrid(x,y)
        malhanobisDist = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                malhanobisDist[i,j] = (np.array([[X[i,j], Y[i,j]]]) - np.array([[estimated_emmiter_location[0], estimated_emmiter_location[1]]])) @ invCovariance @(np.array([[X[i,j], Y[i,j]]]) - np.array([[estimated_emmiter_location[0], estimated_emmiter_location[1]]])).T

        c = ax.contourf(X, Y, malhanobisDist, cmap='viridis',levels = [ 0,1,2,3])
        return c