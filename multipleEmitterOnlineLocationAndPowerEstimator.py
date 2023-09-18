
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import block_diag
from matplotlib import colors

from sklearn.linear_model import RANSACRegressor

class MultipleEmitterOnlineLocationAndPowerEstimator:
    def __init__(self, sensing_range, angle_measurement_std_dev, measurement_cov, X_test):
        self.ekf_list = []
        self.sensing_range = sensing_range
        self.angle_measurement_std_dev = angle_measurement_std_dev
        self.measurement_cov = measurement_cov
        self.X_test = X_test

        self.estimated_emmiter_params = [] 
        self.estimated_emmiter_params_covariances = [] 

        self.outlier_indicies = np.array([], dtype=int)


        self.measurement_locations = []
        self.measurement_values = []

        self.inlier_mask = None
        self.group_lists = []

        self.mahalonobis_distance_inlier_threshold = 3
        self.mal_dist_opt_point = None
    
    def delete_lowest_probability_model(self):
        remove_indicies = []
        min_num_measurements = 100000
        for i,g_list in enumerate(self.group_lists):
            num_measurements = len(g_list)
            if num_measurements < min_num_measurements:
                min_num_measurements = num_measurements
        
        if min_num_measurements < 5:
            for i,g_list in enumerate(self.group_lists):
                num_measurements = len(g_list)
                if num_measurements == min_num_measurements:
                    remove_indicies.append(i)
                    print("deleting model", i)
                    for ind in g_list:
                        self.outlier_indicies = np.append(self.outlier_indicies, ind)

            self.estimated_emmiter_params = [i for j,i in enumerate(self.estimated_emmiter_params) if j not in remove_indicies]
            self.estimated_emmiter_params_covariances = [i for j,i in enumerate(self.estimated_emmiter_params_covariances) if j not in remove_indicies]
            self.group_lists = [i for j,i in enumerate(self.group_lists) if j not in remove_indicies]
    

    def fit_ransac_model(self, measurement_locations, measurements, original_index):
        if len(measurements) > 4:
            try:
                def loss(y_true, y_pred, X, estimator):
                    mean = estimator.get_estimate_emmitor_params()
                    cov = estimator.compute_emmitor_estimate_covariance(X, y_true)
                    return np.array([self.mahalonobis_distance(y, X[i], mean, cov) for i,y in enumerate(y_true)])
                regressionModel = NonlinearEstimator(self.measurement_cov)
                # ransacRegressor = RANSACRegressor(regressionModel, min_samples=2,random_state=0,loss = 'squared_error')#,residual_threshold=.09)
                ransacRegressor = RANSACRegressor(regressionModel, min_samples=2,random_state=0,loss = loss, residual_threshold=.5)#,residual_threshold=.09)
                ransacRegressor.fit(np.array(measurement_locations), np.array(measurements))
                self.estimated_emmiter_params.append(ransacRegressor.estimator_.get_estimate_emmitor_params())
                self.group_lists.append(original_index[ransacRegressor.inlier_mask_== True])
                self.estimated_emmiter_params_covariances.append(ransacRegressor.estimator_.compute_emmitor_estimate_covariance(np.array(self.measurement_locations)[self.group_lists[-1]], np.array(self.measurement_values)[self.group_lists[-1]]))
                for ind in original_index[ransacRegressor.inlier_mask_== True]:
                    self.outlier_indicies = np.delete(self.outlier_indicies, np.argwhere(self.outlier_indicies == ind))
                self.fit_ransac_model(measurement_locations[ransacRegressor.inlier_mask_== False],measurements[ransacRegressor.inlier_mask_== False], original_index[ransacRegressor.inlier_mask_== False])
            except:
                pass
    
    def minimized_angle(self, angle):
        while angle < -np.pi:
            angle += 2*np.pi
        
        while angle > np.pi:
            angle -= 2*np.pi
        return angle

    def ekf_update(self, measurement_pos, measurement_value, measurement_cov, x_prev, sigma_prev):
        xem = x_prev[0]
        yem = x_prev[1]
        pem = x_prev[2]
        x = measurement_pos[0]
        y = measurement_pos[1]

        H = self.measurement_jacobian(xem, yem, pem, x, y)
        K = sigma_prev @ H.T @ np.linalg.inv(H@sigma_prev@H.T + measurement_cov)
        zHat = self.measurement_model(xem, yem, pem, x, y)
        inovation = np.zeros_like(zHat)
        inovation[0] = self.minimized_angle(measurement_value[0] - zHat[0][0])
        inovation[1] = measurement_value[1] - zHat[1][0]

        xHat = x_prev + K@(inovation).reshape((-1,))
        sigmaHat = (np.eye(3) - K@H)@sigma_prev
        return xHat, sigmaHat

    def mahalonobis_distance(self, measurement_value, measurement_location, estimated_emitter_params, emitter_params_covariance):
        z_hat = self.measurement_model(estimated_emitter_params[0], estimated_emitter_params[1], estimated_emitter_params[2], measurement_location[0], measurement_location[1])
        measurement_jacobian = self.measurement_jacobian(estimated_emitter_params[0], estimated_emitter_params[1], estimated_emitter_params[2], measurement_location[0], measurement_location[1])

        z_hat_cov = measurement_jacobian @ emitter_params_covariance @ measurement_jacobian.T

        aoa_value = measurement_value[0]
        power_value = measurement_value[1]
        
        diff_square_aoa = min((aoa_value - z_hat[0][0])**2, min((aoa_value - (z_hat[0][0] + 2*np.pi))**2, (aoa_value - (z_hat[0][0] - 2 * np.pi))**2))
        diff_square_power = (power_value - z_hat[1][0])**2
        diff_square = np.array([[diff_square_aoa],[diff_square_power]])
        return np.sqrt(diff_square.T @ np.linalg.inv(z_hat_cov) @ diff_square)[0][0]


    def get_linear_model_from_aoa_measurement(self, aoa_value, measurement_location):
        m = np.tan(aoa_value)
        b = measurement_location[1] - m * measurement_location[0]
        return m,b

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

        
        opt = np.linalg.solve(C, D)
        x = opt[0:2]
        self.mal_dist_opt_point = x
        
        mahalonobis_distance_squared = ((x-mean).T@inv_cov@(x-mean))[0][0]
        return np.sqrt(mahalonobis_distance_squared)
    
    def measurement_model(self, xem, yem, pem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)], [pem/((xem-x)**2 + (yem-y)**2)]])
    def measurement_jacobian(self, xem, yem, pem, x, y):
        d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        d_h1_d_p_emmitter = 0

        d_h2_d_x_emmitter = -(2*pem*(xem-x))/((xem-x)**2+(yem-y)**2)**2
        d_h2_d_y_emmitter = -(2*pem*(yem-y))/((yem-y)**2+(xem-x)**2)**2
        d_h2_d_p_emmitter = 1/((yem-y)**2+(xem-x)**2)

        return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    def add_measurement(self, measurement_location, measurement_value):
        self.measurement_locations.append(measurement_location)
        self.measurement_values.append(measurement_value)
        if len(self.measurement_values) > 2:
            updated_using_ekf = False
            minimum_mal_dist = 10000
            minimum_mal_dist_index = -1
            for i,emitter_param in enumerate(self.estimated_emmiter_params):
                measurement_mahalonobis_distance = self.mahalonobis_distance(measurement_value, measurement_location, emitter_param, self.estimated_emmiter_params_covariances[i])
                measurement_mahalonobis_distance_in_x_y = self.mahalonobis_distance_in_x_y_space(measurement_value[0], measurement_location, emitter_param[0:2], self.estimated_emmiter_params_covariances[i][0:2,0:2])
                combined_mahalanobis_distance = measurement_mahalonobis_distance_in_x_y + measurement_mahalonobis_distance
                print("measurement_mahalonobis_distance",measurement_mahalonobis_distance)
                print("measurement_mahalonobis_distance_in_x_y",measurement_mahalonobis_distance_in_x_y)
                # if measurement_mahalonobis_distance < minimum_mal_dist:
                #     minimum_mal_dist  = measurement_mahalonobis_distance
                #     minimum_mal_dist_index = i
                # if measurement_mahalonobis_distance_in_x_y < minimum_mal_dist:
                #     minimum_mal_dist  = measurement_mahalonobis_distance_in_x_y
                #     minimum_mal_dist_index = i
                if combined_mahalanobis_distance < minimum_mal_dist:
                    minimum_mal_dist  = combined_mahalanobis_distance
                    minimum_mal_dist_index = i

            if minimum_mal_dist < self.mahalonobis_distance_inlier_threshold:
                
                estimated_emmiter_param,  estimated_emmiter_params_covariances = self.ekf_update(measurement_location, measurement_value, self.measurement_cov, self.estimated_emmiter_params[minimum_mal_dist_index], self.estimated_emmiter_params_covariances[minimum_mal_dist_index])
                if estimated_emmiter_param[2] < 0:
                    print("estimated power level too small")
                    updated_using_ekf = False
                else:
                    self.estimated_emmiter_params[minimum_mal_dist_index] = estimated_emmiter_param
                    self.estimated_emmiter_params_covariances[minimum_mal_dist_index] = estimated_emmiter_params_covariances
                    self.group_lists[minimum_mal_dist_index] = np.append(self.group_lists[minimum_mal_dist_index],(len(self.measurement_values)-1))
                    updated_using_ekf = True

            if not updated_using_ekf:
                self.outlier_indicies = np.append(self.outlier_indicies, len(self.measurement_values)-1)
                self.fit_ransac_model(np.array(self.measurement_locations)[self.outlier_indicies], np.array(self.measurement_values)[self.outlier_indicies], self.outlier_indicies)
            self.delete_lowest_probability_model()
            print("ransac prediction", self.estimated_emmiter_params)
            print("group lists", self.group_lists)
            print("outliers", self.outlier_indicies)
            print()
        elif len(self.measurement_values) == 2:
            self.outlier_indicies = np.append(self.outlier_indicies, 1)
        else:
            self.outlier_indicies = np.append(self.outlier_indicies, 0)
    
    def power_basis_function(self, radial_distance):
        return 1/radial_distance**2

    def create_power_basis_function_matrix(self, measurment_locations, center):
        A = np.zeros((len(measurment_locations),1))
        for i,meas_loc in enumerate(measurment_locations):
            A[i] = self.power_basis_function(np.linalg.norm(meas_loc - center))
        return A

    def prediction(self, prediction_locations, estimated_emittor_params):
        A = self.create_power_basis_function_matrix(prediction_locations, estimated_emittor_params[0:2])
        return A*estimated_emittor_params[2]

    def combined_power_prediction(self, prediction_locations):
        output = np.zeros((len(prediction_locations),1))
        for i, emmitor_params in enumerate(self.estimated_emmiter_params):
            output += self.prediction(prediction_locations, emmitor_params)
        return output
            
    def plot(self, ax):
        color_list = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
        c = None

        for i,angle_indicies in enumerate([self.outlier_indicies]):
            if angle_indicies.size > 0:
                self.plot_angle_of_arrival_measurements(ax, 'r', np.array(self.measurement_locations)[angle_indicies], np.array(self.measurement_values)[:,0][angle_indicies])
        if len(self.estimated_emmiter_params) > 0:
            for i,angle_indicies in enumerate(self.group_lists):
                self.plot_angle_of_arrival_measurements(ax, color_list[i], np.array(self.measurement_locations)[angle_indicies], np.array(self.measurement_values)[:,0][angle_indicies])
            

            for i,estimated_emmiter_params in enumerate(self.estimated_emmiter_params):
                ax.scatter(estimated_emmiter_params[0], estimated_emmiter_params[1], marker='x', c = 'm')
                # c = self.plot_esimate_1_sigma_bounds(ax, estimated_emmiter_params, self.estimated_emmiter_params_covariances[i])
        # if self.mal_dist_opt_point is not None:
        #     ax.scatter(self.mal_dist_opt_point[0], self.mal_dist_opt_point[1])
        c = self.plot_power(ax)



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

    def plot_esimate_1_sigma_bounds(self, ax, estimated_emmiter_params, estimated_emmiter_params_cov):
        invCovariance = np.linalg.inv(estimated_emmiter_params_cov[0:2,0:2])
        
        x = np.linspace(0, 1200, num=100)
        y = np.linspace(0, 1200, num=100)
        X, Y = np.meshgrid(x,y)
        malhanobisDist = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                malhanobisDist[i,j] = (np.array([[X[i,j], Y[i,j]]]) - np.array([[estimated_emmiter_params[0], estimated_emmiter_params[1]]])) @ invCovariance @(np.array([[X[i,j], Y[i,j]]]) - np.array([[estimated_emmiter_params[0], estimated_emmiter_params[1]]])).T

        c = ax.contourf(X, Y, malhanobisDist, cmap='viridis',levels = [ 0,1,2,3])
        return c

    def plot_power(self, ax):
        if len(self.estimated_emmiter_params) > 0:
            X_test = np.array(self.X_test)
            predictive_mean = self.combined_power_prediction(self.X_test)
            num_test_points = int(np.sqrt(len(X_test)))


            # c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.predictive_mean).reshape((self.num_test_points,self.num_test_points)),vmin = -5, vmax =.5)
            Z = predictive_mean.reshape((num_test_points,num_test_points))
            c = ax.pcolormesh(X_test[:,0].reshape((num_test_points,num_test_points)), X_test[:,1].reshape((num_test_points,num_test_points)), Z, norm=colors.SymLogNorm(linthresh = 0.0001, vmin = 0,vmax = 2, base=10))
            # c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.predictive_mean).reshape((self.num_test_points,self.num_test_points)))
            return c
        else:
            return None

class NonlinearEstimator():
    def __init__(self, measurement_cov):
        self.estimated_emmiter_params = None
        self.estimated_emmiter_params_cov = None
        self.measurement_covariance = measurement_cov

    def measurement_residual(self, emitter_params, measurements, measurement_locations):
        return np.array([self.measurement_model(emitter_params[0], emitter_params[1], emitter_params[2], loc[0], loc[1]) for loc in measurement_locations]).reshape((-1,)) - np.array(measurements).reshape((-1,))

    def fit(self, X, y):
        # x0 = np.array([500,500,100])
        x0 = np.array([600,600,1000])
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(y,X), bounds=([0,0,10],[2000,2000,np.inf]))
        # sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(y,X), bounds=([-np.inf,-np.inf,10],[np.inf,np.inf,np.inf]))
        self.estimated_emmiter_params = sol.x
        # self.estimated_emmiter_location_covariances = self.compute_emmitor_estimate_covariance(X, y, self.measurement_variance)

    def score(self, X,y):
        # print("X",X)
        # print("y",y)
        # print("score",np.linalg.norm(self.measurement_residual(self.estimated_emmiter_params, y, X))**2)
        return 1/np.linalg.norm(self.measurement_residual(self.estimated_emmiter_params, y, X))**2
    
    def predict(self, X):
        return np.array([self.measurement_model(self.estimated_emmiter_params[0], self.estimated_emmiter_params[1], self.estimated_emmiter_params[2], loc[0], loc[1]) for loc in X]).reshape((-1,2))

    def measurement_model(self, xem, yem, pem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)], [pem/((xem-x)**2 + (yem-y)**2)]])
    
    def stack_measurement_jacobian(self, emitter_params, measurements, measurement_locations):
        return np.array([self.measurement_jacobian(emitter_params[0], emitter_params[1], emitter_params[2], loc[0], loc[1]) for loc in measurement_locations]).reshape((2*len(measurement_locations),3))

    def measurement_jacobian(self, xem, yem, pem, x, y):
        d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        d_h1_d_p_emmitter = 0

        d_h2_d_x_emmitter = -(2*pem*(xem-x))/((xem-x)**2+(yem-y)**2)**2
        d_h2_d_y_emmitter = -(2*pem*(yem-y))/((yem-y)**2+(xem-x)**2)**2
        d_h2_d_p_emmitter = 1/((yem-y)**2+(xem-x)**2)

        return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_estimate_emmitor_params(self):
        return self.estimated_emmiter_params
    
    def compute_emmitor_estimate_covariance(self, X, y):
        jacobians = self.stack_measurement_jacobian(self.estimated_emmiter_params, y, X)
        combined_measurment_cov = block_diag(*[self.measurement_covariance for i in X])
        emitter_params_cov = np.linalg.inv(jacobians.T@np.linalg.inv(combined_measurment_cov)@jacobians)
        return emitter_params_cov

    def get_params(self, deep=False):
        # return {"position": self.estimated_emmiter_location}
        return {"measurement_cov": self.measurement_covariance}
    

    
