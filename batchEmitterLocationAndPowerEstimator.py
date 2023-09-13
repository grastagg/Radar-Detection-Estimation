import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import block_diag

class BatchEmitterLocationAndPowerEstimator:
    def __init__(self, measurement_cov, ground_truth = None):
        self.ground_truth = ground_truth
        self.measurement_locations = []
        self.measurement_values = []
        # self.aoa_measurement_variances = []
        self.errorHistory = []
        self.estimated_emmiter_location = None
        self.estimated_emmiter_location_cov = None
        self.measurement_covariance = measurement_cov
        
    def add_measurement(self, measurement_location, aoa_value, power_val):
        self.measurement_locations.append(measurement_location)
        self.measurement_values.append([aoa_value, power_val])
        # self.aoa_measurement_variances.append(aoa_var)

        if len(self.measurement_values) >= 2:
            self.batch_estimation()
            # self.errorHistory.append(np.linalg.norm(self.estimated_emmiter_location.reshape((2,)) - self.ground_truth))



        

    def measurement_model(self, xem, yem, pem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)], [pem/((xem-x)**2 + (yem-y)**2)]])
    
    def measurement_residual(self, emitter_params, measurements, measurement_locations):
        return np.array([self.measurement_model(emitter_params[0], emitter_params[1], emitter_params[2], loc[0], loc[1]) for loc in measurement_locations]).reshape((-1,)) - np.array(measurements).reshape((-1,))

    def batch_estimation(self):
        x0 = [600,600,100]
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(self.measurement_values,self.measurement_locations))
        # print("sol",sol)
        self.estimated_emmiter_location = sol.x
        # print("residuals", self.measurement_residual(emmiter_location,self.aoa_measurement_values, self.measurement_locations))
        print("batch emmiter location", self.estimated_emmiter_location)
        jacobians = self.stack_measurement_jacobian(self.estimated_emmiter_location, None, self.measurement_locations)
        combined_measurment_cov = block_diag(*[self.measurement_covariance for i in self.measurement_locations])
        self.estimated_emmiter_location_cov = jacobians.T@combined_measurment_cov@jacobians

    
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
    
    def plot_esimate_1_sigma_bounds(self,ax):
        invCovariance = np.linalg.inv(self.estimated_emmiter_location_cov)
        
        # x = np.linspace(mean_1-3*sigma_1, mean_1+3*sigma_1, num=100)
        # y = np.linspace(mean_2-3*sigma_2, mean_2+3*sigma_2, num=100)
        x = np.linspace(0, 1200, num=100)
        y = np.linspace(0, 1200, num=100)
        X, Y = np.meshgrid(x,y)
        malhanobisDist = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                malhanobisDist[i,j] = (np.array([[X[i,j], Y[i,j]]]) - np.array([[self.estimated_emmiter_location[0], self.estimated_emmiter_location[1]]])) @ invCovariance @(np.array([[X[i,j], Y[i,j]]]) - np.array([[self.estimated_emmiter_location[0], self.estimated_emmiter_location[1]]])).T

        c = ax.contourf(X, Y, malhanobisDist, cmap='viridis',levels = [ 0,1,2,3])
        return c

    def plot(self, ax, plot_var = False):
        if self.estimated_emmiter_location is not None:
            ax.scatter(self.estimated_emmiter_location[0], self.estimated_emmiter_location[1], marker='x', color = 'r')
        
        if plot_var:
            self.plot_esimate_1_sigma_bounds(ax)

        

    