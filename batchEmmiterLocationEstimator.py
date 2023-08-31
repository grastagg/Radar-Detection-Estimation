import numpy as np
from scipy.optimize import least_squares

class BatchEmmiterLocationEstimator:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        self.measurement_locations = []
        self.aoa_measurement_values = []
        self.aoa_measurement_variances = []
        self.errorHistory = []
        self.estimated_emmiter_location = None
        self.estimated_emmiter_location_cov = None
        
    def add_measurement(self, measurement_location, aoa_value, aoa_var):
        self.measurement_locations.append(measurement_location)
        self.aoa_measurement_values.append(aoa_value)
        self.aoa_measurement_variances.append(aoa_var)
        if len(self.aoa_measurement_values) == 2:
            self.estimated_emmiter_location = self.compute_location_from_two_aoa_measurements(self.measurement_locations[0], self.aoa_measurement_values[0], self.measurement_locations[1], self.aoa_measurement_values[1])
            self.estimated_emmiter_location_cov = self.compute_location_covariance_from_two_aoa_measurements(self.measurement_locations[0], self.aoa_measurement_values[0], None, self.aoa_measurement_variances[0], self.measurement_locations[1], self.aoa_measurement_values[1], None, self.aoa_measurement_variances[1])
            self.errorHistory.append(np.linalg.norm(self.estimated_emmiter_location.reshape((2,)) - self.ground_truth))

        elif len(self.aoa_measurement_values) > 2:
            self.batch_estimation()
            self.errorHistory.append(np.linalg.norm(self.estimated_emmiter_location.reshape((2,)) - self.ground_truth))



    def sec(self,theta):
        return 1/np.cos(theta)
    def location_from_two_aoa_measurements_angle_jacobian(self,pos1, theta1, pos2, theta2):
        x1 = pos1[0]
        y1 = pos1[1]
        x2 = pos2[0]
        y2 = pos2[1]
        d_x_emitter_d_theta_1 = -(self.sec(theta1)**2*((y2-np.tan(theta2)*x2+np.tan(theta2)*x1)*np.tan(theta1)**2-2*np.tan(theta2)*y1*np.tan(theta1)+np.tan(theta2)**2*y1))/(np.tan(theta1)**2*(np.tan(theta1)-np.tan(theta2))**2) 
        d_x_emitter_d_theta_2 = ((y2-y1-np.tan(theta1)*x2+np.tan(theta1)*x1)*(1/np.cos(theta2))**2)/(np.tan(theta2)-np.tan(theta1))**2
        d_y_emitter_d_theta_1 = -(np.tan(theta2)*(y2-y1-np.tan(theta2)*x2+np.tan(theta2)*x1)*(1/np.cos(theta1))**2)/(np.tan(theta1)-np.tan(theta2))**2
        d_y_emitter_d_theta_2 = -(np.tan(theta2)*(y2-y1-np.tan(theta2)*x2+np.tan(theta2)*x1)*(1/np.cos(theta1))**2)/(np.tan(theta1)-np.tan(theta2))**2
        return np.array([[d_x_emitter_d_theta_1, d_x_emitter_d_theta_2],
                         [d_y_emitter_d_theta_1,d_y_emitter_d_theta_2]])

    def compute_location_covariance_from_two_aoa_measurements(self, pos1, theta1, pos1_cov, theta1_cov, pos2, theta2, pos2_cov, theta2_cov):
        #not incorperating agent location uncertainty but simple update to this equation to do so
        J_location_angles = self.location_from_two_aoa_measurements_angle_jacobian(pos1, theta1, pos2, theta2)  
        # print("J_location_angles",J_location_angles)
        angle_joing_covariance = np.array([[theta1_cov,0],[0,theta2_cov]])
        # print("angle_joing_covariance",angle_joing_covariance)
        
        return J_location_angles @ angle_joing_covariance @ J_location_angles.T

    def compute_location_from_two_aoa_measurements(self,pos1, theta1, pos2, theta2):
        x1 = pos1[0]
        y1 = pos1[1]
        x2 = pos2[0]
        y2 = pos2[1]
        xem = (y1/np.tan(theta2) - np.tan(theta1)/np.tan(theta2) * x1 - y2/np.tan(theta2)+x2)/(1 - np.tan(theta1)/np.tan(theta2))
        yem = np.tan(theta1)*(xem-x1)+y1
        return np.array([xem, yem])

    def measurement_model(self, xem, yem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)]])
    
    def measurement_residual(self, emmiter_location, aoa_measurements, measurement_locations):
        return np.array([self.measurement_model(emmiter_location[0], emmiter_location[1], loc[0], loc[1]) for loc in measurement_locations]).reshape((len(aoa_measurements),)) - np.array(aoa_measurements).reshape((len(aoa_measurements),))

    def batch_estimation(self):
        print("AOA measurements", self.aoa_measurement_values)
        x0 = self.estimated_emmiter_location
        sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(self.aoa_measurement_values,self.measurement_locations))
        # print("sol",sol)
        self.estimated_emmiter_location = sol.x
        # print("residuals", self.measurement_residual(emmiter_location,self.aoa_measurement_values, self.measurement_locations))
        print("batch emmiter location", self.estimated_emmiter_location)
        jacobians = self.stack_measurement_jacobian(self.estimated_emmiter_location, None, self.measurement_locations)

        self.estimated_emmiter_location_cov = self.aoa_measurement_variances[0] * np.linalg.inv(jacobians.T@jacobians)

        print()
    
    def stack_measurement_jacobian(self, emitter_location, aoa_measurements, measurement_locations):
        return np.array([self.measurement_jacobian(emitter_location[0], emitter_location[1], loc[0], loc[1]) for loc in measurement_locations])

    def measurement_jacobian(self, xem, yem, x, y):
        d_h_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        return np.array([d_h_d_x_emmitter, d_h_d_y_emmitter])
    
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

    def plot(self, ax):
        if self.estimated_emmiter_location is not None:
            self.plot_esimate_1_sigma_bounds(ax)
            ax.scatter(self.estimated_emmiter_location[0], self.estimated_emmiter_location[1], marker='x')

        

    