import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class EmmitterLocationEstimator:
    '''
    Assumptions:
    1) location of agent is known exactly (could incorperate uncertainty due to this in the future)
    2) Uncertainty propogation is acurate enough using linearized models (jacobians, EKF)
    3) AOA measurements containt additive guassian noise with known variance (variance to be found through experimentation)

    '''
    def __init__(self, groundTruth):
        self.xHat = None
        self.sigmaHat = None
        self.groundTruth = groundTruth

        self.errorHistory = []

        self.measurementCount = 0
        self.alreadyComputedInitialEmitterLocation = False

        self.minimumDistanceSeperationForInitialEmitterLocationMeasurements = 30
        self.minimumAngularSeperationForInitialEmitterLocationMeasurements = 20 * np.pi/180

        self.firstMeasurementLocation = None
        self.firstMeasurementValue = None
        self.firstMeasurementCov = None

        self.unusedMeasurementsLocation = []
        self.unusedMeasurementsValue = []
        self.unusedMeasurementsCov = []
        
        
    def add_measurement(self, pos, theta, cov):
        if self.measurementCount == 0:
            print("first measurement")
            self.firstMeasurementValue = theta
            self.firstMeasurementLocation = pos
            self.firstMeasurementCov = cov 
            self.measurementCount += 1
        # elif not self.alreadyComputedInitialEmitterLocation and self.get_distance(pos, self.firstMeasurementLocation) < self.minimumDistanceSeperationForInitialEmitterLocationMeasurements:
        #     print("too close to first measurement")
        #     self.unusedMeasurementsLocation.append(pos)
        #     self.unusedMeasurementsValue.append(theta)
        #     self.unusedMeasurementsCov.append(cov)
        # elif not self.alreadyComputedInitialEmitterLocation and self.get_distance(pos, self.firstMeasurementLocation) > self.minimumDistanceSeperationForInitialEmitterLocationMeasurements:
        #     print("computing initial emmitter location")
        #     self.initial_emmitter_location_estimations(self.firstMeasurementLocation, self.firstMeasurementValue, None, self.firstMeasurementCov, pos, theta, None, cov )
        #     print("updating with unused measurements")
        #     for i in range(len(self.unusedMeasurementsValue)):
        #         self.ekf_update(self.unusedMeasurementsLocation[i], self.unusedMeasurementsValue[i], self.unusedMeasurementsCov[i])
        #     self.alreadyComputedInitialEmitterLocation = True
        elif not self.alreadyComputedInitialEmitterLocation and np.abs(self.firstMeasurementValue - theta)  < self.minimumAngularSeperationForInitialEmitterLocationMeasurements:
            print("too close to first measurement")
            self.unusedMeasurementsLocation.append(pos)
            self.unusedMeasurementsValue.append(theta)
            self.unusedMeasurementsCov.append(cov)
        elif not self.alreadyComputedInitialEmitterLocation and np.abs(self.firstMeasurementValue - theta) > self.minimumAngularSeperationForInitialEmitterLocationMeasurements:
            print("computing initial emmitter location")
            self.initial_emmitter_location_estimations(self.firstMeasurementLocation, self.firstMeasurementValue, None, self.firstMeasurementCov, pos, theta, None, cov )
            print("updating with unused measurements")
            for i in range(len(self.unusedMeasurementsValue)):
                self.ekf_update(self.unusedMeasurementsLocation[i], self.unusedMeasurementsValue[i], self.unusedMeasurementsCov[i])
            self.alreadyComputedInitialEmitterLocation = True
        else:
            print("normal update")
            self.ekf_update(pos, theta, cov)
            
        

    def get_distance(self,p1,p2):
        return np.linalg.norm(np.array(p1)-np.array(p2))       


    def initial_emmitter_location_estimations(self, pos1, theta1, pos1_cov, theta1_cov, pos2, theta2, pos2_cov, theta2_cov):
        xhat, yhat = self.compute_location_from_two_aoa_measurements(pos1, theta1, pos2, theta2)
        cov = self.compute_location_covariance_from_two_aoa_measurements(pos1, theta1, pos1_cov, theta1_cov, pos2, theta2, pos2_cov, theta2_cov)

        self.xHat = np.array([[xhat],[yhat]])
        self.sigmaHat = cov
        print(self.xHat)
        self.errorHistory.append(np.linalg.norm(self.xHat.reshape((2,)) - self.groundTruth))
        
        return self.xHat, self.sigmaHat
        
        
        
    def compute_location_from_two_aoa_measurements(self,pos1, theta1, pos2, theta2):
        x1 = pos1[0]
        y1 = pos1[1]
        x2 = pos2[0]
        y2 = pos2[1]
        xem = (y1/np.tan(theta2) - np.tan(theta1)/np.tan(theta2) * x1 - y2/np.tan(theta2)+x2)/(1 - np.tan(theta1)/np.tan(theta2))
        yem = np.tan(theta1)*(xem-x1)+y1
        return xem, yem
    
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
    
    def location_from_two_aoa_measurements_agent_position_jacobian(self, pos1, theta1, pos2, theta2):
        x1 = pos1[0]
        y1 = pos1[1]
        x2 = pos2[0]
        y2 = pos2[1]

        d_x_emitter_d_x_1 = -np.tan(theta1)/(np.tan(theta2)-np.tan(theta1)) 
        d_x_emitter_d_y_1 = np.tan(theta2)/(np.tan(theta1)*(np.tan(theta2)-np.tan(theta1))) 
        d_x_emitter_d_x_2 = np.tan(theta2)/(np.tan(theta2)-np.tan(theta1))
        d_x_emitter_d_y_2 = -1/(np.tan(theta2)-np.tan(theta1))
        d_y_emitter_d_x_1 =-(np.tan(theta1)*np.tan(theta2))/(np.tan(theta2)-np.tan(theta1))
        d_y_emitter_d_y_1 = 1/(1-np.tan(theta1)/np.tan(theta2))+1 
        d_y_emitter_d_x_2 = np.tan(theta1)/(1-np.tan(theta1)/np.tan(theta2)) 
        d_y_emitter_d_y_2 = -np.tan(theta1)/(np.tan(theta2)-np.tan(theta1)) 

        return np.array([[d_x_emitter_d_x_1, d_x_emitter_d_y_1, d_x_emitter_d_x_2, d_x_emitter_d_y_2],
                         [d_y_emitter_d_x_1, d_y_emitter_d_y_1, d_y_emitter_d_x_2, d_y_emitter_d_y_2]])
        
        
    
    def compute_location_covariance_from_two_aoa_measurements(self, pos1, theta1, pos1_cov, theta1_cov, pos2, theta2, pos2_cov, theta2_cov):
        #not incorperating agent location uncertainty but simple update to this equation to do so
        J_location_angles = self.location_from_two_aoa_measurements_angle_jacobian(pos1, theta1, pos2, theta2)  
        # print("J_location_angles",J_location_angles)
        angle_joing_covariance = np.array([[theta1_cov,0],[0,theta2_cov]])
        # print("angle_joing_covariance",angle_joing_covariance)
        
        return J_location_angles @ angle_joing_covariance @ J_location_angles.T
        
            

            
    def measurement_model(self, xem, yem, x, y):
        return np.array([[np.arctan2(yem-y, xem-x)]])
    
    def measurement_jacobian(self, xem, yem, x, y):
        d_h_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        d_h_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        return np.array([[d_h_d_x_emmitter, d_h_d_y_emmitter]])
    
    def ekf_update(self, aoa_measurement_pos, aoa_measurement_value, aoa_measurement_cov):
        xem = self.xHat[0][0]
        yem = self.xHat[1][0]
        x = aoa_measurement_pos[0]
        y = aoa_measurement_pos[1]

        H = self.measurement_jacobian(xem, yem, x, y)
        K = self.sigmaHat @ H.T @ np.linalg.inv(H@self.sigmaHat@H.T + np.array([[aoa_measurement_cov]]))
        self.xHat = self.xHat + K@(aoa_measurement_value - self.measurement_model(xem, yem, x, y))
        print(self.xHat)
        self.sigmaHat = (np.eye(2) - K@H)@self.sigmaHat
        self.errorHistory.append(np.linalg.norm(self.xHat.reshape((2,)) - self.groundTruth))
    
    def plot_xHatHistory(self):
        plt.figure()
        
        plt.plot(np.linspace(0, len(self.errorHistory), len(self.errorHistory)), self.errorHistory)
        plt.show()

    def plot_esimate_1_sigma_bounds(self,ax):
        if self.xHat is not None:
            # print("emmiter location estimate:",self.xHat)
            distr = multivariate_normal(cov = self.sigmaHat, mean = self.xHat.reshape((2,)))

            # Generating a meshgrid complacent with
            # the 3-sigma boundary
            mean_1, mean_2 = self.xHat[0], self.xHat[1]
            sigma_1, sigma_2 = self.sigmaHat[0,0], self.sigmaHat[1,1]

            invCovariance = np.linalg.inv(self.sigmaHat)
            
            # x = np.linspace(mean_1-3*sigma_1, mean_1+3*sigma_1, num=100)
            # y = np.linspace(mean_2-3*sigma_2, mean_2+3*sigma_2, num=100)
            x = np.linspace(0, 1200, num=100)
            y = np.linspace(0, 1200, num=100)
            X, Y = np.meshgrid(x,y)
            pdf = np.zeros(X.shape)
            malhanobisDist = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    malhanobisDist[i,j] = (np.array([[X[i,j], Y[i,j]]]) - np.array([[self.xHat[0][0], self.xHat[1][0]]])) @ invCovariance @(np.array([[X[i,j], Y[i,j]]]) - np.array([[self.xHat[0][0], self.xHat[1][0]]])).T
                    # pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

            # c = ax.contourf(X, Y, pdf, cmap='viridis')
            c = ax.contourf(X, Y, malhanobisDist, cmap='viridis',levels = [ 0,1,2,3])
            return c
        else:
            return None
        
        
    def plot(self, ax):
        self.plot_esimate_1_sigma_bounds(ax)
        if self.xHat is not None:
            ax.scatter(self.xHat[0], self.xHat[1])