import numpy as np



class EmmitterLocationEstimator:
    '''
    Assumptions:
    1) location of agent is known exactly (could incorperate uncertainty due to this in the future)
    2) Uncertainty propogation is acurate enough using linearized models (jacobians, EKF)
    3) AOA measurements containt additive guassian noise with known variance (variance to be found through experimentation)
    
    '''
    def __init__(self):
        pass

        
        
        
    def compute_location_from_two_aoa_measurements(self,pos1, theta1, pos2, theta2):
        x1 = pos1[0]
        y1 = pos1[1]
        x2 = pos2[0]
        y2 = pos2[1]
        xem = (y1/np.tan(theta2) - np.tan(theta1)/np.tan(theta2) * x1 - y2/np.tan(theta2)+x2)/(1 - np.tan(theta1)/np.tan(theta2))
        yem = np.tan(theta1)*(xem-x1)+y1
        return xem, yem
    
    
    
    # def compute_location_covariance_from_two_aoa_measurements(self,)
            