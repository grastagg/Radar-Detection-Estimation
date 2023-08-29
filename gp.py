import numpy as np


from kernels import radial_kernel


class GaussianProcess:
    def __init__(self, X_test):
        self.predictive_mean = None
        self.predictive_cov = None
        self.X_test = X_test


    
    def create_K_matrix(self, x1_vec, x2_vec, center, sigma_f, length_scale):
        K = np.zeros((len(x1_vec),len(x2_vec)))
        
        for i in range(len(x1_vec)):
            for j in range(len(x2_vec)):
                K[i,j] = radial_kernel(x1_vec[i],x2_vec[j], center, sigma_f, length_scale)
        
        return K
    
    def gp_prediction(self, x_test, x_meas, y_meas, emiter_location):
        print("TEST")
        length_scale = 50
        sigma_f = 1
        
        K_ss = self.create_K_matrix(x_test, x_test, emiter_location, sigma_f, length_scale)
        K_fs = self.create_K_matrix(x_meas, x_test, emiter_location, sigma_f, length_scale)
        K_sf = K_fs.T
        K_ff = self.create_K_matrix(x_meas, x_meas, emiter_location, sigma_f, length_scale)
        print("TEST")

        meas_variance = 0

        inv_K_ff = np.linalg.inv(K_ff + meas_variance * np.eye(len(x_meas)))
        
        predictive_mean = K_sf @ inv_K_ff @ y_meas 
        predictive_cov = K_ss - K_sf @ inv_K_ff @ K_fs
        self.predictive_mean = predictive_mean
        self.predictive_cov = predictive_cov

        return predictive_mean, predictive_cov

    def plot(self, ax):
        if self.predictive_mean is not None:
            X_test = np.array(self.X_test)


            ax.pcolormesh(X_test[:,0].reshape((50,50)), X_test[:,1].reshape((50,50)), self.predictive_mean.reshape((50,50)))