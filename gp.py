import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from kernels import vectorized_radial_kernel


class GaussianProcess:
    def __init__(self, X_test):
        self.predictive_mean = None
        self.predictive_cov = None
        self.X_test = np.array(X_test)
        self.num_test_points = int(np.sqrt(len(self.X_test)))

        
        self.ground_truth = self.compute_ground_truth(self.X_test, np.array([600,600]), 100)
        self.plot_ground_truth()

    def plot_ground_truth(self):
        plt.figure()
        c = plt.pcolormesh(self.X_test[:,0].reshape((self.num_test_points,self.num_test_points)), self.X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.ground_truth.reshape((self.num_test_points,self.num_test_points))))
        plt.colorbar(c)
        plt.show()
        
    def compute_ground_truth(self, x_test, true_radar_pos, true_radar_power):
        dist = np.linalg.norm(x_test - true_radar_pos, axis=1)
        return true_radar_power / np.square(dist)


    
    def create_K_matrix(self, x1_vec, x2_vec, center, sigma_f, length_scale):
        # K = np.zeros((len(x1_vec),len(x2_vec)))
        # print(K.shape)
        
        # for i in range(len(x1_vec)):
        #     for j in range(len(x2_vec)):
        K = np.array([vectorized_radial_kernel(x1,x2_vec, center, sigma_f, length_scale) for x1 in x1_vec])
        
        return np.array(K)
    
    def gp_prediction(self, x_test, x_meas, y_meas, emiter_location):

        # self.parametricEstimator.fit_parametric_estimator(y_meas, x_meas, emiter_location)

        # mu_test = self.parametricEstimator.prediction(x_test, emiter_location).reshape((-1,))
        # mu_meas = self.parametricEstimator.prediction(x_meas, emiter_location).reshape((-1,))
        
        
        length_scale = 50
        sigma_f = 1
        
        # start = time.time()
        K_ss = self.create_K_matrix(x_test, x_test, emiter_location, sigma_f, length_scale)
        # print("Kss time", time.time()-start)
        # start = time.time()
        K_fs = self.create_K_matrix(x_meas, x_test, emiter_location, sigma_f, length_scale)
        # print("Ksf time:", time.time() -start)
        K_sf = K_fs.T
        # start = time.time()
        K_ff = self.create_K_matrix(x_meas, x_meas, emiter_location, sigma_f, length_scale)
        # print("K_ff time:", time.time() - start)

        meas_variance = 0

        # start = time.time()
        inv_K_ff = np.linalg.inv(K_ff + meas_variance * np.eye(len(x_meas)))
        # print("inverse time:", time.time() - start)
        
        # predictive_mean = mu_test #- K_sf @ inv_K_ff @ (np.array(y_meas) - mu_meas) 
        predictive_mean =  K_sf @ inv_K_ff @ np.array(y_meas) 
        predictive_cov = K_ss - K_sf @ inv_K_ff @ K_fs
        # predictive_cov = None
        self.predictive_mean = predictive_mean
        self.predictive_cov = predictive_cov

        return predictive_mean, predictive_cov

    def plot(self, ax):
        if self.predictive_mean is not None:
            X_test = np.array(self.X_test)


            # c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.predictive_mean).reshape((self.num_test_points,self.num_test_points)),vmin = -5, vmax =.5)
            Z = self.predictive_mean.reshape((self.num_test_points,self.num_test_points))
            c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), Z, norm=colors.SymLogNorm(linthresh = 0.0001, vmin = 0,vmax = 2, base=10))
            # c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.predictive_mean).reshape((self.num_test_points,self.num_test_points)))
            return c
        else:
            return None