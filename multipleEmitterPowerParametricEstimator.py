import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from emitterPowerParametricEstimator import EmitterPowerParametricEstimator



class MultipleEmittorPowerParametricEstimator:
    def __init__(self, X_test):
        self.parametric_estimator_list = []
        self.predictive_mean = None
        self.X_test = np.array(X_test)
        self.num_test_points = int(np.sqrt(len(self.X_test)))

    
    def fit(self, measurement_locations, measurement_power_values, estimated_emitter_locations, measurement_groups_index):
        self.parametric_estimator_list = []
        for i, measurement_indices in enumerate(measurement_groups_index):
            self.parametric_estimator_list.append(EmitterPowerParametricEstimator())
            self.parametric_estimator_list[i].fit_parametric_estimator(np.array(measurement_power_values)[measurement_indices], np.array(measurement_locations)[measurement_indices], estimated_emitter_locations[i])
        self.predictive_mean = self.prediction(self.X_test, estimated_emitter_locations)
    
    def prediction(self, prediction_locations, estimated_emitter_locations):
        output = np.zeros((len(prediction_locations),1))
        for i, parametric_estimator in enumerate(self.parametric_estimator_list):
            output += parametric_estimator.prediction(prediction_locations, estimated_emitter_locations[i])

        
        return output

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
            
        