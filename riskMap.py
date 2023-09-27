import numpy as np
import matplotlib.pyplot as plt

from threatProbability import threat_level_multiple_radar


class RiskMap:
    def __init__(self, bounds):
        self.max_radar_detection_range = 300
        self.max_SAM_range = 200
        self.max_no_escape_range = 100
        self.bounds = bounds

        self.map_resolution = 70
        self.estimate_emittor_params = []


    def threat_level(self, location, estimate_radar_location_list):
        max_radar_detection_range_list = self.max_radar_detection_range * np.ones(len(estimate_radar_location_list))
        max_SAM_range_list = self.max_SAM_range * np.ones(len(estimate_radar_location_list))
        max_no_escape_range_list = self.max_no_escape_range * np.ones(len(estimate_radar_location_list))
        return threat_level_multiple_radar(location, estimate_radar_location_list, max_radar_detection_range_list, max_SAM_range_list, max_no_escape_range_list)
    
    def compute_threat_level_map(self, estimate_radar_location_list):
        x_test = np.linspace(0, self.bounds[0], self.map_resolution)
        y_test = np.linspace(0, self.bounds[0], self.map_resolution)

        [X_test, Y_test] = np.meshgrid(x_test, y_test)

        map_value = np.zeros_like(X_test)

        for i in range(self.map_resolution):
            for j in range(self.map_resolution):
                map_value[i,j] = self.threat_level([X_test[i][j], Y_test[i][j]], estimate_radar_location_list)
        return map_value
    
    def update(self, estimate_emittor_params):
        self.estimate_emittor_params = estimate_emittor_params

    def plot(self, ax):
        if len(self.estimate_emittor_params)>0:
            x_test = np.linspace(0, self.bounds[0], self.map_resolution)
            y_test = np.linspace(0, self.bounds[0], self.map_resolution)

            [X_test, Y_test] = np.meshgrid(x_test, y_test)
            map_value = self.compute_threat_level_map(np.array(self.estimate_emittor_params)[:,0:2])
            c = ax.pcolormesh(X_test, Y_test, map_value)
            return c


        