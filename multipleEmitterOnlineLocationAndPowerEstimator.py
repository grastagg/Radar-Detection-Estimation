import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import block_diag
from matplotlib import colors
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# import params
# from params import measurement_jacobian, measurement_model

from sklearn.linear_model import RANSACRegressor

import matplotlib.pyplot as plt


from measurement_model import (
    measurement_model,
    measurement_jacobian,
    measurement_jacobian_jax,
)


class MultipleEmitterOnlineLocationAndPowerEstimator:
    def __init__(
        self,
        sensing_range,
        angle_measurement_std_dev,
        measurement_cov,
        X_test,
        radar_measurement_coeff,
        params,
        radarList,
        dataFile,
    ):
        self.dataFile = dataFile
        self.params = params
        self.radarList = radarList

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
        self.group_lists = [[] for i in range(self.params.numRadar)]
        self.estimated_emmiter_params = [[] for i in range(self.params.numRadar)]
        self.estimated_emmiter_params_covariances = [
            [] for i in range(self.params.numRadar)
        ]

        self.mahalonobis_distance_inlier_threshold = 4
        self.ransacResidualThreshold = 4
        # self.mahalonobis_distance_inlier_threshold = 2.5
        self.mal_dist_opt_point = None

        self.radar_measurement_coeff = radar_measurement_coeff
        # self.radar_measurement_coeff_db = 10*np.log10(radar_measurement_coeff)

        self.best_measurement_location_map = None
        self.plot_alpha = 0.5
        self.sigmaPlotList = []

        # these will be used to make sure aoa are different enough before starting estimate
        self.min_aoa_measurement = None
        self.max_aoa_measurement = None
        self.min_aoa_diff_to_start = 0.1
        # self.minDistBetweenModels = params.minInterRadarDist
        self.minDistBetweenModels = np.average([self.params.minInterRadarDistList])

        self.saveRadarData = self.params.saveDataToFile
        self.fileCounter = 0

        self.estimated_params_dict_list = [{} for i in range(self.params.numRadar)]
        self.estimated_params_cov_dict_list = [{} for i in range(self.params.numRadar)]

    def delete_lowest_probability_model(self):
        remove_indicies = []
        min_num_measurements = 100000
        for i, g_list in enumerate(self.group_lists):
            num_measurements = len(g_list)
            if num_measurements < min_num_measurements:
                min_num_measurements = num_measurements

        if min_num_measurements < 5:
            for i, g_list in enumerate(self.group_lists):
                num_measurements = len(g_list)
                if num_measurements == min_num_measurements:
                    remove_indicies.append(i)
                    for ind in g_list:
                        self.outlier_indicies = np.append(self.outlier_indicies, ind)

            self.estimated_emmiter_params = [
                i
                for j, i in enumerate(self.estimated_emmiter_params)
                if j not in remove_indicies
            ]
            self.estimated_emmiter_params_covariances = [
                i
                for j, i in enumerate(self.estimated_emmiter_params_covariances)
                if j not in remove_indicies
            ]
            self.group_lists = [
                i for j, i in enumerate(self.group_lists) if j not in remove_indicies
            ]

    def compute_inlier_aoa_diff(self, measurements):
        min_measurement = 10000
        max_measurement = -10000
        for measurement in measurements:
            if measurement[0] < min_measurement:
                min_measurement = measurement[0]
            if measurement[0] > max_measurement:
                max_measurement = measurement[0]
        return max_measurement - min_measurement

    def get_min_dist_to_other_models(self, new_model):
        if len(self.estimated_emmiter_params) == 0:
            return np.inf
        else:
            minDist = np.inf
            for mod in self.estimated_emmiter_params:
                dist = np.linalg.norm(mod[0:2] - new_model[0:2])
                if dist < minDist:
                    minDist = dist
            return minDist

    def compute_max_distance_measurement_to_model(
        self, measurement_locations, estimated_emitter_params
    ):
        return np.max(
            np.linalg.norm(
                measurement_locations - estimated_emitter_params[0:2], axis=1
            )
        )

    def fit_ransac_model(self, measurement_locations, measurements, original_index):
        # if len(measurements) > 3:
        if len(measurements) > 3:
            # try:
            def loss(y_true, y_pred, X, estimator):
                mean = estimator.get_estimate_emmitor_params()
                cov = estimator.compute_emmitor_estimate_covariance(X, y_true)
                return np.array(
                    [
                        self.mahalonobis_distance(y, X[i], mean, cov)
                        for i, y in enumerate(y_true)
                    ]
                )

            regressionModel = NonlinearEstimator(
                self.measurement_cov, self.radar_measurement_coeff
            )
            # ransacRegressor = RANSACRegressor(regressionModel, min_samples=2,random_state=0,loss = 'squared_error')#,residual_threshold=.09)
            ransacRegressor = RANSACRegressor(
                regressionModel,
                min_samples=2,
                random_state=0,
                loss=loss,
                residual_threshold=self.ransacResidualThreshold,
            )  # ,residual_threshold=.09)
            ransacRegressor.fit(np.array(measurement_locations), np.array(measurements))
            closestModelDist = self.get_min_dist_to_other_models(
                ransacRegressor.estimator_.get_estimate_emmitor_params()
            )
            if closestModelDist > self.minDistBetweenModels:
                if len(original_index[ransacRegressor.inlier_mask_ == True]) >= 5:
                    inlier_aoa_diff = self.compute_inlier_aoa_diff(
                        measurements[ransacRegressor.inlier_mask_ == True]
                    )
                    max_distance_measurement_to_model = (
                        self.compute_max_distance_measurement_to_model(
                            measurement_locations[ransacRegressor.inlier_mask_ == True],
                            ransacRegressor.estimator_.get_estimate_emmitor_params(),
                        )
                    )
                    if (
                        inlier_aoa_diff > self.min_aoa_diff_to_start
                        and max_distance_measurement_to_model
                        < 1.5 * self.params.agentSensingRange
                    ):
                        self.estimated_emmiter_params.append(
                            ransacRegressor.estimator_.get_estimate_emmitor_params()
                        )
                        self.group_lists.append(
                            original_index[ransacRegressor.inlier_mask_ == True]
                        )
                        self.estimated_emmiter_params_covariances.append(
                            ransacRegressor.estimator_.compute_emmitor_estimate_covariance(
                                np.array(self.measurement_locations)[
                                    self.group_lists[-1]
                                ],
                                np.array(self.measurement_values)[self.group_lists[-1]],
                            )
                        )
                        for ind in original_index[ransacRegressor.inlier_mask_ == True]:
                            self.outlier_indicies = np.delete(
                                self.outlier_indicies,
                                np.argwhere(self.outlier_indicies == ind),
                            )
            self.fit_ransac_model(
                measurement_locations[ransacRegressor.inlier_mask_ == False],
                measurements[ransacRegressor.inlier_mask_ == False],
                original_index[ransacRegressor.inlier_mask_ == False],
            )
            # except:
            #     pass

    def minimized_angle(self, angle):
        while angle < -np.pi:
            angle += 2 * np.pi

        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def ekf_update(
        self, measurement_pos, measurement_value, measurement_cov, x_prev, sigma_prev
    ):
        xem = x_prev[0]
        yem = x_prev[1]
        erp = x_prev[2]
        x = measurement_pos[0]
        y = measurement_pos[1]

        H = self.measurement_jacobian(xem, yem, erp, x, y)
        # K = sigma_prev @ H.T @ np.linalg.inv(H @ sigma_prev @ H.T + measurement_cov)
        regularization = 0.0 * np.eye(measurement_cov.shape[0])
        K = (
            sigma_prev
            @ H.T
            @ np.linalg.solve(
                H @ sigma_prev @ H.T + regularization + measurement_cov,
                np.eye(H.shape[0]),
            )
        )

        # zHat = self.measurement_model(xem, yem, erp, x, y)
        zHat = self.measurement_model(xem, yem, erp, x, y)
        inovation = np.zeros_like(zHat)
        inovation[0] = self.minimized_angle(measurement_value[0] - zHat[0][0])
        inovation[1] = measurement_value[1] - zHat[1][0]

        xHat = x_prev + K @ (inovation).reshape((-1,))
        sigmaHat = (np.eye(3) - K @ H) @ sigma_prev
        return xHat, sigmaHat

    def mahalonobis_distance(
        self,
        measurement_value,
        measurement_location,
        estimated_emitter_params,
        emitter_params_covariance,
    ):
        z_hat = self.measurement_model(
            estimated_emitter_params[0],
            estimated_emitter_params[1],
            estimated_emitter_params[2],
            measurement_location[0],
            measurement_location[1],
        )
        measurement_jacobian = self.measurement_jacobian(
            estimated_emitter_params[0],
            estimated_emitter_params[1],
            estimated_emitter_params[2],
            measurement_location[0],
            measurement_location[1],
        )

        z_hat_cov = (
            measurement_jacobian @ emitter_params_covariance @ measurement_jacobian.T
        )

        aoa_value = measurement_value[0]
        power_value = measurement_value[1]

        diff_square_aoa = min(
            (aoa_value - z_hat[0][0]) ** 2,
            min(
                (aoa_value - (z_hat[0][0] + 2 * np.pi)) ** 2,
                (aoa_value - (z_hat[0][0] - 2 * np.pi)) ** 2,
            ),
        )
        diff_square_power = (power_value - z_hat[1][0]) ** 2
        diff_square = np.array([[diff_square_aoa], [diff_square_power]])
        return np.sqrt(diff_square.T @ np.linalg.inv(z_hat_cov) @ diff_square)[0][0]

    def get_linear_model_from_aoa_measurement(self, aoa_value, measurement_location):
        m = np.tan(aoa_value)
        b = measurement_location[1] - m * measurement_location[0]
        return m, b

    def get_linear_model_from_aoa_measurement(self, aoa_value, measurement_location):
        m = np.tan(aoa_value)
        b = measurement_location[1] - m * measurement_location[0]
        return m, b

    def mahalonobis_distance_in_x_y_space(
        self, aoa_value, measurement_location, mean, covariance
    ):
        mean = mean.reshape((2, 1))
        m, b = self.get_linear_model_from_aoa_measurement(
            aoa_value, measurement_location
        )

        A = np.array([m, -1])
        C = np.zeros((3, 3))
        inv_cov = np.linalg.inv(covariance)
        C[0:2, 0:2] = 2 * inv_cov
        C[0:2, 2] = A.T
        C[2, 0:2] = A
        D = np.zeros((3, 1))
        D[0:2, 0] = 2 * mean.T @ inv_cov
        D[2, 0] = -b

        opt = np.linalg.solve(C, D)
        x = opt[0:2]
        self.mal_dist_opt_point = x

        mahalonobis_distance_squared = ((x - mean).T @ inv_cov @ (x - mean))[0][0]
        return np.sqrt(mahalonobis_distance_squared)

    # def measurement_model(self, xem, yem, erp, x, y):
    #     return np.array([[np.arctan2(yem-y, xem-x)], [(erp*self.radar_measurement_coeff)/((xem-x)**2 + (yem-y)**2)]])

    def measurement_model(self, xem, yem, erp, x, y):
        return measurement_model(xem, yem, erp, x, y, self.params.radarMeasurementCoeff)
        # return np.array([[np.arctan2(yem-y, xem-x)], [erp_db + self.radar_measurement_coeff_db - 10*np.log10((xem-x)**2 + (yem-y)**2)]])

    def measurement_jacobian(self, xem, yem, erp, x, y):
        return measurement_jacobian(
            xem, yem, erp, x, y, self.params.radarMeasurementCoeff
        )
        # d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        # d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        # d_h1_d_p_emmitter = 0

        # d_h2_d_x_emmitter = -(20*(xem-x))/(np.log(10)*((xem-x)**2+(yem-y)**2))
        # d_h2_d_y_emmitter = -(20*(yem-y))/(np.log(10)*((yem-y)**2+(xem-x)**2))
        # d_h2_d_p_emmitter = 1

        # return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    # def measurement_jacobian(self, xem, yem, erp, x, y):
    #     d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
    #     d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
    #     d_h1_d_p_emmitter = 0

    #     d_h2_d_x_emmitter = -(2*erp*self.radar_measurement_coeff*(xem-x))/((xem-x)**2+(yem-y)**2)**2
    #     d_h2_d_y_emmitter = -(2*erp*self.radar_measurement_coeff*(yem-y))/((yem-y)**2+(xem-x)**2)**2
    #     d_h2_d_p_emmitter = self.radar_measurement_coeff/((yem-y)**2+(xem-x)**2)

    #     return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])
    def update_aoa_diff_value(self, measurement_value):
        if self.min_aoa_measurement is None:
            self.min_aoa_measurement = measurement_value[0]
            self.max_aoa_measurement = measurement_value[0]
        else:
            if measurement_value[0] < self.min_aoa_measurement:
                self.min_aoa_measurement = measurement_value[0]
            if measurement_value[0] > self.max_aoa_measurement:
                self.max_aoa_measurement = measurement_value[0]

    def recompute_aoa_diff(self, measurements):
        if len(measurements) == 0:
            self.min_aoa_measurement = None
            self.max_aoa_measurement = None
        elif len(measurements) == 1:
            self.min_aoa_measurement = measurements[0][0]
            self.max_aoa_measurement = measurements[0][0]
        else:
            self.min_aoa_measurement = 10000
            self.max_aoa_measurement = -10000
            for measurement in measurements:
                if measurement[0] < self.min_aoa_measurement:
                    self.min_aoa_measurement = measurement[0]
                if measurement[0] > self.max_aoa_measurement:
                    self.max_aoa_measurement = measurement[0]

    ##########################code for known data association############################################
    def compute_location_from_two_aoa_measurements(self, pos1, theta1, pos2, theta2):
        x1 = pos1[0]
        y1 = pos1[1]
        x2 = pos2[0]
        y2 = pos2[1]
        xem = (
            y1 / np.tan(theta2)
            - np.tan(theta1) / np.tan(theta2) * x1
            - y2 / np.tan(theta2)
            + x2
        ) / (1 - np.tan(theta1) / np.tan(theta2))
        yem = np.tan(theta1) * (xem - x1) + y1
        return xem, yem

    def compute_initial_location_estimate(
        self, measurementLocations, measurements, radarId
    ):
        count = 0
        x_0Sum = 0
        y_0Sum = 0
        for i in range(len(measurementLocations)):
            for j in range(i + 1, len(measurementLocations)):
                x_0, y_0 = self.compute_location_from_two_aoa_measurements(
                    measurementLocations[i],
                    measurements[i][0],
                    measurementLocations[j],
                    measurements[j][0],
                )
                if (
                    x_0 > 0
                    and y_0 > 0
                    and x_0 < self.params.bounds[0]
                    and y_0 < self.params.bounds[1]
                ):
                    x_0Sum += x_0
                    y_0Sum += y_0
                    count += 1
        # if radarId == 0:
        #     fig, ax = plt.subplots()
        #     for i in range(len(measurementLocations)):
        #         loc = measurementLocations[i]
        #
        #         ax.plot(
        #             [loc[0], loc[0] + 10000 * np.cos(measurements[i][0])],
        #             [loc[1], loc[1] + 10000 * np.sin(measurements[i][0])],
        #             color="blue",
        #         )
        #     ax.scatter([x_0Sum / count], [y_0Sum / count], color="red", zorder=10)
        #     plt.show()
        return x_0Sum / count, y_0Sum / count

    def compute_initial_erp_estimate(
        self, measurementLocations, measurements, x_0, y_0
    ):
        radarLocation = np.array([x_0, y_0])
        distSquared = np.linalg.norm(measurementLocations - radarLocation, axis=1) ** 2
        measurementsTemp = measurements.copy()
        mask = measurementsTemp[:, 1] < 1e-5
        erp = np.mean(
            measurementsTemp[mask, 1]
            * distSquared[mask]
            / self.params.radarMeasurementCoeff
        )

        # erp = np.mean(
        #     measurements[:, 1] * distSquared / self.params.radarMeasurementCoeff
        # )
        return erp

    def measurement_residual(self, emitter_params, measurements, measurement_locations):
        return np.array(
            [
                measurement_model(
                    emitter_params[0],
                    emitter_params[1],
                    emitter_params[2],
                    loc[0],
                    loc[1],
                    self.params.radarMeasurementCoeff,
                )
                for loc in measurement_locations
            ]
        ).reshape((-1,)) - np.array(measurements).reshape((-1,))

    def batch_estimation(
        self, measurement_locations, measurement_values, measurement_cov, radarId
    ):
        x_0, y_0 = self.compute_initial_location_estimate(
            measurement_locations, measurement_values, radarId
        )
        erp_0 = self.compute_initial_erp_estimate(
            measurement_locations, measurement_values, x_0, y_0
        )

        x_0 = self.radarList[radarId].position[0]
        y_0 = self.radarList[radarId].position[1]
        x0 = [x_0, y_0, erp_0]
        print("x0", x0)
        print(
            "true value",
            self.radarList[radarId].position,
            self.radarList[radarId].outputPower,
        )
        # if radarId == 0:
        #     print("x0", x0)
        #     print(
        #         "true value",
        #         self.radarList[radarId].position,
        #         self.radarList[radarId].outputPower
        #         * self.radarList[radarId].transmitGain,
        #     )
        # x0 = [
        #     self.radarList[radarId].position[0],
        #     self.radarList[radarId].position[1],
        #     self.radarList[radarId].outputPower * self.radarList[radarId].transmitGain,
        # ]
        # sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(measurement_values,measurement_locations))
        sol = least_squares(
            self.measurement_residual,
            x0,
            jac=self.stack_measurement_jacobian,
            args=(measurement_values, measurement_locations),
            bounds=(
                [x0[0] - 5000, x0[1] - 5000, 1],
                [x0[0] + 5000, x0[1] + 5000, 20 * self.params.erp],
            ),
            loss="soft_l1",
            # bounds=(
            #     [0, 0, 10],
            #     [self.params.bounds[0], self.params.bounds[1], 20 * self.params.erp],
            # ),
        )
        estimated_emmiter_params = sol.x
        jacobians = self.stack_measurement_jacobian(
            estimated_emmiter_params, None, measurement_locations
        )
        combined_measurment_cov = block_diag(
            *[measurement_cov for i in measurement_locations]
        )
        estimated_emmiter_params_cov = np.linalg.inv(
            jacobians.T @ np.linalg.inv(combined_measurment_cov) @ jacobians
        )
        return estimated_emmiter_params, estimated_emmiter_params_cov

    def stack_measurement_jacobian(
        self, emitter_params, measurements, measurement_locations
    ):
        return np.array(
            [
                measurement_jacobian(
                    emitter_params[0],
                    emitter_params[1],
                    emitter_params[2],
                    loc[0],
                    loc[1],
                    self.params.radarMeasurementCoeff,
                )
                for loc in measurement_locations
            ]
        ).reshape((2 * len(measurement_locations), 3))

    # def measurement_jacobian(self, xem, yem, pem, x, y):
    #     d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
    #     d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
    #     d_h1_d_p_emmitter = 0

    #     d_h2_d_x_emmitter = -(2*pem*(xem-x))/((xem-x)**2+(yem-y)**2)**2
    #     d_h2_d_y_emmitter = -(2*pem*(yem-y))/((yem-y)**2+(xem-x)**2)**2
    #     d_h2_d_p_emmitter = 1/((yem-y)**2+(xem-x)**2)

    #     return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    def add_measurement_known_association(
        self, measurement_location, measurement_value, radarId, timestep
    ):
        radarId = int(radarId)
        self.measurement_locations.append(measurement_location)
        self.measurement_values.append(measurement_value)
        self.group_lists[radarId].append(len(self.measurement_values) - 1)
        if measurement_value[1] > 1e-3:
            return

        numMeasurementsNeeded = 10
        # if len(self.group_lists[radarId]) > numMeasurementsNeeded:
        aoaDiff = self.compute_inlier_aoa_diff(
            np.array(self.measurement_values)[self.group_lists[radarId]]
        )
        if len(self.estimated_emmiter_params[radarId]) > 0:
            estimated_emmiter_param, estimated_emmiter_params_covariances = (
                self.ekf_update(
                    measurement_location,
                    measurement_value,
                    self.measurement_cov,
                    self.estimated_emmiter_params[radarId],
                    self.estimated_emmiter_params_covariances[radarId],
                )
            )
            if estimated_emmiter_param[2] < 0:
                print("estimated power level too small")
                updated_using_ekf = False
            elif (
                estimated_emmiter_param[0] < 0
                or estimated_emmiter_param[1] < 0
                or estimated_emmiter_param[0] > self.params.bounds[0]
                or estimated_emmiter_param[1] > self.params.bounds[1]
            ):
                print("estimated location out of bounds")
                updated_using_ekf = False
            else:
                self.estimated_emmiter_params[radarId] = estimated_emmiter_param
                self.estimated_emmiter_params_covariances[radarId] = (
                    estimated_emmiter_params_covariances
                )
        elif (
            len(self.group_lists[radarId]) >= numMeasurementsNeeded
            and aoaDiff > self.min_aoa_diff_to_start
        ):
            estimated_emmiter_param, estimated_emmiter_params_covariances = (
                self.batch_estimation(
                    np.array(self.measurement_locations)[self.group_lists[radarId]],
                    np.array(self.measurement_values)[self.group_lists[radarId]],
                    self.measurement_cov,
                    radarId,
                )
            )
            self.estimated_emmiter_params[radarId] = estimated_emmiter_param
            self.estimated_emmiter_params_covariances[radarId] = (
                estimated_emmiter_params_covariances
            )

        # if radarId == 0:
        #     print("estimated location", self.estimated_emmiter_params[radarId])
        #     print(
        #         "true location",
        #         self.radarList[radarId].position,
        #         self.radarList[radarId].outputPower
        #         * self.radarList[radarId].transmitGain,
        # )
        if self.saveRadarData:
            self.fileCounter += 1
            file = open(self.dataFile + "radarEstimateTimestamps.txt", "a")
            file.write(str(timestep) + "\n")
            file.close()
            for radarId in range(self.params.numRadar):
                self.estimated_params_dict_list[radarId][str(self.fileCounter)] = (
                    self.estimated_emmiter_params[radarId]
                )
                self.estimated_params_cov_dict_list[radarId][str(self.fileCounter)] = (
                    self.estimated_emmiter_params_covariances[radarId]
                )
                # np.save(
                #     self.dataFile
                #     + "/radar_"
                #     + str(radarId)
                #     + "/estimated_params/"
                #     + str(self.fileCounter),
                #     self.estimated_emmiter_params[radarId],
                # )
                # np.save(
                #     self.dataFile
                #     + "/radar_"
                #     + str(radarId)
                #     + "/estimated_params_cov/"
                #     + str(self.fileCounter),
                #     self.estimated_emmiter_params_covariances[radarId],
                # )

        # np.save(params.dataFile+"/estimated_params/"+str(self.fileCounter), self.estimated_emmiter_params)
        # np.save(params.dataFile+"/estimated_params_cov/"+str(self.fileCounter), self.estimated_emmiter_params_covariances)

    def save_radar_estimates_to_file(self):
        for radarId in range(self.params.numRadar):
            np.savez(
                self.dataFile + "radar_" + str(radarId) + "/estimated_params.npz",
                **self.estimated_params_dict_list[radarId],
            )
            np.savez(
                self.dataFile + "radar_" + str(radarId) + "/estimated_params_cov.npz",
                **self.estimated_params_cov_dict_list[radarId],
            )

    ##########################code for known data association############################################

    def add_measurement(self, measurement_location, measurement_value):
        self.measurement_locations.append(measurement_location)
        self.measurement_values.append(measurement_value)

        # print("aoa diff to start", self.max_aoa_measurement - self.min_aoa_measurement)
        # print("measurements", self.measurement_values)
        # print("measurement locations", self.measurement_locations)
        # if len(self.measurement_values) > 5:
        # if self.max_aoa_measurement - self.min_aoa_measurement > self.min_aoa_diff_to_start:
        updated_using_ekf = False
        minimum_mal_dist = np.inf
        minimum_mal_dist_index = -1
        minimum_mal_dist_dist_from_measurement_to_model = np.inf
        mal_dist_list = []
        distance_list = []
        index_list = []
        for i, emitter_param in enumerate(self.estimated_emmiter_params):
            measurement_mahalonobis_distance = self.mahalonobis_distance(
                measurement_value,
                measurement_location,
                emitter_param,
                self.estimated_emmiter_params_covariances[i],
            )
            measurement_mahalonobis_distance_in_x_y = (
                self.mahalonobis_distance_in_x_y_space(
                    measurement_value[0],
                    measurement_location,
                    emitter_param[0:2],
                    self.estimated_emmiter_params_covariances[i][0:2, 0:2],
                )
            )
            combined_mahalanobis_distance = (
                measurement_mahalonobis_distance_in_x_y
                + measurement_mahalonobis_distance
            )
            dist_from_measurement_to_model = np.linalg.norm(
                measurement_location - emitter_param[0:2]
            )
            # print("dist_from_measurement_to_model", dist_from_measurement_to_model)
            # print("measurement_mahalonobis_distance", measurement_mahalonobis_distance)
            # print(
            #     "measurement_mahalonobis_distance_in_x_y",
            #     measurement_mahalonobis_distance_in_x_y,
            # )
            if (
                measurement_mahalonobis_distance
                < self.mahalonobis_distance_inlier_threshold
            ):
                mal_dist_list.append(measurement_mahalonobis_distance)
                distance_list.append(dist_from_measurement_to_model)
                index_list.append(i)
            # if measurement_mahalonobis_distance < minimum_mal_dist:
            #     minimum_mal_dist  = measurement_mahalonobis_distance
            #     minimum_mal_dist_index = i
            #     minimum_mal_dist_dist_from_measurement_to_model = dist_from_measurement_to_model
            # if measurement_mahalonobis_distance_in_x_y < minimum_mal_dist:
            #     minimum_mal_dist  = measurement_mahalonobis_distance_in_x_y
            #     minimum_mal_dist_index = i
            # if combined_mahalanobis_distance < minimum_mal_dist:
            #     minimum_mal_dist  = combined_mahalanobis_distance
            #     minimum_mal_dist_index = i

        if len(mal_dist_list) > 0:
            min_index = np.argmin(np.array(distance_list))
            minimum_mal_dist = mal_dist_list[min_index]
            minimum_mal_dist_index = index_list[min_index]
            minimum_mal_dist_dist_from_measurement_to_model = distance_list[min_index]

        if (
            minimum_mal_dist < self.mahalonobis_distance_inlier_threshold
            and minimum_mal_dist_dist_from_measurement_to_model
            < 1.5 * self.params.agentSensingRange
        ):
            estimated_emmiter_param, estimated_emmiter_params_covariances = (
                self.ekf_update(
                    measurement_location,
                    measurement_value,
                    self.measurement_cov,
                    self.estimated_emmiter_params[minimum_mal_dist_index],
                    self.estimated_emmiter_params_covariances[minimum_mal_dist_index],
                )
            )
            if estimated_emmiter_param[2] < 0:
                print("estimated power level too small")
                updated_using_ekf = False
            else:
                self.estimated_emmiter_params[minimum_mal_dist_index] = (
                    estimated_emmiter_param
                )
                self.estimated_emmiter_params_covariances[minimum_mal_dist_index] = (
                    estimated_emmiter_params_covariances
                )
                self.group_lists[minimum_mal_dist_index] = np.append(
                    self.group_lists[minimum_mal_dist_index],
                    (len(self.measurement_values) - 1),
                )
                updated_using_ekf = True

        if not updated_using_ekf:
            # self.update_aoa_diff_value(measurement_value)
            self.outlier_indicies = np.append(
                self.outlier_indicies, len(self.measurement_values) - 1
            )
            # print("Angle diff",self.max_aoa_measurement - self.min_aoa_measurement)
            # if self.max_aoa_measurement - self.min_aoa_measurement > self.min_aoa_diff_to_start:
            self.fit_ransac_model(
                np.array(self.measurement_locations)[self.outlier_indicies],
                np.array(self.measurement_values)[self.outlier_indicies],
                self.outlier_indicies,
            )
            # self.recompute_aoa_diff(np.array(self.measurement_values)[self.outlier_indicies])
            # if foundFit:

        # print("ransac prediction", self.estimated_emmiter_params)
        # print("covariance", self.estimated_emmiter_params_covariances)
        # print("group lists", self.group_lists)
        # print("outliers", self.outlier_indicies)
        # print()
        if self.saveRadarData:
            self.fileCounter += 1
            np.save(
                params.dataFile + "/estimated_params/" + str(self.fileCounter),
                self.estimated_emmiter_params,
            )
            np.save(
                params.dataFile + "/estimated_params_cov/" + str(self.fileCounter),
                self.estimated_emmiter_params_covariances,
            )
            # np.save("./saved_data/estimated_params/"+str(self.fileCounter), self.estimated_emmiter_params)
            # np.save("./saved_data/estimated_params_cov/"+str(self.fileCounter), self.estimated_emmiter_params_covariances)

        # else:
        #     # self.outlier_indicies = np.append(self.outlier_indicies, 0)
        #     self.outlier_indicies = np.append(self.outlier_indicies, len(self.measurement_values)-1)

    def power_basis_function(self, radial_distance):
        return self.radar_measurement_coeff / radial_distance**2

    def create_power_basis_function_matrix(self, measurment_locations, center):
        A = np.zeros((len(measurment_locations), 1))
        for i, meas_loc in enumerate(measurment_locations):
            A[i] = self.power_basis_function(np.linalg.norm(meas_loc - center))
        return A

    def prediction(self, prediction_locations, estimated_emittor_params):
        A = self.create_power_basis_function_matrix(
            prediction_locations, estimated_emittor_params[0:2]
        )
        return A * estimated_emittor_params[2]

    def combined_power_prediction(self, prediction_locations):
        output = np.zeros((len(prediction_locations), 1))
        for i, emmitor_params in enumerate(self.estimated_emmiter_params):
            output += self.prediction(prediction_locations, emmitor_params)
        return output

    def plot(self, ax):
        color_list = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]
        # color_list = ['tab:blue','tab:orange']
        c = None

        all_lines = []
        all_scatter = []
        # print("TEST 2 ", self.outlier_indicies)
        for i, angle_indicies in enumerate([self.outlier_indicies]):
            if angle_indicies.size > 0:
                line = self.plot_angle_of_arrival_measurements(
                    ax,
                    "r",
                    np.array(self.measurement_locations)[angle_indicies],
                    np.array(self.measurement_values)[:, 0][angle_indicies],
                )
                all_lines += line
        # if len(self.estimated_emmiter_params) > 0:
        if any(self.measurement_locations):
            for i, angle_indicies in enumerate(self.group_lists):
                line = self.plot_angle_of_arrival_measurements(
                    ax,
                    color_list[i % len(color_list)],
                    np.array(self.measurement_locations)[angle_indicies],
                    np.array(self.measurement_values)[:, 0][angle_indicies],
                )
                all_lines += line

            for i, estimated_emmiter_params in enumerate(self.estimated_emmiter_params):
                if len(estimated_emmiter_params) > 0:
                    scatter = ax.scatter(
                        estimated_emmiter_params[0],
                        estimated_emmiter_params[1],
                        marker="x",
                        c="g",
                        zorder=100000,
                    )
                    all_scatter.append(scatter)
                # c = self.plot_esimate_1_sigma_bounds(ax, estimated_emmiter_params, self.estimated_emmiter_params_covariances[i])
        # if self.mal_dist_opt_point is not None:
        #     ax.scatter(self.mal_dist_opt_point[0], self.mal_dist_opt_point[1])
        # c = self.plot_power(ax)

        for c in self.sigmaPlotList:
            c.remove()
        self.sigmaPlotList = []
        if len(self.estimated_emmiter_params) > 0:
            for i in range(len(self.estimated_emmiter_params)):
                if len(self.estimated_emmiter_params[i]) > 0:
                    c = self.plot_esimate_1_sigma_bounds(
                        ax,
                        self.estimated_emmiter_params[i],
                        self.estimated_emmiter_params_covariances[i],
                    )
                    self.sigmaPlotList.append(c)

        return all_lines, all_scatter

    def plot_angle_of_arrival_measurements(
        self, ax, color, measurement_locations, measurement_angle_of_arrival_values
    ):
        line_width = 1
        # for i,angle in enumerate(measurement_angle_of_arrival_values):
        lines = []
        # print("TEST")
        # print("measurement locations", measurement_locations)
        for i in range(len(measurement_angle_of_arrival_values)):
            angle = measurement_angle_of_arrival_values[i]
            start_x = measurement_locations[i][0]
            start_y = measurement_locations[i][1]
            end_x = start_x + self.sensing_range * np.cos(angle)
            end_y = start_y + self.sensing_range * np.sin(angle)
            p1 = ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                c=color,
                linewidth=line_width,
                alpha=self.plot_alpha,
            )
            end_x = start_x + self.sensing_range * np.cos(
                angle + self.angle_measurement_std_dev
            )
            end_y = start_y + self.sensing_range * np.sin(
                angle + self.angle_measurement_std_dev
            )
            p2 = ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                linestyle="--",
                c=color,
                linewidth=line_width,
                alpha=self.plot_alpha,
            )
            end_x = start_x + self.sensing_range * np.cos(
                angle - self.angle_measurement_std_dev
            )
            end_y = start_y + self.sensing_range * np.sin(
                angle - self.angle_measurement_std_dev
            )
            p3 = ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                linestyle="--",
                c=color,
                linewidth=line_width,
                alpha=self.plot_alpha,
            )
            lines.append(p1[0])
            lines.append(p2[0])
            lines.append(p3[0])
        return lines

    def plot_esimate_1_sigma_bounds(
        self, ax, estimated_emmiter_params, estimated_emmiter_params_cov
    ):
        invCovariance = np.linalg.inv(estimated_emmiter_params_cov[0:2, 0:2])

        x = np.linspace(0, self.params.bounds[1], num=100)
        y = np.linspace(0, self.params.bounds[1], num=100)
        X, Y = np.meshgrid(x, y)
        malhanobisDist = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                malhanobisDist[i, j] = (
                    (
                        np.array([[X[i, j], Y[i, j]]])
                        - np.array(
                            [[estimated_emmiter_params[0], estimated_emmiter_params[1]]]
                        )
                    )
                    @ invCovariance
                    @ (
                        np.array([[X[i, j], Y[i, j]]])
                        - np.array(
                            [[estimated_emmiter_params[0], estimated_emmiter_params[1]]]
                        )
                    ).T
                )

        c = ax.contourf(
            X, Y, malhanobisDist, cmap="viridis", levels=[0, 1, 2, 3], zorder=100000
        )
        return c

    def plot_power(self, ax):
        if len(self.estimated_emmiter_params) > 0:
            X_test = np.array(self.X_test)
            predictive_mean = self.combined_power_prediction(self.X_test)
            num_test_points = int(np.sqrt(len(X_test)))

            # c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.predictive_mean).reshape((self.num_test_points,self.num_test_points)),vmin = -5, vmax =.5)
            Z = predictive_mean.reshape((num_test_points, num_test_points))
            c = ax.pcolormesh(
                X_test[:, 0].reshape((num_test_points, num_test_points)),
                X_test[:, 1].reshape((num_test_points, num_test_points)),
                Z,
                norm=colors.SymLogNorm(
                    linthresh=0.00001, vmin=0, vmax=np.max(Z), base=10
                ),
            )
            # c = ax.pcolormesh(X_test[:,0].reshape((self.num_test_points,self.num_test_points)), X_test[:,1].reshape((self.num_test_points,self.num_test_points)), np.log10(self.predictive_mean).reshape((self.num_test_points,self.num_test_points)))
            return c
        else:
            return None


class NonlinearEstimator:
    def __init__(self, measurement_cov, radar_measurement_coeff):
        self.estimated_emmiter_params = None
        self.estimated_emmiter_params_cov = None
        self.measurement_covariance = measurement_cov
        self.radar_measurement_coeff = radar_measurement_coeff
        # self.radar_measurement_coeff_db = 10*np.log10(radar_measurement_coeff)

    def measurement_residual(self, emitter_params, measurements, measurement_locations):
        return np.array(
            [
                self.measurement_model(
                    emitter_params[0],
                    emitter_params[1],
                    emitter_params[2],
                    loc[0],
                    loc[1],
                )
                for loc in measurement_locations
            ]
        ).reshape((-1,)) - np.array(measurements).reshape((-1,))

    def fit(self, X, y):
        x0 = np.array(
            [
                params.bounds[0] / 2,
                params.bounds[1] / 2,
                params.radarOutputPower * params.radarTransmitGain,
            ]
        )
        # x0 = np.array([params.bounds[0]/2,params.bounds[1]/2,params.radarOutputPower*params.radarTransmitGain])
        sol = least_squares(
            self.measurement_residual,
            x0,
            jac=self.stack_measurement_jacobian,
            args=(y, X),
            bounds=([0, 0, 10], [params.bounds[0], params.bounds[1], np.inf]),
        )
        # sol = least_squares(self.measurement_residual, x0,jac=self.stack_measurement_jacobian, args=(y,X), bounds=([-np.inf,-np.inf,10],[np.inf,np.inf,np.inf]))
        self.estimated_emmiter_params = sol.x
        # self.estimated_emmiter_location_covariances = self.compute_emmitor_estimate_covariance(X, y, self.measurement_variance)

    def score(self, X, y):
        # print("X",X)
        # print("y",y)
        # print("score",np.linalg.norm(self.measurement_residual(self.estimated_emmiter_params, y, X))**2)
        return (
            1
            / np.linalg.norm(
                self.measurement_residual(self.estimated_emmiter_params, y, X)
            )
            ** 2
        )

    def predict(self, X):
        return np.array(
            [
                self.measurement_model(
                    self.estimated_emmiter_params[0],
                    self.estimated_emmiter_params[1],
                    self.estimated_emmiter_params[2],
                    loc[0],
                    loc[1],
                )
                for loc in X
            ]
        ).reshape((-1, 2))

    # def measurement_model(self, xem, yem, erp, x, y):
    #     return np.array([[np.arctan2(yem-y, xem-x)], [(self.radar_measurement_coeff * erp)/((xem-x)**2 + (yem-y)**2)]])

    def stack_measurement_jacobian(
        self, emitter_params, measurements, measurement_locations
    ):
        return np.array(
            [
                self.measurement_jacobian(
                    emitter_params[0],
                    emitter_params[1],
                    emitter_params[2],
                    loc[0],
                    loc[1],
                )
                for loc in measurement_locations
            ]
        ).reshape((2 * len(measurement_locations), 3))

    def measurement_model(self, xem, yem, erp, x, y):
        # return np.array([[np.arctan2(yem-y, xem-x)], [erp_db + self.radar_measurement_coeff_db - 10*np.log10((xem-x)**2 + (yem-y)**2)]])
        return measurement_model(xem, yem, erp, x, y)

    def measurement_jacobian(self, xem, yem, erp, x, y):
        return measurement_jacobian(xem, yem, erp, x, y)
        # d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
        # d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
        # d_h1_d_p_emmitter = 0

        # d_h2_d_x_emmitter = -(20*(xem-x))/(np.log(10)*((xem-x)**2+(yem-y)**2))
        # d_h2_d_y_emmitter = -(20*(yem-y))/(np.log(10)*((yem-y)**2+(xem-x)**2))
        # d_h2_d_p_emmitter = 1

        # return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_estimate_emmitor_params(self):
        return self.estimated_emmiter_params

    def compute_emmitor_estimate_covariance(self, X, y):
        jacobians = self.stack_measurement_jacobian(self.estimated_emmiter_params, y, X)
        combined_measurment_cov = block_diag(*[self.measurement_covariance for i in X])
        emitter_params_cov = np.linalg.inv(
            jacobians.T @ np.linalg.inv(combined_measurment_cov) @ jacobians
        )
        return emitter_params_cov

    def temp(self, x, X_):
        return self.measurement_model(x[0], x[1], x[2], X_[0], X_[1]).reshape((2,))

    def get_params(self, deep=False):
        # return {"position": self.estimated_emmiter_location}
        return {
            "measurement_cov": self.measurement_covariance,
            "radar_measurement_coeff": self.radar_measurement_coeff,
        }


def plotMalhalanobisDistance(pursuerPosition, pursuerPositionCov, ax):
    x = np.linspace(-100, 100, 50)
    y = np.linspace(-100, 100, 50)
    [X, Y] = np.meshgrid(x, y)

    malhalanobisDistance = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            malhalanobisDistance[i, j] = np.linalg.norm(
                (np.array([[X[i, j]], [Y[i, j]]]) - pursuerPosition).T
                @ np.linalg.inv(pursuerPositionCov)
                @ (np.array([[X[i, j]], [Y[i, j]]]) - pursuerPosition)
            )
    c = ax.contourf(X, Y, malhalanobisDistance, levels=[0, 1, 2, 3])
    ax.scatter(pursuerPosition[0], pursuerPosition[1], color="red")


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    X_test = params.create_test_points(params.numTestPoints, params.bounds)

    # estimator = MultipleEmitterOnlineLocationAndPowerEstimator(100, .1, np.eye(2)*.1, [[0,0],[1,1],[2,2]], 1)
    estimator = MultipleEmitterOnlineLocationAndPowerEstimator(
        sensing_range=params.agentSensingRange,
        angle_measurement_std_dev=params.agentAngleMeasurementStdDev,
        measurement_cov=params.measurementCov,
        X_test=X_test,
        radar_measurement_coeff=params.radarMeasurementCoeff,
    )
    nonlinear = NonlinearEstimator(params.measurementCov, params.radarMeasurementCoeff)
    estimator.add_measurement([-3000, 3000], [-np.pi / 4, 10])
    # estimator.add_measurement([0,50*np.sqrt(2)],[-np.pi/2, 10])
    estimator.add_measurement([3000, 3000], [-3 * np.pi / 4, 10])
    nonlinear.fit(
        np.array(estimator.measurement_locations),
        np.array(estimator.measurement_values),
    )
    params = nonlinear.get_estimate_emmitor_params()
    print(params)
    cov = nonlinear.compute_emmitor_estimate_covariance(
        np.array(estimator.measurement_locations),
        np.array(estimator.measurement_values),
    )
    print(cov)
    estimator.plot(ax)
    plotMalhalanobisDistance(params[0:2], cov[0:2, 0:2], ax)
    plt.show()
