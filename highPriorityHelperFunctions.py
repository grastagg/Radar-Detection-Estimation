import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
from time import time
from scipy.constants import k

from bspline.matrix_evaluation import (
    matrix_bspline_derivative_evaluation_for_dataset,
    matrix_bspline_evaluation_for_dataset,
)

from main_helper import create_radar_list
from probabilityOfDetectionJax import (
    ground_truth_probability_of_detection,
    compute_probability_of_detection_at_points_multiple_radar,
)
import jax
from jax import vmap
import matplotlib.pyplot as plt

# import params


@partial(jit, static_argnums=(2, 3))
def create_unclamped_knot_points(t0, tf, numControlPoints, splineOrder):
    internalKnots = jnp.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = jnp.concatenate(
        (
            jnp.linspace(t0 - splineOrder * h, t0 - h, splineOrder),
            internalKnots,
            jnp.linspace(tf + h, tf + splineOrder * h, splineOrder),
        )
    )

    return knots


@partial(jit, static_argnums=(2, 3, 4))
def evaluate_spline_derivative(
    controlPoints, knotPoints, splineOrder, derivativeOrder, numSamplesPerInterval
):
    scaleFactor = knotPoints[-splineOrder - 1] / (len(knotPoints) - 2 * splineOrder - 1)
    return matrix_bspline_derivative_evaluation_for_dataset(
        derivativeOrder, scaleFactor, controlPoints.T, knotPoints, numSamplesPerInterval
    )


@partial(jit, static_argnums=(1, 2))
def evaluate_spline(controlPoints, knotPoints, numSamplesPerInterval):
    return matrix_bspline_evaluation_for_dataset(
        controlPoints.T, knotPoints, numSamplesPerInterval
    )


@partial(jit, static_argnums=(2, 3))
def get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval):
    print(controlPoints.shape)
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    return jnp.linalg.norm(out_d1, axis=1)


@partial(jit, static_argnums=(2, 3, 4, 5))
def get_pd_along_spline(
    controlpoints,
    tf,
    radarlist,
    numcontrolpoints,
    splineorder,
    numsamplesperinterval,
    radarwavelengthpriormean,
    agentradarcrosssection,
    radarpulsewidth,
    radarsystemtemperaturepriormean,
    radarprobabilityoffalsealarmpriormean,
):
    controlpoints = controlpoints.reshape((numcontrolpoints, 2))
    knotpoints = create_unclamped_knot_points(0, tf, numcontrolpoints, splineorder)
    pos = evaluate_spline(controlpoints, knotpoints, numsamplesperinterval)
    pd = ground_truth_probability_of_detection(
        pos,
        radarlist,
        radarwavelengthpriormean,
        agentradarcrosssection,
        radarpulsewidth,
        radarsystemtemperaturepriormean,
        radarprobabilityoffalsealarmpriormean,
    )
    return pd


@partial(jit, static_argnums=(2, 3))
def get_spline_turn_rate(controlPoints, tf, splineOrder, numSamplesPerInterval):
    numControlPoints = int(len(controlPoints) / 2)
    controlPoints = controlPoints.reshape((numControlPoints, 2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints, splineOrder)
    out_d1 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 1, numSamplesPerInterval
    )
    out_d2 = evaluate_spline_derivative(
        controlPoints, knotPoints, splineOrder, 2, numSamplesPerInterval
    )
    v = jnp.linalg.norm(out_d1, axis=1)
    u = jnp.cross(out_d1, out_d2) / (v**2)
    return u


@jit
def dist_of_points_to_line_segment(p1, p2, points):  # p3 is the point
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    # px = x2-x1
    # py = y2-y1
    diff = p2 - p1
    # px = x2-x1
    # py = y2-y1

    norm = jnp.sum(diff**2)
    # norm = px*px + py*py

    # u =  ((x3 - x1) * diff[0] + (y3 - y1) * diff[1]) / norm
    u = jnp.dot(points - p1, diff) / norm
    u = u.reshape((-1, 1))

    u = jnp.clip(u, 0, 1)

    # x = x1 + u * diff[0]
    # y = y1 + u * diff[1]
    p = p1 + u * diff

    # dx = x - x3
    # dy = y - y3

    dp = p - points

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    # dist = (dx*dx + dy*dy)
    dist = jnp.linalg.norm(dp, axis=1)

    return dist


# @partial(jit, static_argnums=(2,3,4))
# def get_prob_pd_less_than_threshold_along_spline(controlpoints, tf, numcontrolpoints, splineorder, numsamplesperinterval, estimatedRadarParamsList, estimatedRadarParamsCovList, pdThreshold, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar):
def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(
        -z * z
        - 1.26551223
        + t
        * (
            1.00002368
            + t
            * (
                0.37409196
                + t
                * (
                    0.09678418
                    + t
                    * (
                        -0.18628806
                        + t
                        * (
                            0.27886807
                            + t
                            * (
                                -1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                            )
                        )
                    )
                )
            )
        )
    )

    r = np.array(r)
    r[x < 0] = 2 - r[x < 0]
    # r = r.at[x<0].set(2-r.at[x<0])
    return r


def normcdf(x, mu, sigma):
    t = x - mu
    y = 0.5 * erfcc(-t / (sigma * np.sqrt(2.0)))
    y = np.array(y)
    y[y > 1.0] = 1.0
    return y


# @jit
# def get_prob_pd_less_than_threshold(pos, estimatedRadarParamsList, estimatedRadarParamsCovList, pdThreshold, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar):
#     pdMean, pdCov = compute_probability_of_detection_at_points_multiple_radar(pos, estimatedRadarParamsList, estimatedRadarParamsCovList, radarrecievegainpriormean, radarrecievegainpriorvar, radarwavelengthpriormean, radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth, radarpulsewidthVar, radarsystemtemperaturepriormean, radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean, radarprobabilityoffalsealarmpriorvar)
#     # pdMean, pdCov = compute_probability_of_detection_at_points_multiple_radar(pos, estimatedRadarParamsList, estimatedRadarParamsCovList, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar)
#     # jax.debug.print("cov: {x}", x = estimatedRadarParamsCovList)
#     jax.debug.print("cov: {x}", x = pdCov)
#     probPdLessThanThreshold = jax.scipy.stats.norm.cdf(pdThreshold, pdMean, jnp.sqrt(pdCov))
#     # probPdLessThanThreshold = normcdf(pdThreshold, pdMean, jnp.sqrt(pdCov))
#     return probPdLessThanThreshold


@jit
def get_prob_pd_less_than_threshold(
    X_test,
    estimatedRadarParams,
    estimatedRadarParamsCov,
    pdThreshold,
    radarRecieveGain,
    radarRecieveGainVar,
    radarWavelength,
    radarWavelengthVar,
    agentRadarCrossSection,
    radarPulseWidth,
    radarPulseWidthVar,
    radarSystemTemperature,
    radarSystemTemperatureVar,
    radarProbabilityOfFalseAlarm,
    radarProbabilityOfFalseAlarmVar,
):
    pdMean, pdCov = compute_probability_of_detection_at_points_multiple_radar(
        X_test,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        radarRecieveGain,
        radarRecieveGainVar,
        radarWavelength,
        radarWavelengthVar,
        agentRadarCrossSection,
        radarPulseWidth,
        radarPulseWidthVar,
        radarSystemTemperature,
        radarSystemTemperatureVar,
        radarProbabilityOfFalseAlarm,
        radarProbabilityOfFalseAlarmVar,
    )
    # print("test pd cov", pdCov)

    pdSigma = jnp.sqrt(pdCov)

    # probabilityPdLessThanThreshold = normcdf(pdThreshold,pdMean,pdSigma)
    probabilityPdLessThanThreshold = jax.scipy.stats.norm.cdf(
        pdThreshold, pdMean, pdSigma
    )

    # return probabilityPdLessThanThreshold > likleyhoodThreshold
    return probabilityPdLessThanThreshold


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def get_prob_along_spline(
    controlpoints,
    tf,
    estimatedRadarParams,
    estimatedRadarParamsCov,
    pdThreshold,
    numcontrolpoints,
    splineorder,
    numsamplesperinterval,
    radarReceiveGain,
    radarReceiveGainVar,
    radarwavelengthpriormean,
    radarwavelengthvar,
    agentradarcrosssection,
    radarpulsewidth,
    radarpulsewidthvar,
    radarsystemtemperaturepriormean,
    radarsystemtemperaturepriorvar,
    radarprobabilityoffalsealarmpriormean,
    radarprobabilityoffalsealarmpriorvar,
):
    controlpoints = controlpoints.reshape((numcontrolpoints, 2))
    knotpoints = create_unclamped_knot_points(0, tf, numcontrolpoints, splineorder)
    pos = evaluate_spline(controlpoints, knotpoints, numsamplesperinterval)
    # prob = get_prob_pd_less_than_threshold(pos, estimatedRadarParams, estimatedRadarParamsCov, pdThreshold, radarwavelengthpriormean, agentradarcrosssection, radarpulsewidth, radarsystemtemperaturepriormean, radarprobabilityoffalsealarmpriormean)
    prob = get_prob_pd_less_than_threshold(
        pos,
        estimatedRadarParams,
        estimatedRadarParamsCov,
        pdThreshold,
        radarReceiveGain,
        radarReceiveGainVar,
        radarwavelengthpriormean,
        radarwavelengthvar,
        agentradarcrosssection,
        radarpulsewidth,
        radarpulsewidthvar,
        radarsystemtemperaturepriormean,
        radarsystemtemperaturepriorvar,
        radarprobabilityoffalsealarmpriormean,
        radarprobabilityoffalsealarmpriorvar,
    )
    # prob = get_prob_pd_less_than_threshold(pos, estimatedRadarParamsList, estimatedRadarParamsCovList, pdThreshold, radarrecievegainpriormean,radarrecievegainpriorvar, radarwavelengthpriormean,radarwavelengthpriormeanvar, agentradarcrosssection, radarpulsewidth,radarpulsewidthVar, radarsystemtemperaturepriormean,radarsystemtemperaturepriorvar, radarprobabilityoffalsealarmpriormean,radarprobabilityoffalsealarmpriorvar):
    return prob


@jit
def radar_intercept_snr(
    radarPos,
    agentPos,
    radarTransmitGain,
    radarTransmitPower,
    agentRecieveGain,
    radarWavelength,
    systemTemp,
    radarPulseWidth,
):
    R = jnp.linalg.norm(radarPos - agentPos)
    SNR = (
        radarTransmitPower
        * radarTransmitGain
        * agentRecieveGain
        * radarWavelength**2
        * radarPulseWidth
    ) / ((4 * jnp.pi) ** 2 * R**2 * systemTemp * k * 300000000)
    return SNR


@jit
def radar_probability_of_intercept(
    radarPos,
    agentPos,
    radarTransmitGain,
    radarTransmitPower,
    agentRecieveGain,
    radarWavelength,
    systemTemp,
    interceptProbabilityOfFalseAlarm,
    radarPulseWidth,
):
    SNR = radar_intercept_snr(
        radarPos,
        agentPos,
        radarTransmitGain,
        radarTransmitPower,
        agentRecieveGain,
        radarWavelength,
        systemTemp,
        radarPulseWidth,
    )
    probabilityOfIntercept = jnp.exp(
        (jnp.log(interceptProbabilityOfFalseAlarm)) / (1 + SNR)
    )
    return probabilityOfIntercept


@jit
def probability_radar_at_point_given_path_history(
    point,
    pathHistory,
    radarTransmitGain,
    radarTransmitPower,
    agentRecieveGain,
    radarWavelength,
    systemTemp,
    interceptProbabilityOfFalseAlarm,
    radarPulseWidth,
):
    # Define a vectorized version of the radar_probability_of_intercept function
    vectorized_intercept_prob = vmap(
        radar_probability_of_intercept,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),  # pathPoint is mapped, others are broadcasted
    )

    # Calculate the intercept probabilities for all points in the path history
    interceptProbabilities = vectorized_intercept_prob(
        pathHistory,
        point,
        radarTransmitGain,
        radarTransmitPower,
        agentRecieveGain,
        radarWavelength,
        systemTemp,
        interceptProbabilityOfFalseAlarm,
        radarPulseWidth,
    )

    # Compute the overall probability of avoiding detection
    radarNoInterceptProbGivenRadarAtx = jnp.prod(1 - interceptProbabilities)
    radarNoIntereceptProbGivenNoRadarAtx = (
        1 - interceptProbabilityOfFalseAlarm
    ) ** len(pathHistory)
    probNoIntercept = (
        radarNoInterceptProbGivenRadarAtx * 0.5
        + radarNoIntereceptProbGivenNoRadarAtx * 0.5
    )

    # Return the overall probability of being detected
    return radarNoInterceptProbGivenRadarAtx * 0.5 / probNoIntercept


# @jit
# def probability_radar_at_point_given_path_history(point, pathHistory, radarTransmitGain, radarTransmitPower, agentRecieveGain, radarWavelength, systemTemp, interceptProbabilityOfFalseAlarm, radarPulseWidth):
#     # Compute the probability of intercept for each radar in the path history

#     radarAtXProb = 1.0
#     for pathPoint in pathHistory:
#         # Compute the distance between the path point and the current point
#         distance = jnp.linalg.norm(pathPoint - point)
#         # Compute the probability of intercept for the radar at the path point
#         interceptProbability = radar_probability_of_intercept(pathPoint, point, radarTransmitGain, radarTransmitPower, agentRecieveGain, radarWavelength, systemTemp, interceptProbabilityOfFalseAlarm,radarPulseWidth)
#         radarAtXProb *= (1-interceptProbability)

#     return radarAtXProb


@jit
def path_safety_prob(
    allAgentPathHistory,
    potentialPathPoints,
    radarTransmitGain,
    radarTransmitPower,
    agentRecieveGain,
    radarWavelength,
    systemTemp,
    interceptProbabilityOfFalseAlarm,
    radarPulseWidth,
):
    # Vectorize the probability_radar_at_point_given_path_history function across potentialPathPoints
    vectorized_prob_radar = vmap(
        probability_radar_at_point_given_path_history,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),  # Vectorize across potentialPathPoints
    )

    # Compute the probability of intercept for each radar at each potential path point
    radarAtXProbs = vectorized_prob_radar(
        potentialPathPoints,
        allAgentPathHistory,
        radarTransmitGain,
        radarTransmitPower,
        agentRecieveGain,
        radarWavelength,
        systemTemp,
        interceptProbabilityOfFalseAlarm,
        radarPulseWidth,
    )

    return radarAtXProbs


# def path_safety_prob(allAgentPahtHistory, potentialPathPoints, radarTransmitGain, radarTransmitPower, agentRecieveGain, radarWavelength, systemTemp, interceptProbabilityOfFalseAlarm):
#     # Compute the probability of intercept for each radar at each potential path point
#     radarAtXProbs = jnp.array([probability_radar_at_point_given_path_history(point, allAgentPahtHistory, radarTransmitGain, radarTransmitPower, agentRecieveGain, radarWavelength, systemTemp, interceptProbabilityOfFalseAlarm) for point in potentialPathPoints])
#     return radarAtXProbs


def test_radar_probability_of_interecept(allAgentPathHistory):
    points = params.X_test
    # probIntercept = np.zeros(len(points))

    # for i,point in enumerate(points):
    #     probIntercept[i] = probability_radar_at_point_given_path_history(point, allAgentPathHistory, params.radarTransmitGain, params.radarOutputPower, params.agentELINTAnteneaGain, params.radarWavelength, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm,params.radarPulseWidth)

    probIntercept = path_safety_prob(
        allAgentPathHistory,
        points,
        params.radarTransmitGain,
        params.radarOutputPower,
        params.agentELINTAnteneaGain,
        params.radarWavelength,
        params.radarSystemTemperature,
        params.radarProbabilityOfFalseAlarm,
        params.radarPulseWidth,
    )
    print("max prob", np.max(probIntercept))

    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        points[:, 0].reshape(params.numTestPoints, params.numTestPoints),
        points[:, 1].reshape(params.numTestPoints, params.numTestPoints),
        probIntercept.reshape(params.numTestPoints, params.numTestPoints),
        vmin=0,
        vmax=0.5,
    )
    # c = ax.contourf(points[:,0].reshape(params.numTestPoints,params.numTestPoints),points[:,1].reshape(params.numTestPoints,params.numTestPoints),probIntercept.reshape(params.numTestPoints,params.numTestPoints),levels = [0,.1,.2,.3,.4,.5])
    ax.scatter(allAgentPathHistory[:, 0], allAgentPathHistory[:, 1], c="r")
    ax.set_aspect("equal")
    fig.colorbar(c, ax=ax)
    plt.show()

    # for point in points


@jit
def path_safety(allAgentPathHistory, next_measurements, lengthScale):
    # Compute the pairwise distances between all paths in history and all next measurements
    distances = jnp.linalg.norm(
        allAgentPathHistory[:, None, :] - next_measurements[None, :, :], axis=2
    )
    # return jnp.min(distances, axis=0)

    # Apply the kernel function
    kernel_values = jnp.exp(-distances / 5000)

    # Return the negative sum of the kernel values for each measurement
    return jnp.sum(kernel_values, axis=0)


if __name__ == "__main__":
    dataFilePath = "saved_data/11024122/"
    import params

    # dataIndex = 10
    # numFiles = 5758
    numFiles = 2006
    dataIndex = 2006
    agent1PathHistory = np.genfromtxt(
        dataFilePath + "/agent1PathHistory.txt", delimiter=","
    )
    numPathHistory = int(dataIndex / numFiles * len(agent1PathHistory))
    pathHistoryList = [
        np.genfromtxt(dataFilePath + "/agent0PathHistory.txt", delimiter=",")[
            0:numPathHistory
        ],
        np.genfromtxt(dataFilePath + "/agent1PathHistory.txt", delimiter=",")[
            0:numPathHistory
        ],
        np.genfromtxt(dataFilePath + "/agent2PathHistory.txt", delimiter=",")[
            0:numPathHistory
        ],
    ]
    combinedPathHistory = []
    for pathHistory in pathHistoryList:
        combinedPathHistory.extend(pathHistory)
    combinedPathHistory = np.array(combinedPathHistory)
    test_radar_probability_of_interecept(combinedPathHistory)

    numFiles = 8044
    dataIndex = 1000
    agent1PathHistory = np.genfromtxt(
        dataFilePath + "/agent1PathHistory.txt", delimiter=","
    )
    numPathHistory = int(dataIndex / numFiles * len(agent1PathHistory))
    pathHistoryList = [
        np.genfromtxt(dataFilePath + "/agent0PathHistory.txt", delimiter=",")[
            0:numPathHistory
        ],
        np.genfromtxt(dataFilePath + "/agent1PathHistory.txt", delimiter=",")[
            0:numPathHistory
        ],
        np.genfromtxt(dataFilePath + "/agent2PathHistory.txt", delimiter=",")[
            0:numPathHistory
        ],
    ]
    combinedPathHistory = []
    for pathHistory in pathHistoryList:
        combinedPathHistory.extend(pathHistory)
    combinedPathHistory = np.array(combinedPathHistory)
    test_radar_probability_of_interecept(combinedPathHistory)

