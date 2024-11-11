import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from measurement_model import (
    measurement_jacobian_jax,
    measurement_jacobian,
    measurement_model,
)
from highPriorityHelperFunctions import (
    probability_radar_at_point_given_path_history,
    path_safety_prob,
)


@jit
def distance_from_goal(x0, y0, x1, y1):
    return jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def get_agent_future_path_waypoint(waypoint, pos, agentSpeed, agentPathHistorydt):
    dist = jnp.linalg.norm(waypoint - pos)
    time = dist / agentSpeed
    num_future_points = jnp.floor(time / agentPathHistorydt).astype(int)
    points = jnp.linspace(pos, waypoint, num_future_points)
    return points


@jax.jit
def distance_from_line(x0, y0, x1, y1, x2, y2):
    return jnp.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / jnp.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )


@jit
def next_measurement_covariance(
    pos,
    estimatedRadarParamsList,
    estimatedRadarCovariancesList,
    measurementCov,
    radarMeasurementCoeff,
):
    x = pos[0]
    y = pos[1]

    def compute_single_cov(estimatedRadarParams, estimatedRadarCovariance):
        estimatedRadarCovariance = estimatedRadarCovariance
        estimatedRadarParams = estimatedRadarParams
        x_em = estimatedRadarParams[0]
        y_em = estimatedRadarParams[1]
        erp = estimatedRadarParams[2]

        # JAX-compatible measurement Jacobian function
        H = measurement_jacobian_jax(x_em, y_em, erp, x, y, radarMeasurementCoeff)

        R = measurementCov

        # Kalman gain calculation
        K = (
            estimatedRadarCovariance
            @ H.T
            @ jnp.linalg.inv(H @ estimatedRadarCovariance @ H.T + R)
        )

        # Update covariance using the Kalman gain
        nextCovariance = (jnp.eye(3) - K @ H) @ estimatedRadarCovariance
        return nextCovariance

    # Vectorize the covariance computation for all radar parameters and covariances
    nextCovariances = vmap(compute_single_cov, in_axes=(0, 0))(
        estimatedRadarParamsList, estimatedRadarCovariancesList
    )

    return nextCovariances


import jax
import jax.numpy as jnp


def compute_objective_jax(
    pos,
    estimatedRadarParams,
    estimatedRadarCovariances,
    x1,
    y1,
    x2,
    y2,
    allAgentPathHistory_temp,
    agentPos,
    agentSpeed,
    agentPathHistorydt,
    measurementCov,
    radarMeasurementCoeff,
    radarTransmitGain,
    radarOutputPower,
    agentELINTAnteneaGain,
    radarWavelength,
    radarSystemTemperature,
    radarProbabilityOfFalseAlarm,
    radarPulseWidth,
    nextCovarianceWeight,
    nextCovarianceScale,
    seperationWeight,
    seperationScale,
    distFromStraitWeight,
    distFromStraitScale,
):
    # Compute the distance between pos and agentPos
    dist_to_agent = jnp.linalg.norm(jnp.array(pos) - jnp.array(agentPos)[0:2])
    dist_to_radar = jnp.array(
        [
            jnp.linalg.norm(jnp.array(pos) - radarPos[0:2])
            for radarPos in estimatedRadarParams
        ]
    )
    minDistToRadar = jnp.min(dist_to_radar)

    # Use jax.lax.cond to handle conditional logic
    distanceThresholdAgent = 1000
    distanceThresholdRadar = 0
    condition = jnp.logical_or(
        dist_to_agent < distanceThresholdAgent, minDistToRadar < distanceThresholdRadar
    )
    obj = jax.lax.cond(
        condition,  # condition: is the distance below the threshold?
        lambda _: jnp.inf,  # if true: return infinity
        lambda _: compute_obj(
            pos,  # if false: compute the objective normally
            estimatedRadarParams,
            estimatedRadarCovariances,
            x1,
            y1,
            x2,
            y2,
            allAgentPathHistory_temp,
            agentPos,
            agentSpeed,
            agentPathHistorydt,
            measurementCov,
            radarMeasurementCoeff,
            radarTransmitGain,
            radarOutputPower,
            agentELINTAnteneaGain,
            radarWavelength,
            radarSystemTemperature,
            radarProbabilityOfFalseAlarm,
            radarPulseWidth,
            nextCovarianceWeight,
            nextCovarianceScale,
            seperationWeight,
            seperationScale,
            distFromStraitWeight,
            distFromStraitScale,
        ),
        None,
    )  # no extra arguments passed to the lambda functions

    return obj


def compute_obj(
    pos,
    estimatedRadarParams,
    estimatedRadarCovariances,
    x1,
    y1,
    x2,
    y2,
    allAgentPathHistory_temp,
    agentPos,
    agentSpeed,
    agentPathHistorydt,
    measurementCov,
    radarMeasurementCoeff,
    radarTransmitGain,
    radarOutputPower,
    agentELINTAnteneaGain,
    radarWavelength,
    radarSystemTemperature,
    radarProbabilityOfFalseAlarm,
    radarPulseWidth,
    nextCovarianceWeight,
    nextCovarianceScale,
    seperationWeight,
    seperationScale,
    distFromStraitWeight,
    distFromStraitScale,
):
    # Vectorized covariance calculation for all radar emitters
    nextCovariances = next_measurement_covariance(
        pos,
        estimatedRadarParams,
        estimatedRadarCovariances,
        measurementCov,
        radarMeasurementCoeff,
    )

    # Compute the determinant of each covariance matrix
    covDets = jnp.linalg.det(nextCovariances)

    # Objective: take the mean determinant of all covariance matrices
    covObj = jnp.mean(covDets)

    # Distance from the goal
    dist = distance_from_goal(pos[0], pos[1], x2, y2)

    # Calculate the agent's future path
    num_future_points = 10
    futurePath_temp = jnp.linspace(jnp.array(agentPos[0:2]), pos, num_future_points)

    # Calculate the safety probability for the path
    explore = path_safety_prob(
        allAgentPathHistory_temp,
        futurePath_temp,
        radarTransmitGain,
        radarOutputPower,
        agentELINTAnteneaGain,
        radarWavelength,
        radarSystemTemperature,
        radarProbabilityOfFalseAlarm,
        radarPulseWidth,
    )

    exploreMean = jnp.mean(explore)

    # Objective function calculation
    obj = (
        nextCovarianceWeight * covObj / nextCovarianceScale
        - seperationWeight * exploreMean / 0.5
        + distFromStraitWeight * dist / distFromStraitScale
    )

    return obj


# # @jit
# def compute_objective_jax(
#     pos,
#     estimatedRadarParams,
#     estimatedRadarCovariances,
#     x1,
#     y1,
#     x2,
#     y2,
#     allAgentPathHistory_temp,
#     agentPos,
#     agentSpeed,
#     agentPathHistorydt,
#     measurementCov,
#     radarMeasurementCoeff,
#     radarTransmitGain,
#     radarOutputPower,
#     agentELINTAnteneaGain,
#     radarWavelength,
#     radarSystemTemperature,
#     radarProbabilityOfFalseAlarm,
#     radarPulseWidth,
#     nextCovarianceWeight,
#     nextCovarianceScale,
#     seperationWeight,
#     seperationScale,
#     distFromStraitWeight,
#     distFromStraitScale,
# ):
#     # Extract current test position
#     dist_to_agent = jnp.linalg.norm(jnp.array(agentPos)[0:2] - pos)
#     print("dist_to_agent", dist_to_agent)
#     if jnp.any(dist_to_agent < 100):
#         return jnp.inf
#
#     # Vectorized covariance calculation for all radar emitters
#     nextCovariances = next_measurement_covariance(
#         pos,
#         estimatedRadarParams,
#         estimatedRadarCovariances,
#         measurementCov,
#         radarMeasurementCoeff,
#     )
#
#     # Compute the determinant of each covariance matrix
#     covDets = jnp.linalg.det(nextCovariances)
#
#     # Objective: take the mean determinant of all covariance matrices
#     covObj = jnp.mean(covDets)
#
#     # Distance from the goal
#     dist = distance_from_goal(pos[0], pos[1], x2, y2)
#     # dist = distance_from_line(pos[0], pos[1], x1, y1, x2, y2)
#
#     # Calculate the agent's future path
#     num_future_points = 10
#     futurePath_temp = jnp.linspace(jnp.array(agentPos[0:2]), pos, num_future_points)
#     # futurePath_temp = get_agent_future_path_waypoint(
#     #     pos,
#     #     jnp.array(agentPos[0:2]),
#     #     agentSpeed,
#     #     agentPathHistorydt,
#     # )
#
#     # Calculate the safety probability for the path
#     explore = path_safety_prob(
#         allAgentPathHistory_temp,
#         futurePath_temp,
#         radarTransmitGain,
#         radarOutputPower,
#         agentELINTAnteneaGain,
#         radarWavelength,
#         radarSystemTemperature,
#         radarProbabilityOfFalseAlarm,
#         radarPulseWidth,
#     )
#
#     exploreMean = jnp.mean(explore)
#
#     # Objective function calculation
#     obj = (
#         nextCovarianceWeight * covObj / nextCovarianceScale
#         - seperationWeight * exploreMean / 0.5
#         + distFromStraitWeight * dist / distFromStraitScale
#     )
#
#     return obj


# Vectorized version using vmap
compute_objective_jax_vectorized = jax.vmap(
    compute_objective_jax,
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
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
)


def compute_objective_vec(
    pos,
    estimatedRadarParams,
    estimatedRadarCovariances,
    x1,
    y1,
    x2,
    y2,
    allAgentPathHistory_temp,
    agentPos,
    agentSpeed,
    agentPathHistorydt,
    measurementCov,
    radarMeasurementCoeff,
    radarTransmitGain,
    radarOutputPower,
    agentELINTAnteneaGain,
    radarWavelength,
    radarSystemTemperature,
    radarProbabilityOfFalseAlarm,
    radarPulseWidth,
    nextCovarianceWeight,
    nextCovarianceScale,
    seperationWeight,
    seperationScale,
    distFromStraitWeight,
    distFromStraitScale,
):
    return compute_objective_jax_vectorized(
        pos,
        estimatedRadarParams,
        estimatedRadarCovariances,
        x1,
        y1,
        x2,
        y2,
        allAgentPathHistory_temp,
        agentPos,
        agentSpeed,
        agentPathHistorydt,
        measurementCov,
        radarMeasurementCoeff,
        radarTransmitGain,
        radarOutputPower,
        agentELINTAnteneaGain,
        radarWavelength,
        radarSystemTemperature,
        radarProbabilityOfFalseAlarm,
        radarPulseWidth,
        nextCovarianceWeight,
        nextCovarianceScale,
        seperationWeight,
        seperationScale,
        distFromStraitWeight,
        distFromStraitScale,
    )
