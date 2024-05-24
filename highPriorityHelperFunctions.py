import jax.numpy as jnp
from functools import partial
from jax import jit

from bspline.matrix_evaluation import matrix_bspline_derivative_evaluation_for_dataset, matrix_bspline_evaluation_for_dataset



@partial(jit, static_argnums=(2,3))
def create_unclamped_knot_points(t0, tf, numControlPoints,splineOrder):
    internalKnots = jnp.linspace(t0, tf, numControlPoints - 2, endpoint=True)
    h = internalKnots[1] - internalKnots[0]
    knots = jnp.concatenate((jnp.linspace(t0-splineOrder*h,t0-h,splineOrder), internalKnots, jnp.linspace(tf+h,tf+splineOrder*h,splineOrder)))
    
    return knots

@partial(jit, static_argnums=(2,3,4))
def evaluate_spline_derivative(controlPoints, knotPoints,splineOrder, derivativeOrder, numSamplesPerInterval):
    scaleFactor = knotPoints[-splineOrder-1]/(len(knotPoints)-2*splineOrder-1)
    return matrix_bspline_derivative_evaluation_for_dataset(derivativeOrder, scaleFactor, controlPoints.T, knotPoints, numSamplesPerInterval)


@partial(jit, static_argnums=(2,3))
def get_spline_velocity(controlPoints, tf, splineOrder, numSamplesPerInterval):
    print(controlPoints.shape)
    numControlPoints = int(len(controlPoints)/2)
    controlPoints = controlPoints.reshape((numControlPoints,2))
    knotPoints = create_unclamped_knot_points(0, tf, numControlPoints,splineOrder)
    out_d1 = evaluate_spline_derivative(controlPoints,knotPoints,splineOrder,1,numSamplesPerInterval)
    return jnp.linalg.norm(out_d1,axis=1)