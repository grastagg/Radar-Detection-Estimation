import numpy as np
import jax
import jax.numpy as jnp





def measurement_model(xem, yem, erp, x, y,radarMeasurementCoeff):
    return np.array([[np.arctan2(yem-y, xem-x)], [(erp*radarMeasurementCoeff)/((xem-x)**2 + (yem-y)**2)]])

def measurement_jacobian(xem, yem, erp, x, y,radarMeasurementCoeff):
    d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_p_emmitter = 0

    d_h2_d_x_emmitter = -(2*erp*radarMeasurementCoeff*(xem-x))/((xem-x)**2+(yem-y)**2)**2 
    d_h2_d_y_emmitter = -(2*erp*radarMeasurementCoeff*(yem-y))/((yem-y)**2+(xem-x)**2)**2 
    d_h2_d_p_emmitter = radarMeasurementCoeff/((yem-y)**2+(xem-x)**2)

    return np.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])
    
@jax.jit
def measurement_jacobian_jax(xem, yem, erp, x, y,radarMeasurementCoeff):
    d_h1_d_x_emmitter = -(yem-y)/((xem-x)**2*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_y_emmitter = 1/((xem-x)*((yem-y)**2/(xem-x)**2+1))
    d_h1_d_p_emmitter = 0

    d_h2_d_x_emmitter = -(2*erp*radarMeasurementCoeff*(xem-x))/((xem-x)**2+(yem-y)**2)**2 
    d_h2_d_y_emmitter = -(2*erp*radarMeasurementCoeff*(yem-y))/((yem-y)**2+(xem-x)**2)**2 
    d_h2_d_p_emmitter = radarMeasurementCoeff/((yem-y)**2+(xem-x)**2)

    return jnp.array([[d_h1_d_x_emmitter, d_h1_d_y_emmitter, d_h1_d_p_emmitter],[d_h2_d_x_emmitter, d_h2_d_y_emmitter, d_h2_d_p_emmitter]])
