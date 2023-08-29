import numpy as np
import matplotlib.pyplot as plt


def radial_kernel(x1, x2, center, sigma_f, length_scale):
    return sigma_f * np.exp(-(np.linalg.norm(x1-center)-np.linalg.norm(x2-center))**2/length_scale)

    


if __name__ == '__main__':
    num_points = 10
    x_test = np.linspace(0,1200,num_points)
    y_test = np.linspace(0,1200,num_points)

    [X_test, Y_test] = np.meshgrid(x_test, y_test)
    kernel_value = np.zeros_like(X_test)
    center = np.array([[600,600]])
    sigma_f = 1
    length_scale = 1000

    measurement_location = np.array([[600,10]])
    
    for i in range(num_points):
        for j in range(num_points):
            kernel_value[i,j] = radial_kernel(np.array([[X_test[i,j],Y_test[i,j]]]), measurement_location,center,sigma_f,length_scale)
            
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    c = ax.pcolormesh(X_test, Y_test, kernel_value)
    plt.colorbar(c)
    plt.show()
    
    