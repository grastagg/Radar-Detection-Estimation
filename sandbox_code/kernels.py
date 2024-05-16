import numpy as np
import matplotlib.pyplot as plt


def radial_kernel(x1, x2, center, sigma_f, length_scale):
    return sigma_f * np.exp(-(np.linalg.norm(x1-center)-np.linalg.norm(x2-center))**2/length_scale)

def vectorized_radial_kernel(x1_vec, x2_vec, center, sigma_f, length_scale):
    x1_vec = np.array(x1_vec)
    x2_vec = np.array(x2_vec)
    center = center.reshape((2,))
    return sigma_f * np.exp(-(np.linalg.norm(x1_vec-center)-np.linalg.norm(x2_vec-center, axis=1))**2/length_scale)
    

    


if __name__ == '__main__':
    num_points = 100
    x_test = np.linspace(0,1200,num_points)
    y_test = np.linspace(0,1200,num_points)

    [X_test, Y_test] = np.meshgrid(x_test, y_test)
    X_test = X_test.reshape((-1,1))
    Y_test = Y_test.reshape((-1,1))
    print(X_test)
    print(Y_test)

    center = np.array([[600,600]])
    sigma_f = 1
    length_scale = 1000

    measurement_location = np.array([[600,10]])
    
    combined_x_y_vec = np.hstack((X_test,Y_test))
    print(combined_x_y_vec)
    kernel_value = vectorized_radial_kernel(combined_x_y_vec, measurement_location, center, sigma_f, length_scale)
    print("TEST", kernel_value.shape)
    
    # for i in range(num_points):
    #     for j in range(num_points):
    #         kernel_value[i,j] = radial_kernel(np.array([[X_test[i,j],Y_test[i,j]]]), measurement_location,center,sigma_f,length_scale)
            
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    c = ax.pcolormesh(X_test.reshape((num_points,num_points)), Y_test.reshape((num_points, num_points)), kernel_value.reshape((num_points, num_points)))
    plt.colorbar(c)
    plt.show()
    
    