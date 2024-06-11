import numpy as np
import matplotlib.pyplot as plt


from main_helper import create_radar_list
import params
from weighted_voronoi import ground_truth_radar_voronoi_weighted



def compute_bisector_points_to_point(points, weights, point, weight):
    centers = []
    radii = []
    for i in range(len(points)):
        center, radius = compute_bisector_two_points(points[i],point,weights[i],weight)
        centers.append(center)
        radii.append(radius)
    return centers, radii

def compute_bisector_two_points(p1,p2,w1,w2):
    alpha = w1/w2
    center = (p1-alpha**2*p2)/(1-alpha**2)
    radius = alpha/(1-alpha**2)*np.linalg.norm(p1-p2)

    return center, radius

def plot_dist_grid(X_test, points,weights,ax):

    distances = []
    for i in range(len(weights)):
        distances.append(np.linalg.norm(X_test - points[i],axis=1)/weights[i])
    
    distances = np.array(distances)
    Z = np.argmin(distances,axis=0)

    ax.pcolormesh(X_test[:,0].reshape(params.numTestPoints,params.numTestPoints),X_test[:,1].reshape(params.numTestPoints,params.numTestPoints),Z.reshape(params.numTestPoints,params.numTestPoints))

            
def plot_arc(center,radius,theta1= 0, theta2 = 2*np.pi, ax=None):
    theta = np.linspace(theta1,theta2,100)
    # theta = np.unwrap(theta)
    
    print("Theta1:",theta1)
    print("Theta2:",theta2)
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    ax.plot(x,y,c = 'b')

def minimize_angle(theta1):
    while theta1 < 0:
        theta1 += 2*np.pi
    while theta1 > 2*np.pi:
        theta1 -= 2*np.pi
    return theta1

def plot_weighted_voronoi_arcs(arcs,ax):
    # for arc in arcs:
    # arc = arcs[3]
    for arc in arcs:
        p1 = arc[0:2]
        p2 = arc[2:4]
        center = arc[4:6]
        radius = np.linalg.norm(p1-center)
        # theta1 = minimize_angle(np.arctan2(p1[1]-center[1],p1[0]-center[0]))
        # theta2 = minimize_angle(np.arctan2(p2[1]-center[1],p2[0]-center[0]))
        theta1 = np.arctan2(p1[1]-center[1],p1[0]-center[0])
        theta2 = np.arctan2(p2[1]-center[1],p2[0]-center[0])
        if theta2 < theta1:
            theta2 += 2*np.pi
        
        ax.scatter([p1[0],p2[0]],[p1[1],p2[1]],c = 'r')
        # plot_arc(center, radius, np.min([theta1,theta2]), np.max([theta2,theta1]), ax)
        plot_arc(center, radius, theta1,theta2, ax)

def load_weighted_voronoi_segments_from_file(filename):
    data = np.genfromtxt(filename,delimiter=',')
    # sourcePoints = data[:,0:2]
    # targetPoints = data[:,2:4]
    # centers = data[:,4:6]
    return data

def construct_graph_from_arcs(arcs):
    for arc in arcs:
        
    return graph

if __name__ == '__main__':
    filename = "output.txt"

    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
    ground_truth_radar_voronoi_weighted(params.X_test,radarList,ax)
    
    arcs = load_weighted_voronoi_segments_from_file(filename)
    plot_weighted_voronoi_arcs(arcs,ax)
    plt.show()

    

    # generatorPoints = np.array([radar.position for radar in radarList])
    # weights = np.sqrt(np.sqrt(np.array([radar.outputPower*radar.transmitGain*radar.recieveGain for radar in radarList])))
    # print(generatorPoints)
    # print(weights)
    # # order = np.argsort(weights)


    # # centers, radii = compute_bisector_points_to_point(generatorPoints[0:2],weights[0:2],generatorPoints[2],weights[2])

    # fig,ax = plt.subplots()

    # ax.set_aspect('equal')
    # # ax.scatter(generatorPoints[:,0][0:2],generatorPoints[:,1][0:2],c = 'r')
    # # ax.scatter(generatorPoints[:,0][2],generatorPoints[:,1][2],c = 'b')
    # # # plot_dist_grid(params.X_test,generatorPoints[0:2],weights[0:2],ax)
    # # for center,radius in zip(centers,radii):
    # #     plot_arc(center,radius,ax)

    # plt.show()
    