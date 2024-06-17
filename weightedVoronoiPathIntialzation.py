import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import time


from main_helper import create_radar_list
import params
from weighted_voronoi import ground_truth_radar_voronoi_weighted

from weightedVoronoiHelperFunctions import arc_line_segment_intersection
from probabilityOfDetectionMap import ProbabilityOfDetectionMap
from wevo_py import weighted_voronoi_diagram



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

            
def plot_arc(center,radius,theta1= 0, theta2 = 2*np.pi, ax=None,c = 'b'):
    theta = np.linspace(theta1,theta2,100)
    # theta = np.unwrap(theta)
    
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    ax.plot(x,y,c = c)

def plot_weighted_voronoi_arcs(arcs,boundarySegments,ax):
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
    for seg in boundarySegments:
        ax.plot(seg[:,0],seg[:,1],c = 'g')
    # plt.plot([0,0,params.bounds[0],params.bounds[0],0],[0,params.bounds[1],params.bounds[1],0,0],c = 'k')

# def combine_arcs(arcs):
#     arcs = []
#     for arc1 in arcs:
#         for arc2 in arcs:

            

def load_weighted_voronoi_segments_from_file(filename):
    data = np.genfromtxt(filename,delimiter=',')
    # sourcePoints = data[:,0:2]
    # targetPoints = data[:,2:4]
    # centers = data[:,4:6]
    return data



def intersect_arcs_with_boundary(arcs, bounds,ax):
    intersections = []
    arcsToDelete = []
    for i,arc in enumerate(arcs):
        p1 = arc[0:2]
        p2 = arc[2:4]
        center = arc[4:6]
        radius = np.linalg.norm(p1-center)
        
        currentIntersections = []
        currentIntersections += arc_line_segment_intersection(center, radius, p1, p2, [0,0], [0,bounds[1]])
        currentIntersections += arc_line_segment_intersection(center, radius, p1, p2, [0,bounds[1]], [bounds[0],bounds[1]])
        currentIntersections += arc_line_segment_intersection(center, radius, p1, p2, [bounds[0],bounds[1]], [bounds[0],0])
        currentIntersections += arc_line_segment_intersection(center, radius, p1, p2, [bounds[0],0], [0,0])
        p1In = np.all(np.array([p1[0] >= 0, p1[0] <= bounds[0], p1[1] >= 0, p1[1] <= bounds[1]]))
        p2In = np.all(np.array([p2[0] >= 0, p2[0] <= bounds[0], p2[1] >= 0, p2[1] <= bounds[1]]))

        if len(currentIntersections) == 1:
            if p1In:
                arcs[i][2:4] = currentIntersections[0]
            elif p2In:
                arcs[i][0:2] = currentIntersections[0]
        if len(currentIntersections) == 2:
            arcs[i][0:2] = currentIntersections[0]
            arcs[i][2:4] = currentIntersections[1]
        if not (p1In or p2In):
            arcsToDelete.append(i)
        intersections += currentIntersections
        plot = False
        if plot:
            fig,ax = plt.subplots()
            ax.set_aspect('equal')
            ax.plot([0,0,bounds[0],bounds[0],0],[0,bounds[1],bounds[1],0,0])
            theta1 = np.arctan2(arc[1]-arc[5],arc[0]-arc[4])
            theta2 = np.arctan2(arc[3]-arc[5],arc[2]-arc[4])
            if theta2 < theta1:
                theta2 += 2*np.pi
            plot_arc(arc[4:6],np.linalg.norm(arc[0:2]-arc[4:6]),theta1,theta2,ax=ax,c = 'b')
            for inter in currentIntersections:
                ax.scatter(inter[0],inter[1],c = 'r')
            plt.show()
            
    
    for i in reversed(arcsToDelete):
        arcs = np.delete(arcs,i,axis=0)
    
    boundarySegments = []
    intersections = np.array(intersections)
    # leftBoundaryIntersections = np.sort(intersections[intersections[:,0] == 0],axis=0)
    # rightBoundaryIntersections = np.sort(intersections[intersections[:,0] == bounds[0]],axis=0)
    # topBoundaryIntersections = np.sort(intersections[intersections[:,1] == bounds[1]],axis=0)
    # bottomBoundaryIntersections = np.sort(intersections[intersections[:,1] == 0],axis=0)

    leftBoundaryIntersections = np.sort(intersections[np.isclose(intersections[:,0],0)],axis=0)
    rightBoundaryIntersections = np.sort(intersections[np.isclose(intersections[:,0],bounds[0])],axis=0)
    topBoundaryIntersections = np.sort(intersections[np.isclose(intersections[:,1],bounds[1])],axis=0)
    bottomBoundaryIntersections = np.sort(intersections[np.isclose(intersections[:,1],0)],axis=0)
    


    if len(leftBoundaryIntersections) == 0:
        leftBoundaryIntersections = np.array([[0,0],[0,bounds[1]]])
    else:
        for i in range(len(leftBoundaryIntersections)+1):
            if i == 0:
                boundarySegments.append(np.array([[0,0],leftBoundaryIntersections[0]]))
            elif i == len(leftBoundaryIntersections):
                boundarySegments.append(np.array([leftBoundaryIntersections[-1],[0,bounds[1]]]))
            else:
                boundarySegments.append(np.array([leftBoundaryIntersections[i-1],leftBoundaryIntersections[i]]))
    if len(rightBoundaryIntersections) == 0:
        rightBoundaryIntersections = np.array([[bounds[0],0],[bounds[0],bounds[1]]])
    else:
        for i in range(len(rightBoundaryIntersections)+1):
            if i == 0:
                boundarySegments.append(np.array([[bounds[0],0],rightBoundaryIntersections[0]]))
            elif i == len(rightBoundaryIntersections):
                boundarySegments.append(np.array([rightBoundaryIntersections[-1],[bounds[0],bounds[1]]]))
            else:
                boundarySegments.append(np.array([rightBoundaryIntersections[i-1],rightBoundaryIntersections[i]]))
        
    if len(topBoundaryIntersections) == 0:
        topBoundaryIntersections = np.array([[0,bounds[1]],[bounds[0],bounds[1]]])
    else:
        for i in range(len(topBoundaryIntersections)+1):
            if i == 0:
                boundarySegments.append(np.array([[0,bounds[1]],topBoundaryIntersections[0]]))
            elif i == len(topBoundaryIntersections):
                boundarySegments.append(np.array([topBoundaryIntersections[-1],[bounds[0],bounds[1]]]))
            else:
                boundarySegments.append(np.array([topBoundaryIntersections[i-1],topBoundaryIntersections[i]]))
    if len(bottomBoundaryIntersections) == 0:
        bottomBoundaryIntersections = np.array([[0,0],[bounds[0],0]])
    else:
        for i in range(len(bottomBoundaryIntersections)+1):
            if i == 0:
                boundarySegments.append(np.array([[0,0],bottomBoundaryIntersections[0]]))
            elif i == len(bottomBoundaryIntersections):
                boundarySegments.append(np.array([bottomBoundaryIntersections[-1],[bounds[0],0]]))
            else:
                boundarySegments.append(np.array([bottomBoundaryIntersections[i-1],bottomBoundaryIntersections[i]]))
            
    
    


    
    
    
    
    return arcs,boundarySegments
def is_between_angles_radians(start_angle, stop_angle, angle):
  """
  Checks if a third angle lies between a start and stop angle counter-clockwise (radians).

  Args:
      start_angle: The starting angle in radians (0 to 2*pi).
      stop_angle: The stopping angle in radians (0 to 2*pi).
      angle: The angle to check if it lies between start and stop (0 to 2*pi).

  Returns:
      True if the angle lies between start and stop counter-clockwise, False otherwise.
  """

  # Normalize angles to 0-2*pi range
  start_angle = start_angle % (2 * np.pi)
  stop_angle = stop_angle % (2 * np.pi)
  angle = angle % (2 * np.pi)

  # Handle wraparound
  if stop_angle < start_angle:
    return (angle > start_angle or angle < stop_angle)
  else:
    return start_angle < angle < stop_angle

        
def combine_attached_arcs(arcs):
    # For some reason the c++ function splits arcs into two segments, this function will combine them
    for i in reversed(range(len(arcs))):
        for j in reversed(range(i)):
            if np.linalg.norm(arcs[i][4:6] - arcs[j][4:6]) < 1e-6:
                plot = False 
                if plot:
                    fig,ax = plt.subplots()
                    ax.set_aspect('equal')
                    # ax.set_xlim([-1.5e6,1.5e6])
                    # ax.set_ylim([-.5e6,3e6])
                    ax.set_xlim([-10000,40000])
                    ax.set_ylim([-20000,30000])
                    p1 = arcs[i][0:2]
                    p2 = arcs[i][2:4]
                    center = arcs[i][4:6]
                    theta1 = np.arctan2(p1[1]-center[1],p1[0]-center[0])
                    theta2 = np.arctan2(p2[1]-center[1],p2[0]-center[0])
                    if theta2 < theta1:
                        theta2 += 2*np.pi
                    plot_arc(center,np.linalg.norm(p1-center),theta1,theta2,ax=ax,c = 'b')
                    ax.scatter([p1[0],p2[0]],[p1[1],p2[1]],c = 'r')
                    p1 = arcs[j][0:2]
                    p2 = arcs[j][2:4]
                    center = arcs[j][4:6]
                    theta1 = np.arctan2(p1[1]-center[1],p1[0]-center[0])
                    theta2 = np.arctan2(p2[1]-center[1],p2[0]-center[0])
                    if theta2 < theta1:
                        theta2 += 2*np.pi
                    plot_arc(center,np.linalg.norm(p1-center),theta1,theta2,ax=ax,c = 'b')
                    ax.scatter([p1[0],p2[0]],[p1[1],p2[1]],c = 'r')

                theta1 = np.arctan2(arcs[i][1]-arcs[i][5],arcs[i][0]-arcs[i][4])
                theta2 = np.arctan2(arcs[i][3]-arcs[i][5],arcs[i][2]-arcs[i][4])
                theta3 = np.arctan2(arcs[j][1]-arcs[j][5],arcs[j][0]-arcs[j][4])
                theta4 = np.arctan2(arcs[j][3]-arcs[j][5],arcs[j][2]-arcs[j][4])
                thetas = np.array([theta1,theta2,theta3,theta4])
                thetas = np.unwrap(thetas)
                # thetas[thetas < 0] += 2*np.pi

                roundedThetas = np.round(thetas,decimals=4)

                # Step 3: Find unique elements and their counts
                unique_elements, counts = np.unique(roundedThetas, return_counts=True)

                # Step 4: Filter elements that occur exactly once
                unique_elements_single_occurrence = unique_elements[counts == 1]

                # Step 5: Get indices of these unique elements in the original array
                indices = np.array([index for index, element in enumerate(roundedThetas) if element in unique_elements_single_occurrence])
                
                if len(indices) == 2:
                    middleAngle = unique_elements[counts==2]
                    points = np.array([arcs[i][0:2],arcs[i][2:4],arcs[j][0:2],arcs[j][2:4]])
                    minIndex = np.argmin(thetas[indices])
                    maxIndex = np.argmax(thetas[indices])
                    startAngle = thetas[indices[minIndex]]
                    stopAngle = thetas[indices[maxIndex]]
                    if not is_between_angles_radians(startAngle,stopAngle,middleAngle):
                        temp = maxIndex
                        maxIndex = minIndex
                        minIndex = temp
                    arcs[j][0:2] = points[indices[minIndex]]
                    arcs[j][2:4] = points[indices[maxIndex]]

                    
                    
                    

                    arcs = np.delete(arcs,i,axis=0)
                
                    if plot:
                        p1 = arcs[j][0:2]
                        p2 = arcs[j][2:4]
                        center = arcs[j][4:6]
                        theta1 = np.arctan2(p1[1]-center[1],p1[0]-center[0])
                        theta2 = np.arctan2(p2[1]-center[1],p2[0]-center[0])
                        print("theta1: ",theta1)
                        if theta2 < theta1:
                            theta2 += 2*np.pi
                        print("theta2: ",theta2)
                        plot_arc(center,np.linalg.norm(p1-center),theta1,theta2,ax=ax,c = 'g')
                        plt.show()
                    break

                

    return arcs

def get_arc_length(arc):
    p1 = arc[0:2]
    p2 = arc[2:4]
    center = arc[4:6]
    radius = np.linalg.norm(p1-center)
    theta1 = np.arctan2(p1[1]-center[1],p1[0]-center[0])
    theta2 = np.arctan2(p2[1]-center[1],p2[0]-center[0])
    if theta2 < theta1:
        theta2 += 2*np.pi
    return radius*(theta2-theta1)
    

def create_adjacency_matrix_from_arcs_and_bounary(arcs,boundarySegments,ax=None):
    nodes = dict()
    currentNodeNumber = 0
    nodes[currentNodeNumber] = np.array([0,0])
    
    for arc in arcs:
        p1 = arc[0:2]
        p2 = arc[2:4]
        p1Exists = False
        p2Exists = False
        for key in nodes.keys():
            if np.all(np.isclose(p1,nodes[key])):
                p1Exists = True
            if np.all(np.isclose(p2,nodes[key])):
                p2Exists = True
        if not p1Exists:
            currentNodeNumber += 1
            nodes[currentNodeNumber] = p1
        if not p2Exists:
            currentNodeNumber += 1
            nodes[currentNodeNumber] = p2
    
    currentNodeNumber+=1
    nodes[currentNodeNumber] = np.array([params.bounds[0],0])
    currentNodeNumber+=1
    nodes[currentNodeNumber] = np.array([0,params.bounds[0]])
    currentNodeNumber+=1
    nodes[currentNodeNumber] = np.array([params.bounds[0],params.bounds[1]])

    adjacencyMatrix = np.zeros((len(nodes),len(nodes)))
    edges = dict()

    for arc in arcs:
        node1 = None
        node2 = None
        for key in nodes.keys():
            if np.all(np.isclose(arc[0:2],nodes[key])):
                node1 = key
            if np.all(np.isclose(arc[2:4],nodes[key])):
                node2 = key
        arcLength = get_arc_length(arc)
        adjacencyMatrix[node1,node2] = arcLength
        adjacencyMatrix[node2,node1] = arcLength
        edges[(node1,node2)] = {"arc":True,"center":arc[4:6]}
        edges[(node2,node1)] = {"arc":True,"center":arc[4:6]}
    
    for seg in boundarySegments:
        node1 = None
        node2 = None
        for key in nodes.keys():
            if np.all(np.isclose(seg[0],nodes[key])):
                node1 = key
            if np.all(np.isclose(seg[1],nodes[key])):
                node2 = key
        adjacencyMatrix[node1,node2] = np.linalg.norm(seg[0]-seg[1])
        adjacencyMatrix[node2,node1] = np.linalg.norm(seg[0]-seg[1])
        edges[(node1,node2)] = {"arc":False}
        edges[(node2,node1)] = {"arc":False}
            
    return adjacencyMatrix,nodes,edges
    

def create_graph_and_find_shortest_path(adejacenyMatrix,nodes):
    g = ig.Graph.Weighted_Adjacency(adejacenyMatrix.tolist(),mode=ig.ADJ_UNDIRECTED,attr="weight")
    path = g.get_shortest_paths(0,to=len(nodes)-1,weights=g.es["weight"])

    return path

# def evaluate_circular_arc(center,radius,theta1,theta2,point):
def evaluate_arc(center,radius,theta1, theta2, spacing):
    # dTheta = np.arccos((-spacing**2+2*radius**2)/(2*radius**2))
    dTheta = spacing/radius
    numTheta = int((abs(theta2-theta1))/dTheta)
    theta = np.linspace(theta1,theta2,numTheta)
    # theta = np.unwrap(theta)
    
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    return np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

def fill_in_path(path,edges,nodes,spacing = 500):
    # print("Path before filling in: ",path[0])
    newPath = []
    for i in range(len(path[0])-1):
        if edges[(path[0][i],path[0][i+1])]["arc"]:
            p1 = nodes[path[0][i]]
            p2 = nodes[path[0][i+1]]
            center = edges[(path[0][i],path[0][i+1])]["center"]
            theta1 = np.arctan2(p1[1]-center[1],p1[0]-center[0])
            theta2 = np.arctan2(p2[1]-center[1],p2[0]-center[0])
            # if theta2 < theta1:
            #     theta2 += 2*np.pi
            arc = evaluate_arc(center,np.linalg.norm(p1-center),theta1,theta2,spacing)
            newPath.append(arc)
        else:
            numPoints = int(np.linalg.norm(nodes[path[0][i]]-nodes[path[0][i+1]])/spacing)
            points = np.linspace(nodes[path[0][i]],nodes[path[0][i+1]],numPoints)
            newPath.append(points)
            pass
    return np.vstack(newPath)
        
def save_points_and_weights_to_file(radarList,filename):
    generatorPoints = np.array([radar.position for radar in radarList])
    weights = np.sqrt(np.sqrt(np.array([radar.outputPower*radar.transmitGain*radar.recieveGain for radar in radarList])))
    weights *= 1000
    data = np.rint(np.hstack((generatorPoints,weights.reshape(-1,1)))).astype(int)
    np.savetxt(filename,data,delimiter=' ',fmt='%i')
        
def compute_path_weighted_voronoi(radarList,plot =False,ax=None):

    save_points_and_weights_to_file(radarList,"my_input.pnts")

    weighted_voronoi_diagram()
    
    filename = "output.txt"

    arcs = load_weighted_voronoi_segments_from_file(filename)

    arcs = combine_attached_arcs(arcs)


    arcs,boundarySegments = intersect_arcs_with_boundary(arcs,params.bounds,ax)
    adjacencyMatrix,nodes,edges = create_adjacency_matrix_from_arcs_and_bounary(arcs,boundarySegments,ax)
    path = create_graph_and_find_shortest_path(adjacencyMatrix,nodes)

    path = fill_in_path(path,edges,nodes,spacing=50)

    if plot:
        plot_weighted_voronoi_arcs(arcs,boundarySegments,ax)
        ax.scatter(path[:,0],path[:,1],c = 'k')
    
    
    return path
    


if __name__ == '__main__':

    radarList = tuple(create_radar_list(params.radarPositions, params.radarPhases, params.radarAngularRates, params.radarOutputPowerList, params.radarTransmitGainList, params.radarRecieveGainList, params.radarWavelength, params.radarPulseWidth, params.radarSystemTemperature, params.radarProbabilityOfFalseAlarm))
    pdMap = ProbabilityOfDetectionMap(params.X_test,tuple(radarList))
    
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    c = pdMap.plot_mean(ax,plotGroundTruth=True)
    fig.colorbar(c,ax=ax)
    compute_path_weighted_voronoi(radarList=radarList,plot = True,ax=ax)



    
    
    plt.show()

    

    # generatorPoints = np.array([radar.position for radar in radarList])
    # weights = np.sqrt(np.sqrt(np.array([radar.outputPower*radar.transmitGain*radar.recieveGain for radar in radarList])))
    # weights *= 1000
    # print(generatorPoints)
    # print(weights)
    # data = np.rint(np.hstack((generatorPoints,weights.reshape(-1,1)))).astype(int)
    # print(data)
    # np.savetxt("my_input.pnts",data,delimiter=' ',fmt='%i')
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
    