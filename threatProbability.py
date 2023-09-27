import numpy as np
import matplotlib.pyplot as plt


def threat_level(distance, max_radar_detection_range, max_SAM_range, max_no_escape_range):
    
    if distance >= max_radar_detection_range:
        return 0
    elif distance >= max_SAM_range and distance < max_radar_detection_range:
        return 0.5 * np.exp(-(distance - max_SAM_range)/(max_radar_detection_range - max_SAM_range))
    elif distance >= max_no_escape_range and distance < max_SAM_range:
        return 2**(-(distance - max_no_escape_range)/(max_SAM_range - max_no_escape_range))
    else:
        return 1

def threat_level_multiple_radar(agent_location, radar_location_list, max_radar_detection_range_list, max_SAM_range_list, max_no_escape_range_list):
    cumulative_threat_level = 1
    for i, radar_location in enumerate(radar_location_list):
        cumulative_threat_level *= (1-threat_level(get_distance(agent_location, radar_location), max_radar_detection_range_list[i], max_SAM_range_list[i], max_no_escape_range_list[i]))
        
    return 1 - cumulative_threat_level


def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1).reshape((2,))-np.array(p2).reshape((2,)))




if __name__ == '__main__':
    bounds = (1200,1200)
    radar_location_1 = (600,600)
    radar_location_2 = (1100,700)
    max_radar_detection_range_1 = 300
    max_SAM_range_1 = 200
    max_no_escape_range_1 = 100
    max_radar_detection_range_2 = 400
    max_SAM_range_2 = 350
    max_no_escape_range_2 = 300

    radar_location_list = [radar_location_1, radar_location_2]
    max_radar_detection_range_list = [max_radar_detection_range_1, max_radar_detection_range_2]
    max_SAM_range_list = [max_SAM_range_1, max_SAM_range_2]
    max_no_escape_range_list = [max_no_escape_range_1, max_no_escape_range_2]

    numTestPoints = 600
    
    x_test = np.linspace(0,bounds[0],numTestPoints)
    y_test = np.linspace(0,bounds[1],numTestPoints)

    [X_test, Y_test] = np.meshgrid(x_test,y_test)
    Threat_Map = np.zeros_like(X_test)
    
    
    for i in range(numTestPoints):
        for j in range(numTestPoints):
            # Threat_Map[i][j] = threat_level(get_distance((X_test[i][j],Y_test[i][j]), radar_location), max_radar_detection_range, max_SAM_range, max_no_escape_range)
            Threat_Map[i][j] = threat_level_multiple_radar((X_test[i][j],Y_test[i][j]), radar_location_list, max_radar_detection_range_list, max_SAM_range_list, max_no_escape_range_list)

    print(Threat_Map.shape)
    plt.figure()
    c = plt.pcolormesh(X_test,Y_test, Threat_Map)
    plt.colorbar(c)
    plt.show()
    
    
    
    dist = np.linspace(0,400,10000)
    threat_l = np.zeros_like(dist)
    for i in range(len(dist)):
        threat_l[i] = threat_level(dist[i], max_radar_detection_range_1, max_SAM_range_1, max_no_escape_range_1)
        
    plt.figure()
    plt.plot(dist, threat_l)
    plt.show()
        

    
    
    