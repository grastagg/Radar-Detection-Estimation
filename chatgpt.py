import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt

points = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]])
vor = Voronoi(points)

fig,ax = plt.subplots()
voronoi_plot_2d(vor,ax)

print("vor.points",vor.points)
print("vor.point_region",vor.point_region)
print("vor.regions",vor.regions)

for i,point in enumerate(vor.vertices):
    ax.text(point[0],point[1],str(i))

for i in range(len(points)):
    point = points[i]
    ax.text(point[0],point[1],str(i))
    ax.scatter(point[0],point[1])
    
# for i in vor.point_region:
    
#     point = vor.points[i-1]
#     # ax.scatter(point[0],point[1])
#     ax.text(point[0],point[1],str(i))
plt.show()
