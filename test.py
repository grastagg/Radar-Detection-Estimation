import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from itertools import combinations

def compute_bisector(point1, weight1, point2, weight2):
    """
    Compute the bisector between two weighted points.
    """
    p1, p2 = np.array(point1), np.array(point2)
    w1, w2 = weight1, weight2
    
    mid_point = (p1 + p2) / 2
    direction = p2 - p1
    norm_direction = np.array([-direction[1], direction[0]])
    
    t = np.linspace(-5, 5, 400)
    bisector = mid_point[:, None] + t * norm_direction[:, None]
    
    return bisector

def multiplicative_weighted_voronoi(points, weights):
    """
    Construct a multiplicative weighted Voronoi diagram.
    """
    bisectors = []
    for (point1, weight1), (point2, weight2) in combinations(zip(points, weights), 2):
        bisector = compute_bisector(point1, weight1, point2, weight2)
        bisectors.append(bisector)
    
    return bisectors

def plot_weighted_voronoi(points, weights, bisectors):
    """
    Plot the multiplicative weighted Voronoi diagram.
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab20')
    
    for i, point in enumerate(points):
        ax.plot(point[0], point[1], 'o', color=cmap(i), markersize=10)
        ax.text(point[0], point[1], f'{weights[i]:.1f}', color='black', fontsize=12, ha='center')
    
    for bisector in bisectors:
        ax.plot(bisector[0], bisector[1], 'k--', lw=1)
    
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# Example usage
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
weights = np.array([1, 2, 1, 2, 1.5])

bisectors = multiplicative_weighted_voronoi(points, weights)
plot_weighted_voronoi(points, weights, bisectors)
