import numpy as np

# Number of points along each axis
n_points = 10

# Generate evenly spaced points along the parameterization of the surface x + y + z = 1
x = np.linspace(0, 1, n_points)
points = []
for xi in x:
    for yi in np.linspace(0, 1 - xi, n_points):
        zi = 1 - xi - yi
        points.append((xi, yi, zi))

# Convert to numpy array for easier manipulation
points = np.array(points)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Prepare points for plotting
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="b", marker="o", s=20)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
ax.set_title("Evenly Spaced Points on Surface x + y + z = 1")

plt.show()

