import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

n = 12

# List to hold barycentric coordinates
barycentric_points = []

# Generate points in barycentric coordinates covering the entire triangle
for i in range(n + 1):
    for j in range(n + 1 - i):
        u = i / n
        v = j / n
        w = 1 - u - v
        barycentric_points.append((u, v, w))

# Convert to numpy array and select only the required number of points
barycentric_points = np.array(barycentric_points)
x_sampled = barycentric_points[:, 0]
y_sampled = barycentric_points[:, 1]
z_sampled = barycentric_points[:, 2]

# Define the color value (e.g., V = x + y for coloring)
V = x_sampled + y_sampled
norm = Normalize(vmin=V.min(), vmax=V.max())
colors = cm.viridis(norm(V))
print(x_sampled.shape)
print(y_sampled.shape)
print(z_sampled.shape)

# Create a 3D plot with the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
array_str = " ".join(map(str, x_sampled))
print(array_str)
print()
array_str = " ".join(map(str, y_sampled))
print(array_str)
print()
array_str = " ".join(map(str, z_sampled))
print(array_str)


# Plot the sampled points on the surface
scatter = ax.scatter(x_sampled, y_sampled, z_sampled, facecolors=colors, s=50)

# Add a color bar for the surface
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(V)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
cbar.set_label("Color mapped to x + y")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("100 evenly spaced points on the surface x + y + z = 1")

plt.show()
