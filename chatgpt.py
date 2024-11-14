import matplotlib.pyplot as plt
import ternary
import matplotlib.cm as cm
import numpy as np

# Sample data (three components that sum to 1) and a fourth variable for color
data = [
    (0.2, 0.3, 0.5, 0.7),
    (0.1, 0.7, 0.2, 0.3),
    (0.4, 0.4, 0.2, 0.9),
    (0.3, 0.3, 0.4, 0.4),
]

# Verify that each tuple (x1, x2, x3) sums to 1, or normalize if needed
coordinates = [(x1, x2, x3) for x1, x2, x3, _ in data]
coordinates = [
    (x1 / (x1 + x2 + x3), x2 / (x1 + x2 + x3), x3 / (x1 + x2 + x3))
    for x1, x2, x3 in coordinates
]
color_values = [c for _, _, _, c in data]

# Normalize the color values to the range [0, 1] for the colormap
norm = plt.Normalize(min(color_values), max(color_values))
cmap = cm.viridis  # Choose a colormap (e.g., viridis, plasma, etc.)

# Initialize the ternary plot with a scale of 1 (since x1 + x2 + x3 = 1)
scale = 1
fig, tax = ternary.figure(scale=scale)
tax.boundary(linewidth=2.0)
tax.gridlines(color="blue", multiple=0.1)

# Plot each point in the data with color mapped to the fourth variable
for point, color in zip(coordinates, color_values):
    print("point: ", point)
    tax.scatter([point], marker="o", color=cmap(norm(color)), s=100)

# Set labels for each corner of the ternary plot
tax.left_axis_label("x3", fontsize=12)
tax.right_axis_label("x2", fontsize=12)
tax.bottom_axis_label("x1", fontsize=12)

# Adjust ticks to display correctly
tax.ticks(axis="lbr", multiple=0.1, linewidth=1, tick_formats="%.1f")

# Add a colorbar to indicate the scale of the fourth variable
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation="vertical")
cbar.set_label("Fourth Variable", fontsize=12)

# Set plot title and show
plt.title("Barycentric Coordinates Plot with Color Mapping")
tax.clear_matplotlib_ticks()
plt.show()

