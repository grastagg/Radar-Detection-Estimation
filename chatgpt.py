import numpy as np
import matplotlib.pyplot as plt

def circle_line_intersection(center, radius, p1, p2):
    cx, cy = center
    radius_squared = radius ** 2
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    dr_squared = dx**2 + dy**2
    D = x1*y2 - x2*y1

    discriminant = radius_squared * dr_squared - D**2
    if discriminant < 0:
        return []  # No intersection

    sqrt_discriminant = np.sqrt(discriminant)
    sign_dy = np.sign(dy) if dy != 0 else 1

    ix1 = (D*dy + sign_dy * dx * sqrt_discriminant) / dr_squared + cx
    iy1 = (-D*dx + abs(dy) * sqrt_discriminant) / dr_squared + cy

    ix2 = (D*dy - sign_dy * dx * sqrt_discriminant) / dr_squared + cx
    iy2 = (-D*dx - abs(dy) * sqrt_discriminant) / dr_squared + cy

    return [(ix1, iy1), (ix2, iy2)]

def is_point_on_arc(point, center, p1, p2):
    angle_p1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    angle_p2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    angle_point = np.arctan2(point[1] - center[1], point[0] - center[0])

    if angle_p1 < 0:
        angle_p1 += 2 * np.pi
    if angle_p2 < 0:
        angle_p2 += 2 * np.pi
    if angle_point < 0:
        angle_point += 2 * np.pi

    if angle_p1 > angle_p2:
        return angle_point >= angle_p1 or angle_point <= angle_p2
    else:
        return angle_p1 <= angle_point <= angle_p2

def is_point_on_segment(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Check if point is within the bounding box of the segment
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        return True
    return False

def arc_line_segment_intersection(center, radius, p1, p2, line_start, line_end):
    intersections = circle_line_intersection(center, radius, line_start, line_end)
    arc_intersections = [pt for pt in intersections if is_point_on_arc(pt, center, p1, p2) and is_point_on_segment(pt, line_start, line_end)]
    return arc_intersections

def plot_circular_arc(center, radius, p1, p2):
    # Calculate angles of p1 and p2 relative to the center
    angle_p1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    angle_p2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])

    # Ensure angle_p2 is larger than angle_p1
    if angle_p2 < angle_p1:
        angle_p2 += 2 * np.pi

    # Generate points along the arc
    angles = np.linspace(angle_p1, angle_p2, 100)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    # Plot the arc
    plt.plot(x, y, label='Circular arc from p1 to p2')
    plt.scatter(*center, color='red', label='Center')
    plt.scatter(*p1, color='green', label='p1')
    plt.scatter(*p2, color='blue', label='p2')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Circular Arc from p1 to p2')
    plt.legend()
    plt.grid(True)
    return plt

# Example points and center of the supporting circle
center = (0, 0)
radius = 2  # Correct radius
p1 = (2, 0)
p2 = (0, 2)
line_start = (0.1, 2.1)
line_end = (2.1, 0.1)

# Find intersections
intersections = circle_line_intersection(center, radius, line_start, line_end)
# arc_intersections = [pt for pt in intersections if is_point_on_arc(pt, center, p1, p2) and is_point_on_segment(pt, line_start, line_end)]
arc_intersections = arc_line_segment_intersection(center, radius, p1, p2, line_start, line_end)
print("Intersection points:", arc_intersections)

# Plot the arc and the line segment
plt = plot_circular_arc(center, radius, p1, p2)
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'k-', label='Line segment')
for ix, iy in arc_intersections:
    plt.scatter(ix, iy, color='purple', label=f'Intersection ({ix:.2f}, {iy:.2f})')

plt.legend()
plt.show()
