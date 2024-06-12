import numpy as np

# def circle_line_intersection(center, radius, p1, p2):
#     cx, cy = center
#     radius_squared = radius ** 2
#     x1, y1 = p1
#     x2, y2 = p2

#     dx = x2 - x1
#     dy = y2 - y1
#     dr_squared = dx**2 + dy**2
#     D = x1*y2 - x2*y1

#     discriminant = radius_squared * dr_squared - D**2
#     if discriminant < 0:
#         return []  # No intersection

#     sqrt_discriminant = np.sqrt(discriminant)
#     sign_dy = np.sign(dy) if dy != 0 else 1

#     ix1 = (D*dy + sign_dy * dx * sqrt_discriminant) / dr_squared + cx
#     iy1 = (-D*dx + abs(dy) * sqrt_discriminant) / dr_squared + cy

#     ix2 = (D*dy - sign_dy * dx * sqrt_discriminant) / dr_squared + cx
#     iy2 = (-D*dx - abs(dy) * sqrt_discriminant) / dr_squared + cy

#     return [(ix1, iy1), (ix2, iy2)]

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

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
    # intersections = circle_line_intersection(center, radius, line_start, line_end)
    intersections = circle_line_segment_intersection(center, radius, line_start, line_end, full_line=False, tangent_tol=1e-9)
        
    arc_intersections = [pt for pt in intersections if is_point_on_arc(pt, center, p1, p2) and is_point_on_segment(pt, line_start, line_end)]
    return arc_intersections