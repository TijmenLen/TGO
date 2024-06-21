import numpy as np
from GUI_plate_transformation import centralize_plate

def calculate_plate_corners(radius_stl, plate_stl, radius_selection, plate_selection):
    plate = centralize_plate(radius_stl, plate_stl, radius_selection, plate_selection)

    def calculate_centroid(vertices):
        return np.mean(vertices, axis=0)

    points = np.vstack((plate.v0, plate.v1, plate.v2)).reshape(-1, 3)
    centroid = calculate_centroid(points)

    above_left = points[(points[:, 0] <= centroid[0]) & (points[:, 2] > centroid[2])]
    above_right = points[(points[:, 0] > centroid[0]) & (points[:, 2] > centroid[2])]
    below_left = points[(points[:, 0] <= centroid[0]) & (points[:, 2] <= centroid[2])]
    below_right = points[(points[:, 0] > centroid[0]) & (points[:, 2] <= centroid[2])]

    distances_above_left = np.linalg.norm(above_left - centroid, axis=1)
    distances_above_right = np.linalg.norm(above_right - centroid, axis=1)
    distances_below_left = np.linalg.norm(below_left - centroid, axis=1)
    distances_below_right = np.linalg.norm(below_right - centroid, axis=1)

    farthest_point_above_left = above_left[np.argmax(distances_above_left)]
    farthest_point_above_right = above_right[np.argmax(distances_above_right)]
    farthest_point_below_left = below_left[np.argmax(distances_below_left)]
    farthest_point_below_right = below_right[np.argmax(distances_below_right)]
    average_point_bottom = np.mean(np.vstack((farthest_point_below_left, farthest_point_below_right)), axis=0)

    return farthest_point_above_left, farthest_point_above_right, average_point_bottom