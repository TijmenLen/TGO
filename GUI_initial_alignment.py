import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

# Importing the required functions
from GUI_cut_lower_half import process_upper_half
from GUI_plate_center_line import calculate_plate_center_line
from GUI_plate_corners import calculate_plate_corners
from GUI_define_cut import define_cut_planes
from GUI_watershedline import process_watershedline

def align_initial(radius_stl, plate_stl, radius_selection, plate_selection):
    # Ensure reproducibility by setting the seed at the start
    np.random.seed(44)
    
    plate, line_through_midpoint = calculate_plate_center_line(radius_stl, plate_stl, radius_selection, plate_selection)
    upper_radius_stl = process_upper_half(radius_stl, plate_stl, radius_selection, plate_selection)
    radius, line_of_starting_point = define_cut_planes(upper_radius_stl)
    farthest_point_above_left, farthest_point_above_right, average_point_bottom = calculate_plate_corners(radius_stl, plate_stl, radius_selection, plate_selection)

    # Extract vertices and faces for the Trimesh object
    bone_vertices = radius.vertices
    bone_faces = radius.faces

    # Extract vertices and faces for the numpy-stl object
    plate_vertices = plate.vectors.reshape(-1, 3)
    plate_faces = np.arange(plate_vertices.shape[0]).reshape(-1, 3)

    # Calculate direction vectors and midpoints
    def calculate_direction_and_midpoint(line):
        direction = line[1] - line[0]
        midpoint = np.mean(line, axis=0)
        return direction, midpoint

    direction_starting_point, midpoint_starting_point = calculate_direction_and_midpoint(line_of_starting_point)
    direction_midpoint, midpoint_plate_line = calculate_direction_and_midpoint(line_through_midpoint)

    # Invert the direction of the midpoint vector
    direction_midpoint = -direction_midpoint

    # Calculate the translation vector to align the midpoints
    translation_vector = midpoint_starting_point - midpoint_plate_line

    # Apply the translation to vertices and points
    def translate(vertices, points, translation_vector):
        translated_vertices = vertices + translation_vector
        translated_points = points + translation_vector
        return translated_vertices, translated_points

    translated_plate_vertices, translated_points = translate(
        plate_vertices, np.array([average_point_bottom, farthest_point_above_left, farthest_point_above_right]), translation_vector)
    translated_line_through_midpoint, _ = translate(line_through_midpoint, np.array([average_point_bottom]), translation_vector)

    # Calculate the rotation matrix to align direction vectors
    def calculate_rotation_matrix(v_from, v_to):
        v_from /= np.linalg.norm(v_from)
        v_to /= np.linalg.norm(v_to)
        v = np.cross(v_from, v_to)
        c = np.dot(v_from, v_to)
        s = np.linalg.norm(v)
        I = np.identity(3)
        Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        if s == 0:
            return I  # No rotation needed
        R = I + Vx + np.dot(Vx, Vx) * ((1 - c) / (s ** 2))
        return R

    rotation_matrix = calculate_rotation_matrix(direction_midpoint, direction_starting_point)

    # Apply rotation to translated vertices and points
    def apply_rotation(vertices, points, midpoint, rotation_matrix):
        rotated_vertices = np.dot(vertices - midpoint, rotation_matrix.T) + midpoint
        rotated_points = np.dot(points - midpoint, rotation_matrix.T) + midpoint
        return rotated_vertices, rotated_points

    aligned_plate_vertices, aligned_points = apply_rotation(translated_plate_vertices, translated_points, midpoint_starting_point, rotation_matrix)
    aligned_line_through_midpoint, _ = apply_rotation(translated_line_through_midpoint, np.array([translated_points[0]]), midpoint_starting_point, rotation_matrix)

    # Build KDTree for efficient closest point search
    bone_kdtree = cKDTree(bone_vertices)

    # Function to find the closest point in vertices to a given point using KDTree
    def find_closest_point(tree, point):
        distance, index = tree.query(point)
        return tree.data[index]

    # Define the function to minimize (distance between bottom point and closest point on radius)
    def distance_to_radius(angle, vertices, points, midline_start, midline_end, kdtree):
        rotation_axis = midline_end - midline_start
        rotation_axis /= np.linalg.norm(rotation_axis)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        rotated_vertices = np.dot(vertices - midline_start, R.T) + midline_start
        rotated_points = np.dot(points - midline_start, R.T) + midline_start
        bottom_point = rotated_points[0]
        closest_point = find_closest_point(kdtree, bottom_point)
        distance = np.linalg.norm(closest_point - bottom_point)
        
        # Add a penalty if the x-coordinate of the bottom point is negative
        if bottom_point[0] < 0:
            distance += 1e3 * np.abs(bottom_point[0])
            
        return distance
    
    # Process the watershedline
    bone_x, bone_y, bone_z = process_watershedline(radius_stl, plate_stl, radius_selection, plate_selection)
    watershedline = np.array([bone_x, bone_y, bone_z]).T
    max_x_watershedline = np.max(watershedline[:, 0])

    # Interpolate the z-value corresponding to the x-value of the left top point and right top point
    left_top_point = aligned_points[1]  # the left top point
    right_top_point = aligned_points[2]  # the right top point
    
    # Find the highest z-coordinate in the watershedline
    average_z_watershedline = np.average(bone_z)

    # Calculate the z difference for left top point
    z_difference_left_top = left_top_point[2] - average_z_watershedline - 2  # 2mm below the highest point in watershedline

    # Calculate the z difference for right top point
    z_difference_right_top = right_top_point[2] - average_z_watershedline - 2  # 2mm below the highest point in watershedline

    # Determine the maximum z difference
    max_z_difference = max(z_difference_left_top, z_difference_right_top)

    # Apply the downward translation based on the maximum z difference
    translation_vector_down = np.array([0, 0, -abs(max_z_difference)])  # move in z-direction

    # Apply the downward translation to vertices and points
    final_aligned_plate_vertices = aligned_plate_vertices + translation_vector_down
    final_aligned_points = aligned_points + translation_vector_down

    # Optimize the rotation angle to minimize the distance to the radius of the bottom point.
    initial_angle = 0
    result = minimize(distance_to_radius, initial_angle, args=(final_aligned_plate_vertices, final_aligned_points, line_of_starting_point[0], line_of_starting_point[1], bone_kdtree), method='BFGS', tol=1e-12, options={'maxiter': 1000})
    optimal_angle = result.x[0]

    # Apply the optimal rotation to the plate
    rotation_axis = line_of_starting_point[1] - line_of_starting_point[0]
    rotation_axis /= np.linalg.norm(rotation_axis)
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
    R = np.eye(3) + np.sin(optimal_angle) * K + (1 - np.cos(optimal_angle)) * np.dot(K, K)
    final_aligned_plate_vertices = np.dot(final_aligned_plate_vertices - line_of_starting_point[0], R.T) + line_of_starting_point[0]
    final_aligned_points = np.dot(final_aligned_points - line_of_starting_point[0], R.T) + line_of_starting_point[0]

    # Define the function to align top points' x-values to the max x-value of the watershedline
    def align_top_points_to_max_x(angle, vertices, points, bottom_point, max_x):
        rotation_axis = np.array([0, 1, 0])  # Rotate around the y-axis to adjust the x-values
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        rotated_vertices = np.dot(vertices - bottom_point, R.T) + bottom_point
        rotated_points = np.dot(points - bottom_point, R.T) + bottom_point
        top_points = rotated_points[1:]
        distance = np.sum((top_points[:, 0] - max_x)**2)
        return distance

    # Optimize the rotation angle to align the top points' x-values
    initial_angle = 0
    result = minimize(align_top_points_to_max_x, initial_angle, args=(final_aligned_plate_vertices, final_aligned_points, final_aligned_points[0], max_x_watershedline), method='BFGS', tol=1e-12, options={'maxiter': 5000})
    optimal_angle_top_points = result.x[0]

    # Apply the optimal rotation to the plate for top points alignment
    rotation_axis = np.array([0, 1, 0])
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
    R = np.eye(3) + np.sin(optimal_angle_top_points) * K + (1 - np.cos(optimal_angle_top_points)) * np.dot(K, K)
    final_aligned_plate_vertices = np.dot(final_aligned_plate_vertices - final_aligned_points[0], R.T) + final_aligned_points[0]
    final_aligned_points = np.dot(final_aligned_points - final_aligned_points[0], R.T) + final_aligned_points[0]

    return final_aligned_plate_vertices, plate_faces, final_aligned_points[0], final_aligned_points[1:], bone_vertices, bone_faces

