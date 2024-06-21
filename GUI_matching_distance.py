import numpy as np
from stl import mesh
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from GUI_initial_alignment import align_initial
from GUI_watershedline import process_watershedline
import plotly.graph_objects as go
import os

def matching_distance(radius_stl, plate_stl, radius_selection, plate_selection, watershedline_constraint):
    # Ensure reproducibility by setting the seed at the start
    np.random.seed(44)

    filename = f"transformed_radius_{os.path.basename(radius_stl)}"
    print(f"Exported radius as {filename}, starting alignment and optimisation (typically about 45 seconds)")

    # Initial alignment of the plate and radius models
    aligned_plate_vertices, plate_faces, aligned_bottom_point, aligned_top_points, bone_vertices, bone_faces = align_initial(radius_stl, plate_stl, radius_selection, plate_selection)
    bone_x, bone_y, bone_z = process_watershedline(radius_stl, plate_stl, radius_selection, plate_selection)
    watershedline = np.array([bone_x, bone_y, bone_z]).T

    print("Initial alignment complete, starting optimisation")

    # Functions to translate and rotate vertices
    def translate(vertices, translation_vector):
        return vertices + translation_vector

    def rotate(vertices, rotation_vector):
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        return vertices @ rotation_matrix.T

    # Function to compute the longest axis of a set of vertices
    def compute_longest_axis(vertices):
        cov_matrix = np.cov(vertices, rowvar=False)
        eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
        longest_axis = eig_vectors[:, np.argmax(eig_values)]
        return longest_axis

    # Compute the longest axes for the initial alignment of plate and radius
    initial_plate_axis = compute_longest_axis(aligned_plate_vertices)
    initial_radius_axis = compute_longest_axis(bone_vertices)

    # Precompute the Delaunay triangulation
    delaunay_tri = Delaunay(bone_vertices)

    # Function to check if points are inside the mesh
    def is_inside_mesh(points):
        return delaunay_tri.find_simplex(points) >= 0

    # Objective function to minimize, with more weight on top and bottom points
    def objective_function(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        transformed_top_points = translate(aligned_top_points, translation_vector)
        transformed_top_points = rotate(transformed_top_points, rotation_vector)
        transformed_bottom_point = translate(aligned_bottom_point[np.newaxis, :], translation_vector)
        transformed_bottom_point = rotate(transformed_bottom_point, rotation_vector)[0]

        # Compute the distance from the transformed points to the bone vertices
        tree = cKDTree(bone_vertices)
        dist_top, _ = tree.query(transformed_top_points, k=1)
        dist_bottom, _ = tree.query(transformed_bottom_point[np.newaxis, :], k=1)
        dist_plate, _ = tree.query(transformed_plate, k=1)

        # Give higher weight to the distances of top and bottom points
        total_distance = np.sum(dist_plate) + 105 * np.sum(dist_top) + 100 * dist_bottom[0]

        # Compute the penalty for points inside the radius
        penalty = np.sum(is_inside_mesh(transformed_plate))

        # Compute the penalty for misalignment of the longest axes
        transformed_plate_axis = rotate(initial_plate_axis, rotation_vector)
        angle_penalty = np.arccos(np.clip(np.dot(transformed_plate_axis, initial_radius_axis), -1.0, 1.0))

        # Compute the penalty for the bottom point not being in the middle of the y-range
        y_middle = (y_min + y_max) / 2
        y_distance_penalty = abs(transformed_bottom_point[1] - y_middle)

        # Total objective is the total distance plus the penalty and the axis alignment penalty with reduced weight
        total_objective = total_distance + 4 * penalty + 5 * angle_penalty + 60 * y_distance_penalty

        return total_objective

    # Initial parameters for optimization
    initial_translation = np.zeros(3)
    initial_rotation = np.zeros(3)
    initial_params = np.concatenate([initial_translation, initial_rotation])

    # Constraints
    y_min, y_max = np.min(watershedline[:, 1]), np.max(watershedline[:, 1])
    y_range = y_max - y_min
    y_min_constr = y_min + 0.05 * y_range
    y_max_constr = y_max - 0.05 * y_range
    z_min_radius = np.min(bone_vertices[:, 2])
    z_min_watershed = np.min(watershedline[:, 2])
    z_max_watershed = np.max(watershedline[:, 2])

    # Tighten z-constraint to be 1.5 plate lengths beneath the top of the radius
    plate_length = np.max(aligned_plate_vertices[:, 2]) - np.min(aligned_plate_vertices[:, 2])
    z_min_constraint = z_max_watershed - 1.5 * plate_length

    z_margin = 0.5
    z_min_range = z_min_radius
    z_max_range = z_min_radius + z_margin
    z_range_indices = np.where((bone_vertices[:, 2] >= z_min_range) & (bone_vertices[:, 2] <= z_max_range))
    x_at_z_min = bone_vertices[z_range_indices][:, 0]
    x_min_radius, x_max_radius = np.min(x_at_z_min), np.max(x_at_z_min)
    radius_width = x_max_radius - x_min_radius
    quarter_radius_width = radius_width / 4

    # Define x translation constraints based on initial plate position
    plate_x_min_initial = np.min(aligned_plate_vertices[:, 0])
    plate_x_max_initial = np.max(aligned_plate_vertices[:, 0])
    x_min_constraint = plate_x_min_initial - quarter_radius_width
    x_max_constraint = plate_x_max_initial + quarter_radius_width

    def constraint_x_translation_lower(params):
        translation_vector = params[:3]
        transformed_plate_x_min = np.min(aligned_plate_vertices[:, 0] + translation_vector[0])
        return transformed_plate_x_min - x_min_constraint

    def constraint_x_translation_upper(params):
        translation_vector = params[:3]
        transformed_plate_x_max = np.max(aligned_plate_vertices[:, 0] + translation_vector[0])
        return x_max_constraint - transformed_plate_x_max

    def constraint_y_translation_upper(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        plate_y_max = np.max(transformed_plate[:, 1])
        y_upper_bound = y_max_constr
        return y_upper_bound - plate_y_max

    def constraint_y_translation_lower(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        plate_y_min = np.min(transformed_plate[:, 1])
        y_lower_bound = y_min_constr
        return plate_y_min - y_lower_bound

    def constraint_z_translation_lower(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        plate_z_min = np.min(transformed_plate[:, 2])
        return plate_z_min - z_min_constraint

    def constraint_z_translation_upper(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        plate_z_max = np.max(transformed_plate[:, 2])
        return z_max_watershed - plate_z_max

    def constraint_z_translation_upper_watershed(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        plate_z_max = np.max(transformed_plate[:, 2])
        return (z_min_watershed - 2) - plate_z_max

    def constraint_x_rotation_lower(params):
        return params[3] + np.pi / 6

    def constraint_x_rotation_upper(params):
        return np.pi / 6 - params[3]

    def constraint_y_rotation_lower(params):
        return params[4] + np.pi / 6

    def constraint_y_rotation_upper(params):
        return np.pi / 6 - params[4]

    def constraint_z_rotation_lower(params):
        return params[5] + np.pi / 6

    def constraint_z_rotation_upper(params):
        return np.pi / 6 - params[5]

    constraints = [
        {'type': 'ineq', 'fun': constraint_x_translation_lower},
        {'type': 'ineq', 'fun': constraint_x_translation_upper},
        {'type': 'ineq', 'fun': constraint_y_translation_upper},
        {'type': 'ineq', 'fun': constraint_y_translation_lower},
        {'type': 'ineq', 'fun': constraint_z_translation_lower},
        {'type': 'ineq', 'fun': constraint_x_rotation_lower},
        {'type': 'ineq', 'fun': constraint_x_rotation_upper},
        {'type': 'ineq', 'fun': constraint_y_rotation_lower},
        {'type': 'ineq', 'fun': constraint_y_rotation_upper},
        {'type': 'ineq', 'fun': constraint_z_rotation_lower},
        {'type': 'ineq', 'fun': constraint_z_rotation_upper}
    ]

    # Conditionally add the z-constraint if the watershedline_constraint is enabled
    if watershedline_constraint:
        constraints.append({'type': 'ineq', 'fun': constraint_z_translation_upper_watershed})
    else:
        constraints.append({'type': 'ineq', 'fun': constraint_z_translation_upper})

    # Check initial feasibility of constraints
    def check_initial_constraints(params):
        feasibility = True
        for constraint in constraints:
            constraint_value = constraint['fun'](params)
            if constraint_value < 0:
                feasibility = False
        return feasibility

    # Adjust initial parameters to satisfy constraints
    def adjust_initial_parameters(params):
        max_attempts = 1000
        step_size = 0.1  # Smaller step size for finer adjustments
        adjusted_params = np.copy(params)
        for attempt in range(max_attempts):
            feasibility = True
            for i in range(len(adjusted_params)):
                for sign in [-1, 1]:
                    temp_params = np.copy(adjusted_params)
                    temp_params[i] += sign * step_size
                    if all(constraint['fun'](temp_params) >= 0 for constraint in constraints):
                        adjusted_params[i] = temp_params[i]
                        feasibility = True
                        break
                if not feasibility:
                    break
            if feasibility:
                break
        return adjusted_params

    # Manually adjust the initial parameters for constraint 4 (z_translation_upper)
    def manually_adjust_initial_parameters(params):
        translation_vector = params[:3]
        rotation_vector = params[3:]
        transformed_plate = translate(aligned_plate_vertices, translation_vector)
        transformed_plate = rotate(transformed_plate, rotation_vector)
        z_translation_adjustment = z_min_watershed - np.max(transformed_plate[:, 2])
        
        # Apply the adjustment to the initial parameters
        params[2] += z_translation_adjustment
        return params

    # Manually adjust initial parameters to satisfy constraint 4
    initial_params = manually_adjust_initial_parameters(initial_params)

    # Check and adjust initial parameters
    if not check_initial_constraints(initial_params):
        initial_params = adjust_initial_parameters(initial_params)

    # Optimization
    result = minimize(objective_function, initial_params, method='SLSQP', constraints=constraints, options={'disp': False, 'maxiter': 5000})

    # Final parameters
    final_params = result.x
    translation_vector = final_params[:3]
    rotation_vector = final_params[3:]
    transformed_plate = translate(aligned_plate_vertices, translation_vector)
    transformed_plate = rotate(transformed_plate, rotation_vector)
    transformed_top_points = translate(aligned_top_points, translation_vector)
    transformed_top_points = rotate(transformed_top_points, rotation_vector)
    transformed_bottom_point = translate(aligned_bottom_point[np.newaxis, :], translation_vector)
    transformed_bottom_point = rotate(transformed_bottom_point, rotation_vector)[0]

    # Mirror the radius and plate in the x-z plane if the right radius is selected
    if radius_selection == "Right Radius":
        bone_vertices[:, 0] = -bone_vertices[:, 0]
        watershedline[:, 0] = -watershedline[:, 0]
        transformed_plate[:, 0] = -transformed_plate[:, 0]
        transformed_bottom_point[0] = -transformed_bottom_point[0]
        transformed_top_points[:, 0] = -transformed_top_points[:, 0]

    # Plotting
    fig = go.Figure(data=[
        go.Mesh3d(
            x=bone_vertices[:, 0], y=bone_vertices[:, 1], z=bone_vertices[:, 2],
            i=bone_faces[:, 0], j=bone_faces[:, 1], k=bone_faces[:, 2],
            color='gray', opacity=0.6, name='Radius'
        ),
        go.Mesh3d(
            x=transformed_plate[:, 0], y=transformed_plate[:, 1], z=transformed_plate[:, 2],
            i=plate_faces[:, 0], j=plate_faces[:, 1], k=plate_faces[:, 2],
            color='blue', opacity=1, name='Plate'
        ),
        go.Scatter3d(
            x=transformed_top_points[:, 0], y=transformed_top_points[:, 1], z=transformed_top_points[:, 2],
            mode='markers', marker=dict(size=4, color='red'), name='Top Points'
        ),
        go.Scatter3d(
            x=[transformed_bottom_point[0]], y=[transformed_bottom_point[1]], z=[transformed_bottom_point[2]],
            mode='markers', marker=dict(size=4, color='green'), name='Bottom Point'
        ),
        go.Scatter3d(
            x=watershedline[:, 0], y=watershedline[:, 1], z=watershedline[:, 2],
            mode='lines', line=dict(color='purple', width=2), name='Watershedline'
        )
    ])

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title=f"Plate and Radius Alignment: Radius = {os.path.basename(radius_stl)}, Plate = {os.path.basename(plate_stl)} Selection = {radius_selection}, {plate_selection}"
    )

    fig.show()

    # Export the transformed plate to an STL file using numpy-stl
    output_filename = f"transformed_plate_{os.path.basename(radius_stl)}_{radius_selection}_{os.path.basename(plate_stl)}_{plate_selection}_{'watershedline' if watershedline_constraint else 'no_watershedline'}.stl"

    # Create the mesh
    transformed_plate_mesh = mesh.Mesh(np.zeros(transformed_plate.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(plate_faces):
        for j in range(3):
            transformed_plate_mesh.vectors[i][j] = transformed_plate[f[j],:]

    transformed_plate_mesh.save(output_filename)

    print(f"Exported plate stl-file as {output_filename}")
