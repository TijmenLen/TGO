import numpy as np
from sklearn.decomposition import PCA
from GUI_stl_load import load_stl

def centralize_plate(radius_stl, plate_stl, radius_selection, plate_selection):
    # Load STL files and compute initial transformations
    radius, plate, *_ = load_stl(radius_stl, plate_stl, radius_selection, plate_selection)

    mean_x_plate = (np.min(plate.x) + np.max(plate.x)) / 2
    mean_y_plate = (np.min(plate.y) + np.max(plate.y)) / 2
    mean_z_plate = (np.min(plate.z) + np.max(plate.z)) / 2

    plate.x -= mean_x_plate
    plate.y -= mean_y_plate
    plate.z -= mean_z_plate

    points = np.vstack((plate.v0, plate.v1, plate.v2)).reshape(-1, 3)
    pca = PCA(n_components=3)
    pca.fit(points)
    principal_components = pca.components_

    def calculate_rotation_matrix(principal_components):
        target_orientation = np.array([0, 0, 1])
        current_orientation = principal_components[0]
        axis = np.cross(current_orientation, target_orientation)
        axis_length = np.linalg.norm(axis)
        if axis_length != 0:
            axis = axis / axis_length
            angle = np.arccos(np.dot(current_orientation, target_orientation))
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        else:
            rotation_matrix = np.eye(3)
        return rotation_matrix

    rotation_matrix = calculate_rotation_matrix(principal_components)
    transformed_points = np.dot(points, rotation_matrix.T)

    num_faces = len(plate.v0)
    plate.v0 = transformed_points[:num_faces]
    plate.v1 = transformed_points[num_faces:2*num_faces]
    plate.v2 = transformed_points[2*num_faces:]

    flip_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    plate.v0 = np.dot(plate.v0, flip_matrix.T)
    plate.v1 = np.dot(plate.v1, flip_matrix.T)
    plate.v2 = np.dot(plate.v2, flip_matrix.T)
    return plate

