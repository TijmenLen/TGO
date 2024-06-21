from stl import mesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from GUI_stl_load import load_stl
import os

def centralize_stl(radius_stl, plate_stl, radius_selection, plate_selection):

    radius, plate = load_stl(radius_stl, plate_stl, radius_selection, plate_selection)

    # Calculate the covariance matrix of the vertices
    vertices = radius.vectors.reshape(-1, 3)
    covariance_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    longest_axis = eigenvectors[:, np.argmax(eigenvalues)]

    z_as = np.array([0, 0, 1])
    rotatie_vector = np.cross(longest_axis, z_as)
    rotatie_angle = np.arccos(np.dot(longest_axis, z_as) / np.linalg.norm(longest_axis))

    if np.linalg.norm(rotatie_vector) != 0:
        rotatie_vector /= np.linalg.norm(rotatie_vector)
        rotation_matrix = R.from_rotvec(rotatie_angle * rotatie_vector).as_matrix()
    else:
        rotation_matrix = np.eye(3)

    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    if np.mean(rotated_vertices[:, 2]) < 0:
        extra_rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
        rotated_vertices = np.dot(rotated_vertices, extra_rotation_matrix.T)

    initial_x_axis = np.array([1, 0, 0])
    transformed_x_axis = np.dot(rotation_matrix, initial_x_axis)
    transformed_x_axis_xy = transformed_x_axis.copy()
    transformed_x_axis_xy[2] = 0

    correction_rotation_vector = np.cross(transformed_x_axis_xy, initial_x_axis)
    correction_rotation_angle = np.arccos(np.dot(transformed_x_axis_xy, initial_x_axis) / (np.linalg.norm(transformed_x_axis_xy) * np.linalg.norm(initial_x_axis)))

    if np.linalg.norm(correction_rotation_vector) != 0:
        correction_rotation_vector /= np.linalg.norm(correction_rotation_vector)
        correction_rotation_matrix = R.from_rotvec(correction_rotation_angle * correction_rotation_vector).as_matrix()
        rotated_vertices = np.dot(rotated_vertices, correction_rotation_matrix)

    mean_x_rotated = np.mean(rotated_vertices[:, 0])
    mean_y_rotated = np.mean(rotated_vertices[:, 1])
    mean_z_rotated = np.mean(rotated_vertices[:, 2])
    rotated_vertices[:, 0] -= mean_x_rotated
    rotated_vertices[:, 1] -= mean_y_rotated
    rotated_vertices[:, 2] -= mean_z_rotated

    highest_z_vertex_index = np.argmax(rotated_vertices[:, 2])
    highest_z_vertex = rotated_vertices[highest_z_vertex_index]
    translation_y = -highest_z_vertex[1]
    rotated_vertices[:, 1] += translation_y

    rotation_matrix = np.dot(rotation_matrix, correction_rotation_matrix)

    rotated_vertices = rotated_vertices.reshape((-1, 3, 3))
    centralized_stl = mesh.Mesh(np.zeros(rotated_vertices.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(rotated_vertices):
        for j in range(3):
            centralized_stl.vectors[i][j] = f[j]
    
    # Create a separate copy for export
    centralized_stl_export = mesh.Mesh(np.copy(centralized_stl.data))

    # Mirror the radius mesh in the x-z plane if the right radius is selected
    if radius_selection == "Right Radius":
        centralized_stl_export.x = -centralized_stl_export.x
   
    # Export the centralized STL as a file
    filename = f"transformed_radius_{os.path.basename(radius_stl)}"
    centralized_stl_export.save(filename)

    return centralized_stl
