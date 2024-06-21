import numpy as np
from stl import mesh
from GUI_z_x_y_axis import centralize_stl

def delete_lower_half(mesh_data):
    filtered_faces = [face for face in mesh_data.vectors if np.all(face[:, 2] >= 0)]
    new_mesh = mesh.Mesh(np.zeros(len(filtered_faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(filtered_faces):
        new_mesh.vectors[i] = face
    return new_mesh

def process_upper_half(radius, plate, radius_selection, plate_selection):    
    # Assuming centralize_stl function adjusts the radius and plate meshes and returns them.
    centralized_stl = centralize_stl(radius, plate, radius_selection, plate_selection)

    upper_radius_stl = delete_lower_half(centralized_stl)
    
    return upper_radius_stl