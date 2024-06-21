import numpy as np
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import plotly.graph_objects as go
from stl import mesh
from scipy.spatial.transform import Rotation

radiustg = mesh.Mesh.from_file('')
platescript = mesh.Mesh.from_file('')
platetg=mesh.Mesh.from_file('')


def filter_points_with_z_equals_zero(mesh_data):
    if isinstance(mesh_data, mesh.Mesh):
        # Get vertices
        vertices = mesh_data.vectors.reshape(-1, 3, 3)  # Reshape to (n_triangles, 3 vertices, 3 coordinates)
        
        # Filter out triangles where all vertices have z=0
        non_zero_triangles = np.where(vertices[:, :, 2] != 0)[0]
        
        # Create filtered mesh data
        filtered_mesh_data = vertices[non_zero_triangles]
        
        # Reshape filtered mesh data back to (n_triangles, 3, 3)
        filtered_mesh_data = filtered_mesh_data.reshape(-1, 3, 3)
        
        # Create a new mesh object with filtered data
        filtered_mesh = mesh.Mesh(np.zeros_like(mesh_data.data))
        filtered_mesh.vectors = filtered_mesh_data
        
        return filtered_mesh, filtered_mesh_data
    else:
        return mesh_data, []
    
def calculate_hausdorff_distance(vertices1, vertices2):
    vertices1 = vertices1.reshape(-1, 3) if vertices1.ndim == 3 else vertices1
    vertices2 = vertices2.reshape(-1, 3) if vertices2.ndim == 3 else vertices2
    distances_1_to_2 = directed_hausdorff(vertices1, vertices2)[0]
    distances_2_to_1 = directed_hausdorff(vertices2, vertices1)[0]
    hausdorff_distance = max(distances_1_to_2, distances_2_to_1)
    return hausdorff_distance

# Calculate the transformation matrix
def calculate_transformation_matrix(vertices1, vertices2):
    vertices1 = vertices1.reshape(-1, 3) if vertices1.ndim == 3 else vertices1
    vertices2 = vertices2.reshape(-1, 3) if vertices2.ndim == 3 else vertices2
    centroid1 = np.mean(vertices1, axis=0)
    centroid2 = np.mean(vertices2, axis=0)
    
    translation = centroid2 - centroid1
    
    highest_vertex1 = vertices1[np.argmax(vertices1[:, 2])]
    highest_vertex2 = vertices2[np.argmax(vertices2[:, 2])]
    
    v1 = highest_vertex1 - centroid1
    v2 = highest_vertex2 - centroid2
    
    rotation, _ = Rotation.align_vectors([v2], [v1])
    euler_angles = rotation.as_euler('xyz', degrees=True)
    
    return translation, euler_angles

platescript, vertices_plate_script = filter_points_with_z_equals_zero(platescript)
vertices_plate_tg = platetg.vectors.reshape(-1,3)
print(vertices_plate_script)
print(vertices_plate_tg)

# Calculate Hausdorff distance
hausdorff_distance_script_plaatpositietg = calculate_hausdorff_distance(vertices_plate_tg, vertices_plate_script)

# Calculate transformation matrix
translation_scriptplacement, euler_angles_scriptplacement = calculate_transformation_matrix(vertices_plate_tg, vertices_plate_script)

# Print the results
print("Hausdorff distance between script placement and plaatpositietg:", hausdorff_distance_script_plaatpositietg, "mm")
print("Transformation matrix for scriptplacement:")
print("Rotation in x direction:", euler_angles_scriptplacement[0], "degrees")
print("Rotation in y direction:", euler_angles_scriptplacement[1], "degrees")
print("Rotation in z direction:", euler_angles_scriptplacement[2], "degrees")
print("Movement in x-direction:", translation_scriptplacement[0], "mm")
print("Movement in y-direction:", translation_scriptplacement[1], "mm")
print("Movement in z-direction:", translation_scriptplacement[2], "mm")

def create_mesh_plot(mesh_data, name, color):
    # Create a trace for the mesh
    if isinstance(mesh_data, mesh.Mesh):
        # Reshape the vertices for Plotly
        vertices = mesh_data.vectors.reshape(-1, 3)
    else:
        # mesh_data is assumed to be a numpy array
        vertices = mesh_data.reshape(-1, 3)
    
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = np.arange(0, len(x), 3), np.arange(1, len(x), 3), np.arange(2, len(x), 3)
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        name=name,
        color=color,
        opacity=0.5,
        flatshading=True
    )

# Create mesh plots
plaatpositietg_plot = create_mesh_plot(platetg, 'plaatpositietg', 'green')
scriptplacement_plot = create_mesh_plot(platescript, 'scriptplacement', 'blue')

# Non-transformed radius plot
normal_radius_plot = create_mesh_plot(radiustg, 'normal_radius', 'red')

# Combine all plots
fig = go.Figure(data=[plaatpositietg_plot, scriptplacement_plot, normal_radius_plot])

# Calculate the bounding box for all meshes to set axis ranges
all_x = np.concatenate([plaatpositietg_plot.x, scriptplacement_plot.x, normal_radius_plot.x])
all_y = np.concatenate([plaatpositietg_plot.y, scriptplacement_plot.y, normal_radius_plot.y])
all_z = np.concatenate([plaatpositietg_plot.z, scriptplacement_plot.z, normal_radius_plot.z])

x_range = [np.min(all_x), np.max(all_x)]
y_range = [np.min(all_y), np.max(all_y)]
z_range = [np.min(all_z), np.max(all_z)]

# Set axis to have equal scaling
fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        aspectmode='data'
    ),
    title='3D Mesh Plot',
    width=800,
    height=800
)

# Show the plot
fig.show()
