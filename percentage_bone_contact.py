import numpy as np
from stl import mesh
from scipy.spatial import cKDTree

# Load the STL files
radius = mesh.Mesh.from_file('')
plate = mesh.Mesh.from_file('')

# Extract vertices and faces for the radius mesh
radius_vertices = radius.vectors.reshape(-1, 3)
radius_faces = np.arange(radius_vertices.shape[0]).reshape(-1, 3)

# Extract vertices and faces for the plate mesh
plate_vertices = plate.vectors.reshape(-1, 3)
plate_faces = np.arange(plate_vertices.shape[0]).reshape(-1, 3)

# Create a KDTree for fast nearest-neighbor lookup
radius_tree = cKDTree(radius_vertices)

# Define the threshold distance in mm
threshold_distance = 0.1

# Query the distances from plate vertices to the radius mesh
distances, _ = radius_tree.query(plate_vertices, k=1)

# Count the number of plate points within the threshold distance
within_threshold = np.sum(distances <= threshold_distance)

# Calculate the percentage of points within the threshold distance
percentage_within_threshold = (within_threshold / len(plate_vertices)) * 100

print(f"Percentage of plate mesh points within {threshold_distance} mm of the radius mesh: {percentage_within_threshold:.2f}%")

# Get the points within the threshold distance
points_within_threshold = plate_vertices[distances <= threshold_distance]

# Find and print the top 5 closest distances
old_top_5_distances = np.sort(distances)[:5]
print(f"Old Top 5 closest distances from the plate to the bone: {old_top_5_distances}")

# Define a function to filter out distances that are the same or within a specified tolerance
def filter_distances(distances, tolerance=0.000000001):
    filtered_distances = []
    for distance in distances:
        if not filtered_distances or np.abs(distance - filtered_distances[-1]) > tolerance:
            filtered_distances.append(distance)
    return np.array(filtered_distances)

# Apply the filtering function
filtered_distances = filter_distances(np.sort(distances))

# Get the new top 5 closest distances after filtering
new_top_5_distances = filtered_distances[:5]
print(f"New Top 5 closest distances from the plate to the bone: {new_top_5_distances}")




