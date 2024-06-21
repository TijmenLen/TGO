import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from GUI_plate_transformation import centralize_plate

def calculate_plate_center_line(radius_stl, plate_stl, radius_selection, plate_selection):
    plate = centralize_plate(radius_stl, plate_stl, radius_selection, plate_selection)

    def calculate_centroid(vertices):
        return np.mean(vertices, axis=0)

    def find_closest_points_to_centroid(points, centroid, percentage=30):
        distances = np.linalg.norm(points - centroid, axis=1)
        num_points = int(len(points) * percentage / 100)
        closest_indices = np.argsort(distances)[:num_points]
        return points[closest_indices]

    def find_plate_holes(points, num_clusters=3):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(points)
        cluster_centers = kmeans.cluster_centers_
        plate_holes = cluster_centers[[0, 2, 3]]
        return plate_holes

    def calculate_parallel_line(midpoint, direction, line_length):
        direction = direction.astype(float)
        direction /= np.linalg.norm(direction)
        line_endpoint1 = midpoint - direction * (line_length / 2)
        line_endpoint2 = midpoint + direction * (line_length / 2)
        return line_endpoint1, line_endpoint2

    points = np.vstack((plate.v0, plate.v1, plate.v2)).reshape(-1, 3)

    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(points)
    density_scores = kde.score_samples(points)
    threshold = np.percentile(density_scores, 30)
    high_density_points = points[density_scores > threshold]

    centroid = calculate_centroid(points)
    closest_points = find_closest_points_to_centroid(high_density_points, centroid, 10)

    plate_holes = find_plate_holes(closest_points, num_clusters=4)
    plate_holes[:, 2] = np.where(plate_holes[:, 2] < 0, -plate_holes[:, 2], plate_holes[:, 2])
    centroid[2] = np.where(centroid[2] < 0, -centroid[2], centroid[2])
    above_centroid = plate_holes[plate_holes[:, 2] > centroid[2]]
    below_centroid = plate_holes[plate_holes[:, 2] <= centroid[2]]

    if len(above_centroid) >= 2 and len(below_centroid) >= 1:
        midpoint_above_centroid = np.mean(above_centroid[:2], axis=0)
        lower_point = below_centroid[0]
        midpoint_line = (midpoint_above_centroid + lower_point) / 2
        new_line_direction = above_centroid[1] - above_centroid[0]
        new_line_direction /= np.linalg.norm(new_line_direction)
        line_length = 50
        line_through_midpoint = calculate_parallel_line(midpoint_line, new_line_direction, line_length)

        return plate, line_through_midpoint
    else:
        raise ValueError("Insufficient points above or below the centroid to perform the calculation.")