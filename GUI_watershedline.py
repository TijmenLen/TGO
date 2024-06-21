import numpy as np
from collections import defaultdict
from GUI_cut_lower_half import process_upper_half
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import trimesh

def find_mins_maxs(obj):
    minx = np.min(obj.vectors[:, :, 0])
    maxx = np.max(obj.vectors[:, :, 0])
    miny = np.min(obj.vectors[:, :, 1])
    maxy = np.max(obj.vectors[:, :, 1])
    minz = np.min(obj.vectors[:, :, 2])
    maxz = np.max(obj.vectors[:, :, 2])
    return minx, maxx, miny, maxy, minz, maxz

def process_watershedline(radius_stl, plate_stl, radius_selection, plate_selection):
    radiushead_stl = process_upper_half(radius_stl, plate_stl, radius_selection, plate_selection)

    def calculate_max_x_per_y_step(vertices, step=0.5, exclude_percent=0.1):
        max_x_for_y = defaultdict(lambda: (float('-inf'), float('-inf'), float('-inf')))
        y_value = vertices[:, 1]
        sorted_y_value = np.sort(y_value)

        # determining exclusion boundaries
        lower_bound_index = int((exclude_percent) * len(sorted_y_value))
        upper_bound_index = int((1 - exclude_percent) * len(sorted_y_value))

        lower_bound_y = sorted_y_value[lower_bound_index]
        upper_bound_y = sorted_y_value[upper_bound_index]
        
        for x, y, z in vertices:
            if lower_bound_y <= y <= upper_bound_y:
                y_group = (y // step) * step
                max_x, _, _ = max_x_for_y[y_group]
                if x > max_x:
                    max_x_for_y[y_group] = (x, y, z)
        return max_x_for_y

    def watershed_line(mesh_data, max_x_for_y):
        y_values = sorted(max_x_for_y.keys())
        x_values = [max_x_for_y[y][0] for y in y_values]
        z_values = [max_x_for_y[y][2] for y in y_values]
        
        # Create polynomial regression model
        degree = 99  # Degree of polynomial (adjust as needed)
        polyreg_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        polyreg_model.fit(np.array(y_values).reshape(-1, 1), np.array(z_values))
        
        # Predict z values using the polynomial regression model
        z_values = polyreg_model.predict(np.array(y_values).reshape(-1, 1))
        
        return x_values, y_values, z_values

    # Separate the top and bottom components
    vertices = radiushead_stl.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    radius_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Find connected components in the mesh
    components = radius_trimesh.split(only_watertight=False)

    # Identify top and bottom components based on their centroids
    centroids = [component.centroid for component in components]
    top_component, bottom_component = (components[0], components[1]) if centroids[0][2] > centroids[1][2] else (components[1], components[0])

    # Process the top component
    top_vertices = top_component.vertices
    max_x_for_y = calculate_max_x_per_y_step(top_vertices)
    x_values, y_values, z_values = watershed_line(top_component, max_x_for_y)

    return x_values, y_values, z_values