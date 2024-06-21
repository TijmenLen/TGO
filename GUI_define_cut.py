import numpy as np
import trimesh

def find_mins_maxs(obj):
    minx = np.min(obj.vectors[:, :, 0])
    maxx = np.max(obj.vectors[:, :, 0])
    miny = np.min(obj.vectors[:, :, 1])
    maxy = np.max(obj.vectors[:, :, 1])
    minz = np.min(obj.vectors[:, :, 2])
    maxz = np.max(obj.vectors[:, :, 2])
    return minx, maxx, miny, maxy, minz, maxz

def plane_from_points(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = -np.dot(normal, p1)
    return a, b, c, d

def plane_func_from_points(three_points):
    a, b, c, d = plane_from_points(*three_points)
    return lambda x, y: (-d - a * x - b * y) / c

def mid_plane_func(top_plane_func, bottom_plane_func):
    return lambda x, y: (top_plane_func(x, y) + bottom_plane_func(x, y)) / 2

def calculate_intersection_line(plane_func, x_value, y_range):
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    z_vals = plane_func(x_value, y_vals)
    return np.column_stack((np.full_like(y_vals, x_value), y_vals, z_vals))

def define_cut_planes(radiushead_stl):
    # Convert the loaded STL mesh to a Trimesh object
    vertices = radiushead_stl.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    radius_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Find min and max coordinates for the radius mesh
    minx_radius, maxx_radius, miny_radius, maxy_radius, minz_radius, maxz_radius = find_mins_maxs(radiushead_stl)

    # Find connected components in the mesh
    components = radius_trimesh.split(only_watertight=False)

    # Identify top and bottom components based on their centroids
    centroids = [component.centroid for component in components]
    top_component, bottom_component = (components[0], components[1]) if centroids[0][2] > centroids[1][2] else (components[1], components[0])

    def triangle_area(p1, p2, p3):
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

    def largest_triangle(mesh):
        max_area = 0
        largest_triangle_vertices = None
        for face in mesh.faces:
            p1, p2, p3 = mesh.vertices[face]
            area = triangle_area(p1, p2, p3)
            if area > max_area:
                max_area = area
                largest_triangle_vertices = (p1, p2, p3)
        return np.array(largest_triangle_vertices)

    largest_triangle_bottom = largest_triangle(bottom_component)
    largest_triangle_top = largest_triangle(top_component)

    bottom_plane_func = plane_func_from_points(largest_triangle_bottom)
    top_plane_func = plane_func_from_points(largest_triangle_top)

    def mid_plane_func(top_plane_func, bottom_plane_func):
        return lambda x, y: (top_plane_func(x, y) + bottom_plane_func(x, y)) / 2

    mid_plane_func = mid_plane_func(top_plane_func, bottom_plane_func)

    xx, yy = np.meshgrid(np.linspace(minx_radius, maxx_radius, 10), np.linspace(miny_radius, maxy_radius, 10))
    zz_mid = mid_plane_func(xx, yy)
    zz_top = top_plane_func(xx, yy)
    zz_bottom = bottom_plane_func(xx, yy)

    largest_x_point = bottom_component.vertices[np.argmax(bottom_component.vertices[:, 0])]
    x_largest, y_largest = largest_x_point[0], largest_x_point[1]
    z_largest_on_mid_plane = mid_plane_func(x_largest, y_largest)
    point_P = np.array([x_largest, y_largest, z_largest_on_mid_plane])

    yy_plane, zz_plane = np.meshgrid(np.linspace(miny_radius, maxy_radius, 10), np.linspace(minz_radius, maxz_radius, 10))
    xx_plane = np.full_like(yy_plane, point_P[0])

    line_of_starting_point = calculate_intersection_line(mid_plane_func, point_P[0], (miny_radius, maxy_radius))

    return radius_trimesh, line_of_starting_point
