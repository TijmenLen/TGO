def load_stl(radius_stl, plate_stl, radius_selection, plate_selection):
    from stl import mesh
    
    def load_mesh(stl_input):
        if isinstance(stl_input, str):
            return mesh.Mesh.from_file(stl_input)
        elif isinstance(stl_input, mesh.Mesh):
            return stl_input
        else:
            raise TypeError("Expected str or mesh.Mesh, got {}".format(type(stl_input)))
    
    # Loading radius and plate
    radius = load_mesh(radius_stl)
    plate = load_mesh(plate_stl)

    # Mirror the radius mesh in the x-z plane if the right radius is selected
    if radius_selection == "Right Radius":
        radius.x = -radius.x
    if plate_selection == "Right Plate":
        plate.x = -plate.x
    
    return radius, plate