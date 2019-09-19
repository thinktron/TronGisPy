import affine

def transfer_xy_to_coord(xy, geo_transform):
    forward_transform =  affine.Affine.from_gdal(*geo_transform)
    coord_xy = forward_transform * xy
    return coord_xy

def transfer_coord_to_xy(coord, geo_transform):
    coord_x, coord_y = coord[0], coord[1]
    forward_transform =  affine.Affine.from_gdal(*geo_transform)
    reverse_transform = ~forward_transform
    x, y = reverse_transform * (coord_x, coord_y)
    x, y = int(x + 0.5), int(y + 0.5)
    return x, y