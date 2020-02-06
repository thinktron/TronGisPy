import affine
from shapely.geometry import Polygon

def __transfer_xy_to_coord(xy, geo_transform):
    """inner usage
    input opencv xy, return the coord_xy for the left top of the cell
    xy: should (row_idx, col_idx)
    coord_xy: (lng, lat)
    """
    forward_transform =  affine.Affine.from_gdal(*geo_transform)
    coord_xy = forward_transform * xy
    return coord_xy

def __transfer_coord_to_xy(coord, geo_transform):
    """inner usage"""
    coord_x, coord_y = coord[0], coord[1]
    forward_transform =  affine.Affine.from_gdal(*geo_transform)
    reverse_transform = ~forward_transform
    x, y = reverse_transform * (coord_x, coord_y)
    x, y = int(x), int(y)
    return x, y

def transfer_npidx_to_coord(npidx, geo_transform):
    """
    input npumpy idx, return the coord_xy for the left top of the cell
    npidx: (row_idx, col_idx)
    coord_xy: (lng, lat)
    """
    xy = (npidx[1], npidx[0])
    coord_xy = __transfer_xy_to_coord(xy, geo_transform)
    return coord_xy

def transfer_coord_to_npidx(coord, geo_transform):
    """
    input npumpy idx, return the coord_xy for the left top of the cell
    npidx: (row_idx, col_idx)
    coord_xy: (lng, lat)
    """
    x, y = __transfer_coord_to_xy(coord, geo_transform)
    npidx = (y, x)
    return npidx

def transfer_npidx_to_coord_polygon(npidx, geo_transform):
    """return shapely.geometry.Polygon"""
    coord_x, coord_y = transfer_npidx_to_coord(npidx, geo_transform)
    xoffset, px_w, rot1, yoffset, rot2, px_h = geo_transform
    minx, miny, maxx, maxy = coord_x, coord_y+px_h, coord_x+px_w, coord_y
    polygon = Polygon([(minx, miny), (maxx, miny),  (maxx, maxy),  (minx, maxy)])
    return polygon