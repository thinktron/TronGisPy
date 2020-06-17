import osr
import gdal
import numba
import affine
import pyproj
import numpy as np
from shapely.geometry import Polygon



@numba.jit(nopython=True)
def ziyu_from_gdal(c, a, b, f, d, e):
    members = [a, b, c, d, e, f, 0.0, 0.0, 1.0]
    mat3x3 = [x * 1.0 for x in members[:-3]]
    return mat3x3

@numba.jit(nopython=True)
def invert_geo_transform( a, b, c, d, e, f):
    determinant = a * e - b * d
    idet = 1.0 / determinant
    sa, sb, sc, sd, se, sf = a, b, c, d, e, f
    ra = se * idet
    rb = -sb * idet
    rd = -sd * idet
    re = sa * idet
    return [ra, rb, -sc * ra - sf * rb,
         rd, re, -sc * rd - sf * re,
         0.0, 0.0, 1.0]

@numba.jit(nopython=True)
def __numba_transfer_coord_to_xy(x, y, a, b, c, d, e, f):
    coord_x, coord_y = x, y
    geo_transform = (a, b, c, d, e, f)
    forward_transform =  ziyu_from_gdal(*geo_transform)
    reverse_transform = invert_geo_transform(forward_transform[0], forward_transform[1], forward_transform[2],
                                             forward_transform[3], forward_transform[4], forward_transform[5] )
    reverse_transform = np.array(reverse_transform).reshape((3, 3))
    x, y, _ = reverse_transform.dot(np.array([coord_x, coord_y, 1]))
    # x, y = np.int(x), np.int(y)
    return x, y

@numba.jit(nopython=True)
def coords_to_npidxs(coords, geo_transform):
    """
    input numpy idxs, return the coords of left-top points of the cells, using the function
    | x' |   | a  b  c | | x |
    | y' | = | d  e  f | | y |
    | 1  |   | 0  0  1 | | 1 |
    npidxs: [(row_idx, col_idx), ......]
    coords: must be np.array() type. [(lng, lat), ......]
    """
    group_npidx = []
    for i in range(0, len(coords)):
        x, y = coords[i][0], coords[i][1]
        x, y = __numba_transfer_coord_to_xy(x, y, *geo_transform)
        group_npidx.append((y, x))
    return np.array(group_npidx, np.int64)

def npidxs_to_coords(npidxs, geo_transform): # TODO: reproduce to numba
    """
    input coord idxs, return the npidxs of the cells.
    coords: should be np.array() type, [(lng, lat), ......].
    npidxs: [(row_idx, col_idx), ......]
    """
    # prepare M
    c, a, b, f, d, e = geo_transform
    M = np.array([[a, b, c], 
                  [d, e, f], 
                  [0, 0, 1]])

    # prepare npidxs_maxtrix
    row_idxs, col_idxs = np.array(npidxs).T
    npidxs_maxtrix = np.ones((len(npidxs), 3)).T # => (3, -1)
    npidxs_maxtrix[0], npidxs_maxtrix[1] = col_idxs, row_idxs

    # apply multiplication
    coords = np.matmul(M, npidxs_maxtrix).T[:, :2] # (3, 3) ‧ (3, -1) => (3, -1) => (-1, 3)
    return coords

def npidx_to_coord_polygon(npidx, geo_transform): # TODO: reproduce to parallel
    """return shapely.geometry.Polygon"""
    poly_points = [ [npidx[0]+0, npidx[1]+0],  # ul
                    [npidx[0]+0, npidx[1]+1],  # ur
                    [npidx[0]+1, npidx[1]+1],  # lr
                    [npidx[0]+1, npidx[1]+0],] # ll
    poly_points = npidxs_to_coords(poly_points, geo_transform)
    return Polygon(poly_points)

def get_extent(rows, cols, geo_transform, return_poly=True):
    """get the extent(boundry) coordnate"""
    points = [[0,0], [0,cols], [rows,cols], [rows,0]]
    poly = npidxs_to_coords(points, geo_transform)
    if return_poly:
        return poly
    else:
        return (np.min(poly[:, 0]), np.max(poly[:, 0]), np.min(poly[:, 1]), np.max(poly[:, 1]))

def epsg_to_wkt(epsg=4326):
    return pyproj.CRS.from_epsg(epsg).to_wkt()

def wkt_to_epsg(wkt):
    epsg = pyproj.CRS(wkt).to_epsg()
    if epsg is not None:
        return epsg
    else:
        assert False, "the wkt connot be converted."
