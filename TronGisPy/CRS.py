import osr
import gdal
import affine
import numpy as np
from shapely.geometry import Polygon
import numba



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
def numba_transfer_coord_to_npidx(coord, geo_transform):
    x, y = coord[0], coord[1]
    x, y = __numba_transfer_coord_to_xy(x, y, *geo_transform)
    npidx = [y, x]
    return npidx


@numba.jit(nopython=True)
def __numba_transfer_coord_to_xy(x, y, a, b, c, d, e, f):
    coord_x, coord_y = x, y
    geo_transform = (a, b, c, d, e, f)
    forward_transform =  ziyu_from_gdal(*geo_transform)
    reverse_transform = invert_geo_transform(forward_transform[0], forward_transform[1], forward_transform[2],
                                             forward_transform[3], forward_transform[4], forward_transform[5] )
    reverse_transform = np.array(reverse_transform).reshape((3, 3))
    x, y, _ = reverse_transform.dot(np.array([coord_x, coord_y, 1]))
    x, y = np.int(x), np.int(y)
    return x, y

@numba.jit(nopython=True)
def numba_transfer_group_coord_to_npidx(coords, geo_transform):
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


def transfer_group_npidx_to_coord(npidxs, geo_transform):
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
    coords = np.dot(M, npidxs_maxtrix).T[:, :2] # (3, 3) ‧ (3, -1) => (3, -1) => (-1, 3)
    return coords

def transfer_npidx_to_coord(npidx, geo_transform):
    """
    input numpy idx, return the coord of the left-top point of the cell.
    npidx: (row_idx, col_idx)
    coord: (lng, lat)
    """
    npidx_reverse = (npidx[1], npidx[0])
    forward_transform =  affine.Affine.from_gdal(*geo_transform)
    coord = forward_transform * npidx_reverse
    return coord

def transfer_coord_to_npidx(coord, geo_transform, convert_to_int=True):
    """
    input coord idx, return the npidx of the cell.
    npidx: (row_idx, col_idx)
    coord: (lng, lat)
    """
    reverse_transform = ~affine.Affine.from_gdal(*geo_transform)
    col_idx, row_idx = reverse_transform * tuple(coord)
    if convert_to_int: col_idx, row_idx = int(col_idx), int(row_idx)
    npidx = (row_idx, col_idx)
    return npidx

def transfer_npidx_to_coord_polygon(npidx, geo_transform):
    """return shapely.geometry.Polygon"""
    coord_x, coord_y = transfer_npidx_to_coord(npidx, geo_transform)
    xoffset, px_w, rot1, yoffset, rot2, px_h = geo_transform
    minx, miny, maxx, maxy = coord_x, coord_y+px_h, coord_x+px_w, coord_y
    polygon = Polygon([(minx, miny), (maxx, miny),  (maxx, maxy),  (minx, maxy)])
    return polygon

def get_wkt_from_epsg(epsg=4326):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()

def get_epsg_from_wkt(wkt):
    srs = osr.SpatialReference(wkt=wkt)
    epsg = srs.GetAuthorityCode(None)
    return int(epsg)
    

