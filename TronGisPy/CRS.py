import osr
import gdal
import numba
import affine
import pyproj
import numpy as np

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
    """Find the cells' numpy index the coordinates belong to using 
    the following functions.
    | npidx_col |   | a  b  c | | coord_x |
    | npidx_row | = | d  e  f | | coord_y |
    |     1     |   | 0  0  1 | |    1    |

    Parameters
    ----------
    coords: `numpy.array`. The coordinates with shape (n_points, 2). The order of
    last dimension is (lng, lat).

    geo_transform: tuple or list. Affine transform parameters (c, a, b, f, d, e
    = geo_transform).

    Returns
    -------
    npidxs: `numpy.array`. The numpy indeices with shape (n_points, 2). The order of
    last dimension is (row_idx, col_idx).

    Examples
    --------
    >>> import numpy as np
    >>> import TronGisPy as tgp
    >>> geo_transform = tgp.get_raster_info(tgp.get_testing_fp(), 'geo_transform')
    >>> coords = np.array([(271986.23416588, 2769971.94666942)])
    >>> tgp.coords_to_npidxs(coords, geo_transform)
    array([[1, 3]], dtype=int64)
    """
    group_npidx = []
    for i in range(0, len(coords)):
        x, y = coords[i][0], coords[i][1]
        x, y = __numba_transfer_coord_to_xy(x, y, *geo_transform)
        group_npidx.append((y, x))
    return np.array(group_npidx, np.int64)

def npidxs_to_coords(npidxs, geo_transform): # TODO: reproduce to numba
    """Get coordinates of cells' left-top corner by its numpy index using 
    the following functions.
    | npidx_col |   | a  b  c | | coord_x |
    | npidx_row | = | d  e  f | | coord_y |
    |     1     |   | 0  0  1 | |    1    |

    Parameters
    ----------
    npidxs: `numpy.array`. The numpy indeices with shape (n_points, 2). The order of
    last dimension is (row_idx, col_idx).

    geo_transform: tuple or list. Affine transform parameters (c, a, b, f, d, e
    = geo_transform).

    Returns
    -------
    coords: `numpy.array`. The coordinates with shape (n_points, 2). The order of
    last dimension is (lng, lat).

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> geo_transform = tgp.get_raster_info(tgp.get_testing_fp(), 'geo_transform')
    >>> tgp.npidxs_to_coords([(1,3)], geo_transform)
    array([[ 271986.23416588, 2769971.94666942]])
    """
    npidxs = np.array(npidxs)
    assert len(npidxs.shape) == 2, "npidxs.shape should be (n_points, 2)"

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

def npidxs_to_coord_polygons(npidxs, geo_transform):
    """Get coordinates of cells' four corners by its numpy index. 

    Parameters
    ----------
    npidxs: `numpy.array`. The numpy indeices with shape (n_points, 2). The order
    of last dimension is (row_idx, col_idx).

    geo_transform: tuple or list. Affine transform parameters (c, a, b, f, d, e
    = geo_transform).

    Returns
    -------
    poly_points: `numpy.array`. The four corner coordinates for each npidxs with
    shape (-1, 4, 2). The order of last dimension is (lng, lat).

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> geo_transform = tgp.get_raster_info(tgp.get_testing_fp(), 'geo_transform')
    >>> tgp.npidxs_to_coord_polygons([(1,3)], geo_transform)
    array([[[ 271986.23416588, 2769971.94666942],
            [ 271987.35278783, 2769971.94666942],
            [ 271987.35278783, 2769970.82803885],
            [ 271986.23416588, 2769970.82803885]]])
    """
    npidxs = np.array(npidxs)
    assert len(npidxs.shape) == 2, "npidxs.shape should be (n_points, 2)"
    poly_points = [ [npidxs[:, 0]+0, npidxs[:, 1]+0],  # ul
                    [npidxs[:, 0]+0, npidxs[:, 1]+1],  # ur
                    [npidxs[:, 0]+1, npidxs[:, 1]+1],  # lr
                    [npidxs[:, 0]+1, npidxs[:, 1]+0],] # ll
    poly_points = np.array(poly_points).transpose([2, 0, 1]).reshape(-1, 2)
    poly_points = npidxs_to_coords(poly_points, geo_transform)
    poly_points = poly_points.reshape(-1, 4, 2)
    return poly_points

def get_extent(rows, cols, geo_transform, return_poly=True):
    """Get the boundary of a raster file

    Parameters
    ----------
    rows: int. The number of rows in the raster.

    cols: int. The number of cols in the raster.

    geo_transform : tuple or list. Affine transform parameters (c, a, b, f, d, e
    = geo_transform).

    return_poly: bool. If True, return four corner coordinates, else return
    (xmin, xmax, ymin, ymax)

    Returns
    -------
    extent: `numpy.array` ot tuple. If return_poly==True, return four corner coordinates, else return
    (xmin, xmax, ymin, ymax)

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> rows, cols, geo_transform = tgp.get_raster_info(tgp.get_testing_fp(), ['rows', 'cols', 'geo_transform'])
    >>> tgp.get_extent(rows, cols, geo_transform, False)
    (271982.8783, 272736.8295, 2769215.7524, 2769973.0653)
    """
    points = [[0,0], [0,cols], [rows,cols], [rows,0]]
    poly = npidxs_to_coords(points, geo_transform)
    if return_poly:
        extent = poly
    else:
        extent = (np.min(poly[:, 0]), np.max(poly[:, 0]), np.min(poly[:, 1]), np.max(poly[:, 1]))
    return extent
def epsg_to_wkt(epsg=4326):
    """convert epsg code to well known text (WKT).

    Parameters
    ----------
    epsg: int. The epsg code.

    Returns
    -------
    wkt: string. The well known text of the epsg code.
    
    Examples
    --------
    >>> import TronGisPy as tgp
    >>> tgp.epsg_to_wkt(4326)
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["unknown"],AREA["World"],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
    """
    return pyproj.CRS.from_epsg(epsg).to_wkt()

def wkt_to_epsg(wkt):
    """convert well known text (WKT) to epsg code.

    Parameters
    ----------
    wkt: string. The well known text of the epsg code.

    Returns
    -------
    epsg: int. The epsg code.

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> projection = tgp.get_raster_info(tgp.get_testing_fp(), 'projection')
    >>> tgp.wkt_to_epsg(projection)
    """
    srs = osr.SpatialReference(wkt=wkt) # pyproj.CRS(wkt).to_epsg() will fail
    epsg = srs.GetAuthorityCode(None)
    if epsg is not None:
        return int(epsg)
    else:
        assert False, "the wkt connot be converted."
