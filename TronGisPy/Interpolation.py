import os
import numpy as np
from scipy.interpolate import griddata



def band_interpolation(X, method='linear', mask=None):
    """
    X: please whole image.
    method: 'nearest', 'linear', 'cubic', see scipy.interpolate.griddata documentation.
    mask: The location to be filled with value. If mask == None, use np.nan(X) as mask.
    """
    assert len(X.shape) == 2, "X should be one band image!"
    if mask is not None:
        assert X.shape == mask.shape, 'X.shape should be mask.shape!'
        assert mask.dtype == np.bool, 'mask.dtype should be np.bool!'
        points = np.array(np.where(~mask)).T
        values = X[~mask].copy()
    else:
        points = np.array(np.where(~np.isnan(X))).T
        values = X[~np.isnan(X)].copy()
    grid_x, grid_y = np.where(np.ones_like(X))
    grid_x, grid_y = grid_x.reshape(X.shape), grid_y.reshape(X.shape)
    X_interp = griddata(points, values, (grid_x, grid_y), method=method)
    return X_interp

def img_interpolation(X, method='linear', mask=None):
    X_interp = X.copy()
    if len(X_interp.shape) == 2:
        X_interp = np.expand_dims(X_interp, axis=2)
    for b in range(X_interp.shape[2]):
        X_interp[:,:,b] = band_interpolation(X_interp[:,:,b])
    return X_interp

# import os
# import subprocess
# base_dir = os.path.dirname(os.path.realpath(__file__))
# saga_cmd_fp = os.path.join(base_dir, "apps", "saga-7.0.0_x64", "saga_cmd")

# def inverse_distance_weighted(POINTS, FIELD, DOWNLOAD_DIR, TARGET_TEMPLATE=None, CV_METHOD=0, CV_SAMPLES=10, TARGET_DEFINITION=0, TARGET_USER_SIZE=1.0, TARGET_USER_XMIN=0.0, TARGET_USER_XMAX=100.0, TARGET_USER_YMIN=0.0, TARGET_USER_YMAX=100.0, TARGET_USER_FITS=0, SEARCH_RANGE=1, SEARCH_RADIUS=1000.0, SEARCH_POINTS_ALL=1, SEARCH_POINTS_MIN=-1, SEARCH_POINTS_MAX=20, SEARCH_DIRECTION=0, DW_WEIGHTING=1, DW_IDW_POWER=2.0, DW_IDW_OFFSET=False, DW_BANDWIDTH=1.0):
#     """
#     POINTS:<str>             	Points
#         Shapes (input)
#     FIELD:<str>              	Attribute
#         Table field
#     CV_METHOD:<str>          	Cross Validation
#         Choice
#         Available Choices:
#         [0] none
#         [1] leave one out
#         [2] 2-fold
#         [3] k-fold
#         Default: 0
#     CV_SUMMARY:<str>         	Cross Validation Summary
#         Table (optional output)
#     CV_RESIDUALS:<str>       	Cross Validation Residuals
#         Shapes (optional output)
#     CV_SAMPLES:<num>         	Cross Validation Subsamples
#         Integer
#         Minimum: 0
#         Default: 10
#     TARGET_DEFINITION:<str>  	Target Grid System
#         Choice
#         Available Choices:
#         [0] user defined
#         [1] grid or grid system
#         Default: 0
#     TARGET_USER_SIZE:<double>	Cellsize
#         Floating point
#         Minimum: 0.000000
#         Default: 1.000000
#     TARGET_USER_XMIN:<double>	Left
#         Floating point
#         Default: 0.000000
#     TARGET_USER_XMAX:<double>	Right
#         Floating point
#         Default: 100.000000
#     TARGET_USER_YMIN:<double>	Bottom
#         Floating point
#         Default: 0.000000
#     TARGET_USER_YMAX:<double>	Top
#         Floating point
#         Default: 100.000000
#     TARGET_USER_FITS:<str>   	Fit
#         Choice
#         Available Choices:
#         [0] nodes
#         [1] cells
#         Default: 0
#     TARGET_TEMPLATE:<str>    	Target System
#         Grid (optional input)
#     TARGET_OUT_GRID:<str>    	Target Grid
#         Grid (output)
#     SEARCH_RANGE:<str>       	Search Range
#         Choice
#         Available Choices:
#         [0] local
#         [1] global
#         Default: 1
#     SEARCH_RADIUS:<double>   	Maximum Search Distance
#         Floating point
#         Minimum: 0.000000
#         Default: 1000.000000
#     SEARCH_POINTS_ALL:<str>  	Number of Points
#         Choice
#         Available Choices:
#         [0] maximum number of nearest points
#         [1] all points within search distance
#         Default: 1
#     SEARCH_POINTS_MIN:<num>  	Minimum
#         Integer
#         Minimum: 0
#         Default: -1
#     SEARCH_POINTS_MAX:<num>  	Maximum
#         Integer
#         Minimum: 0
#         Default: 20
#     SEARCH_DIRECTION:<str>   	Direction
#         Choice
#         Available Choices:
#         [0] all directions
#         [1] quadrants
#         Default: 0
#     DW_WEIGHTING:<str>       	Weighting Function
#         Choice
#         Available Choices:
#         [0] no distance weighting
#         [1] inverse distance to a power
#         [2] exponential
#         [3] gaussian weighting
#         Default: 1
#     DW_IDW_POWER:<double>    	Inverse Distance Weighting Power
#         Floating point
#         Minimum: 0.000000
#         Default: 2.000000
#     DW_IDW_OFFSET:<str>      	Inverse Distance Offset
#         Boolean
#         Default: 0
#     DW_BANDWIDTH:<double>    	Gaussian and Exponential Weighting Bandwidth
#         Floating point
#         Minimum: 0.000000
#         Default: 1.000000
#     """
#     assert type(POINTS) == str, "param POINTS should be str type!"
#     assert type(FIELD) == str, "param FIELD should be str type!"
#     if TARGET_TEMPLATE:
#         assert type(TARGET_TEMPLATE) == str, "param TARGET_TEMPLATE should be str type!"
#     assert type(CV_METHOD) == int, "param CV_METHOD should be int type!"
#     assert CV_SAMPLES>=0, "param CV_SAMPLES should follow minimum rule!"
#     assert type(CV_SAMPLES) == int, "param CV_SAMPLES should be int type!"
#     assert type(TARGET_DEFINITION) == int, "param TARGET_DEFINITION should be int type!"
#     assert TARGET_USER_SIZE>=0.000000, "param TARGET_USER_SIZE should follow minimum rule!"
#     assert type(TARGET_USER_SIZE) in [float, int], "param TARGET_USER_SIZE should be float type!"
#     assert type(TARGET_USER_XMIN) in [float, int], "param TARGET_USER_XMIN should be float type!"
#     assert type(TARGET_USER_XMAX) in [float, int], "param TARGET_USER_XMAX should be float type!"
#     assert type(TARGET_USER_YMIN) in [float, int], "param TARGET_USER_YMIN should be float type!"
#     assert type(TARGET_USER_YMAX) in [float, int], "param TARGET_USER_YMAX should be float type!"
#     assert type(TARGET_USER_FITS) == int, "param TARGET_USER_FITS should be int type!"
#     assert type(SEARCH_RANGE) == int, "param SEARCH_RANGE should be int type!"
#     assert SEARCH_RADIUS>=0.000000, "param SEARCH_RADIUS should follow minimum rule!"
#     assert type(SEARCH_RADIUS) in [float, int], "param SEARCH_RADIUS should be float type!"
#     assert type(SEARCH_POINTS_ALL) == int, "param SEARCH_POINTS_ALL should be int type!"
#     assert SEARCH_POINTS_MIN>=0, "param SEARCH_POINTS_MIN should follow minimum rule!"
#     assert type(SEARCH_POINTS_MIN) == int, "param SEARCH_POINTS_MIN should be int type!"
#     assert SEARCH_POINTS_MAX>=0, "param SEARCH_POINTS_MAX should follow minimum rule!"
#     assert type(SEARCH_POINTS_MAX) == int, "param SEARCH_POINTS_MAX should be int type!"
#     assert type(SEARCH_DIRECTION) == int, "param SEARCH_DIRECTION should be int type!"
#     assert type(DW_WEIGHTING) == int, "param DW_WEIGHTING should be int type!"
#     assert DW_IDW_POWER>=0.000000, "param DW_IDW_POWER should follow minimum rule!"
#     assert type(DW_IDW_POWER) in [float, int], "param DW_IDW_POWER should be float type!"
#     assert type(DW_IDW_OFFSET) == bool, "param DW_IDW_OFFSET should be bool type!"
#     assert DW_BANDWIDTH>=0.000000, "param DW_BANDWIDTH should follow minimum rule!"
#     assert type(DW_BANDWIDTH) in [float, int], "param DW_BANDWIDTH should be float type!"
#     POINTS = str(POINTS)
#     FIELD = str(FIELD)
#     if TARGET_TEMPLATE:
#         TARGET_TEMPLATE = str(TARGET_TEMPLATE)
#     CV_METHOD = str(CV_METHOD)
#     CV_SAMPLES = str(CV_SAMPLES)
#     TARGET_DEFINITION = str(TARGET_DEFINITION)
#     TARGET_USER_SIZE = str(TARGET_USER_SIZE)
#     TARGET_USER_XMIN = str(TARGET_USER_XMIN)
#     TARGET_USER_XMAX = str(TARGET_USER_XMAX)
#     TARGET_USER_YMIN = str(TARGET_USER_YMIN)
#     TARGET_USER_YMAX = str(TARGET_USER_YMAX)
#     TARGET_USER_FITS = str(TARGET_USER_FITS)
#     SEARCH_RANGE = str(SEARCH_RANGE)
#     SEARCH_RADIUS = str(SEARCH_RADIUS)
#     SEARCH_POINTS_ALL = str(SEARCH_POINTS_ALL)
#     SEARCH_POINTS_MIN = str(SEARCH_POINTS_MIN)
#     SEARCH_POINTS_MAX = str(SEARCH_POINTS_MAX)
#     SEARCH_DIRECTION = str(SEARCH_DIRECTION)
#     DW_WEIGHTING = str(DW_WEIGHTING)
#     DW_IDW_POWER = str(DW_IDW_POWER)
#     DW_IDW_OFFSET = str(int(DW_IDW_OFFSET))
#     DW_BANDWIDTH = str(DW_BANDWIDTH)
#     os.mkdir(os.path.join(DOWNLOAD_DIR, "TARGET_OUT_GRID"))
#     TARGET_OUT_GRID = os.path.join(DOWNLOAD_DIR, "TARGET_OUT_GRID", "TARGET_OUT_GRID")
#     os.mkdir(os.path.join(DOWNLOAD_DIR, "CV_SUMMARY"))
#     CV_SUMMARY = os.path.join(DOWNLOAD_DIR, "CV_SUMMARY", "CV_SUMMARY")
#     os.mkdir(os.path.join(DOWNLOAD_DIR, "CV_RESIDUALS"))
#     CV_RESIDUALS = os.path.join(DOWNLOAD_DIR, "CV_RESIDUALS", "CV_RESIDUALS")
#     calls = saga_cmd_fp + " grid_gridding 1" + " -POINTS=" + POINTS + " -FIELD=\"" + FIELD + "\"" + " -CV_METHOD=" + CV_METHOD + " -CV_SAMPLES=" + CV_SAMPLES + " -TARGET_DEFINITION=" + TARGET_DEFINITION + " -TARGET_USER_SIZE=" + TARGET_USER_SIZE + " -TARGET_USER_XMIN=" + TARGET_USER_XMIN + " -TARGET_USER_XMAX=" + TARGET_USER_XMAX + " -TARGET_USER_YMIN=" + TARGET_USER_YMIN + " -TARGET_USER_YMAX=" + TARGET_USER_YMAX + " -TARGET_USER_FITS=" + TARGET_USER_FITS + " -TARGET_OUT_GRID=" + TARGET_OUT_GRID + " -SEARCH_RANGE=" + SEARCH_RANGE + " -SEARCH_RADIUS=" + SEARCH_RADIUS + " -SEARCH_POINTS_ALL=" + SEARCH_POINTS_ALL + " -SEARCH_POINTS_MIN=" + SEARCH_POINTS_MIN + " -SEARCH_POINTS_MAX=" + SEARCH_POINTS_MAX + " -SEARCH_DIRECTION=" + SEARCH_DIRECTION + " -DW_WEIGHTING=" + DW_WEIGHTING + " -DW_IDW_POWER=" + DW_IDW_POWER + " -DW_IDW_OFFSET=" + DW_IDW_OFFSET + " -DW_BANDWIDTH=" + DW_BANDWIDTH + ""
#     if CV_SUMMARY: calls += " -CV_SUMMARY=" + CV_SUMMARY
#     if CV_RESIDUALS: calls += " -CV_RESIDUALS=" + CV_RESIDUALS
#     if TARGET_TEMPLATE: calls += " -TARGET_TEMPLATE=" + TARGET_TEMPLATE
#     print(calls)
#     p = subprocess.Popen(calls, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#     out, err = p.communicate()
#     return out, err
