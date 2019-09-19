import os
import numpy as np
import gdal

# bands compositions
def get_geo_info(fp):
    """cols, rows, bands, geo_transform, projection, dtype_gdal = get_geo_info(fp)"""
    ds = gdal.Open(fp)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount 
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    dtype_gdal = ds.GetRasterBand(1).DataType # gdal.GetDataTypeName(self.dtype_gdal)
    ds = None 
    return cols, rows, bands, geo_transform, projection, dtype_gdal

def get_nparray(fp, opencv_shape=True):
    """if opencv_shape the shape will be (cols, rows, bnads), else (bnads, cols, rows)"""
    ds = gdal.Open(fp)
    X = ds.ReadAsArray()
    ds = None 
    if not opencv_shape:
        return X
    else:
        return np.transpose(X, axes=[1,2,0])

def get_extend(fp):
    """get the extend(boundry) coordnate"""
    ds = gdal.Open(fp)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    gt = ds.GetGeoTransform()
    extend=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            extend.append([x,y])
        yarr.reverse()
    return extend

def get_testing_fp():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    satellite_tif_path = os.path.join(base_dir, 'data', 'P0015913_SP5_006_001_002_021_002_005.tif')
    return os.path.abspath(satellite_tif_path)