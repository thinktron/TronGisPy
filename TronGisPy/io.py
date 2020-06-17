import os
import gdal
import time
import shutil
import numpy as np
import TronGisPy as tgp
from TronGisPy import ShapeGrid

# operation on raster files
# ===========================
def get_raster_info(fp, attributes=["rows", "cols", "bands", "geo_transform", "projection", "gdaldtype", "no_data_value", "metadata"]):
    """
    rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = get_raster_info(fp)
    """
    ds = gdal.Open(fp)
    rows, cols, bands = ds.RasterYSize, ds.RasterXSize, ds.RasterCount
    geo_transform, projection, metadata = ds.GetGeoTransform(), ds.GetProjection(), ds.GetMetadata()
    gdaldtype = ds.GetRasterBand(1).DataType
    no_data_value = ds.GetRasterBand(1).GetNoDataValue()
    ds = None 

    geo_info = rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata
    default_attributes = ["rows", "cols", "bands", "geo_transform", "projection", "gdaldtype", "no_data_value", "metadata"]
    if type(attributes) == list:
        attributes_idxs = [default_attributes.index(attribute) for attribute in attributes]
        return [geo_info[attributes_idx] for attributes_idx in attributes_idxs]
    else:
        attributes_idx = default_attributes.index(attributes)
        return geo_info[attributes_idx]

def get_raster_data(fp):
    """output shape will be (rows, cols, bnads)rows"""
    ds = gdal.Open(fp)
    data = ds.ReadAsArray()
    ds = None 
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    return np.transpose(data, axes=[1,2,0])

def get_raster_extent(fp, return_poly=True):
    """get the extent(boundry) coordnate"""
    rows, cols, geo_transform = get_raster_info(fp, ['rows', 'cols', 'geo_transform'])
    return tgp.get_extent(rows, cols, geo_transform, return_poly)

def update_raster_info(fp, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
    all_none = geo_transform is None and projection is None and gdaldtype is None and no_data_value is None and metadata is None
    assert not all_none, "at least one of geo_transform and projection params should not be None!"
    ds = gdal.Open(fp, gdal.GA_Update)
    if geo_transform is not None:
        ds.SetGeoTransform(geo_transform)
    if projection is not None:
        ds.SetProjection(projection)
    if metadata is not None:
        ds.SetMetadata(metadata)
    if no_data_value is not None:
        for b in range(ds.RasterCount):
            band = ds.GetRasterBand(b+1)
            band.SetNoDataValue(no_data_value)
            band.FlushCache()
    ds = None 


# operation on Raster class
# ===========================
def read_raster(fp):
    data = get_raster_data(fp)
    rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = get_raster_info(fp)
    from TronGisPy import Raster
    return Raster(data, geo_transform, projection, gdaldtype, no_data_value, metadata)

def write_raster(fp, data, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
    """data should be in (n_rows, n_cols, n_bands) shape"""
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=2)
    rows, cols, bands = data.shape
    gdaldtype = tgp.npdtype_to_gdaldtype(data.dtype) if gdaldtype is None else gdaldtype
    ds = gdal.GetDriverByName('GTiff').Create(fp, cols, rows, bands, gdaldtype) # dst_filename, xsize=512, ysize=512, bands=1, eType=gdal.GDT_Byte
    if geo_transform is not None:
        ds.SetGeoTransform(geo_transform)
    if projection is not None:
        ds.SetProjection(projection)
    if metadata is not None:
        ds.SetMetadata(metadata)

    for b in range(bands):
        band = ds.GetRasterBand(b+1)
        band.WriteArray(data[:, :, b], 0, 0)
        if no_data_value is not None:
            band.SetNoDataValue(no_data_value)
        band.FlushCache()
    ds = None

def read_gdal_ds(ds):
    rows, cols, bands = ds.RasterYSize, ds.RasterXSize, ds.RasterCount
    geo_transform, projection, metadata = ds.GetGeoTransform(), ds.GetProjection(), ds.GetMetadata()
    gdaldtype = ds.GetRasterBand(1).DataType
    no_data_value = ds.GetRasterBand(1).GetNoDataValue()
    data = ds.ReadAsArray()
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    data = np.transpose(data, axes=[1,2,0])
    from TronGisPy import Raster
    return Raster(data, geo_transform, projection, gdaldtype, no_data_value, metadata)


def write_gdal_ds(data=None, bands=None, cols=None, rows=None, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
    """data should be in (n_rows, n_cols, n_bands) shape"""
    if data is None:
        assert (bands is not None) and (cols is not None) and (rows is not None), "bands, cols, rows should not be None"
        assert (gdaldtype is not None), "gdaldtype should not be None"
    else:    
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        bands = data.shape[2] if bands is None else bands
        cols = data.shape[1] if cols is None else cols
        rows = data.shape[0] if rows is None else rows
        gdaldtype = tgp.npdtype_to_gdaldtype(data.dtype) if gdaldtype is None else gdaldtype

    ds = gdal.GetDriverByName('MEM').Create('', cols, rows, bands, gdaldtype) # dst_filename, xsize=512, ysize=512, bands=1, eType=gdal.GDT_Byte
    if geo_transform is not None:
        ds.SetGeoTransform(geo_transform)
    if projection is not None:
        ds.SetProjection(projection)
    if metadata is not None:
        ds.SetMetadata(metadata)

    if no_data_value is not None:
        for b in range(bands):
            band = ds.GetRasterBand(b+1)
            band.SetNoDataValue(no_data_value)
            band.WriteArray(np.full((rows, cols), no_data_value), 0, 0)
            band.FlushCache()
    
    if data is not None:
        for b in range(data.shape[2]):
            band = ds.GetRasterBand(b+1)
            band.WriteArray(data[:, :, b], 0, 0)
            band.FlushCache()
    return ds

def remove_shp(shp_fp):
    assert shp_fp.endswith('.shp'), 'shp_fp should be ends with ".shp"'
    dst_shp_fp = os.path.abspath(shp_fp)
    base_dir = os.path.split(dst_shp_fp)[0]
    print(len(os.listdir(base_dir)))
    shp_fn = os.path.split(dst_shp_fp)[-1].split(".")[0]
    del_fps = [os.path.join(base_dir, f) for f in  os.listdir(base_dir) if shp_fn == f.split('.')[0]]
    for fp in del_fps:
        os.remove(fp)
    print(len(os.listdir(base_dir)))

# testing data
# ===========================
def get_testing_fp(fn):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    if fn == 'satellite_tif':
        fp = os.path.join(data_dir, 'satellite_tif', 'satellite_tif.tif')
    elif fn == 'satellite_tif_clipper':
        fp = os.path.join(data_dir, 'satellite_tif_clipper', 'satellite_tif_clipper.shp')
    elif fn == 'satellite_tif_kmeans':
        fp = os.path.join(data_dir, 'satellite_tif_kmeans', 'satellite_tif_kmeans.tif')
    elif fn == 'rasterized_image':
        fp = os.path.join(data_dir, 'rasterized_image', 'rasterized_image.tif')
    elif fn == 'rasterized_image_1':
        fp = os.path.join(data_dir, 'rasterized_image', 'rasterized_image_1.tif')
    elif fn == 'poly_to_be_clipped':
        fp = os.path.join(data_dir, 'clip_shp_by_shp', 'poly_to_be_clipped.shp')
    elif fn == 'point_to_be_clipped':
        fp = os.path.join(data_dir, 'clip_shp_by_shp', 'point_to_be_clipped.shp')
    elif fn == 'line_to_be_clipped':
        fp = os.path.join(data_dir, 'clip_shp_by_shp', 'line_to_be_clipped.shp')
    elif fn == 'multiline_to_be_clipped':
        fp = os.path.join(data_dir, 'clip_shp_by_shp', 'multiline_to_be_clipped.shp')
    elif fn == 'shp_clipper':
        fp = os.path.join(data_dir, 'clip_shp_by_shp', 'shp_clipper.shp')
    elif fn == 'remap_rgb_clipper_path':
        fp = os.path.join(data_dir, 'remap', 'rgb_3826_clipper.tif')
    elif fn == 'remap_ndvi_path':
        fp = os.path.join(data_dir, 'remap', 'ndvi_32651.tif')
    elif fn == 'dem_process_path':
        fp = os.path.join(data_dir, 'dem_tif', 'crop_dem.tif')
    elif fn == 'tif_forinterpolation':
        fp = os.path.join(data_dir, 'interpolation', 'X_forinterpolation.tif')
    elif fn == 'aero_triangulation_PXYZs':
        fp = os.path.join(data_dir, 'aero_triangulation', 'P_XYZs.npy')
    else:
        assert False, "cannot find the file!!"
    return os.path.abspath(fp)


def create_temp_dir():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.abspath(os.path.join(base_dir, 'temp'))
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    else:
        shutil.rmtree(temp_dir)
        time.sleep(0.5)
        os.mkdir(temp_dir)
    return temp_dir

def remove_temp_dir():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.abspath(os.path.join(base_dir, 'temp'))
    shutil.rmtree(temp_dir)
    return temp_dir

