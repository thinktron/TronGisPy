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
    dtype_gdal = ds.GetRasterBand(1).DataType
    no_data_value = ds.GetRasterBand(1).GetNoDataValue()
    ds = None 
    return cols, rows, bands, geo_transform, projection, dtype_gdal, no_data_value

def get_nparray(fp, opencv_shape=True):
    """if opencv_shape the shape will be (cols, rows, bnads), else (bnads, cols, rows)"""
    ds = gdal.Open(fp)
    X = ds.ReadAsArray()
    ds = None 
    if not opencv_shape:
        return X
    else:
        if len(X.shape) == 2:
            X = X.reshape(-1, *X.shape)
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


def write_output_tif(X, dst_tif_path, bands, cols, rows, geo_transform, projection):
    assert len(X.shape) == 3, "please reshape it to (n_rows, n_cols, n_bands)"
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, cols, rows, bands, gdal.GDT_Int32) # dst_filename, xsize=512, ysize=512, bands=1, eType=gdal.GDT_Byte
    dst_ds.SetGeoTransform(geo_transform)
    dst_ds.SetProjection(projection)

    for b in range(bands):
        band = dst_ds.GetRasterBand(b+1)
        band.WriteArray(X[:, :, b], 0, 0)

    band.FlushCache()
    band.SetNoDataValue(-99)
    dst_ds = None

def get_testing_fp():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    satellite_tif_path = os.path.join(base_dir, 'data', 'P0015913_SP5_006_001_002_021_002_005.tif')
    return os.path.abspath(satellite_tif_path)


def clip_image_by_shp(src_image_path, src_shp_path, dst_image_path):
    result = gdal.Warp(dst_image_path,
                       src_image_path,
                       cutlineDSName=src_shp_path,
                       cropToCutline=True)
    result = None

def tif_composition(ref_tif_image, src_tif_paths, dst_tif_path):
    """
    crs_image: should be used to create the canvas with final coordinate system, geo_transform and projection, 
    src_tifs: should be in list type with elements with full path of tif images.
    dst_tif_path: output file path
    """
    # get geo info
    cols, rows, bands, geo_transform, projection, dtype_gdal, no_data_value = get_geo_info(ref_tif_image)
    
    # cal bands count
    bands_for_each_tif = [get_geo_info(tif_path)[2] for tif_path in src_tif_paths]
    bands = sum(bands_for_each_tif)

    # bands compositions: create new tif
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, cols, rows, bands, dtype_gdal)
    dst_ds.SetGeoTransform(geo_transform)
    dst_ds.SetProjection(projection)
    
    # bands compositions: write bands
    band_num = 1
    for tif_path, bands_for_the_tif in zip(src_tif_paths, bands_for_each_tif):
        nparr = get_nparray(tif_path)
        for band_num_for_the_tif in range(bands_for_the_tif):
            band = dst_ds.GetRasterBand(band_num)
            band.WriteArray(nparr[:, :, band_num_for_the_tif], 0, 0)
            band.FlushCache()
            band.SetNoDataValue(no_data_value)
            band_num += 1
    dst_ds = None