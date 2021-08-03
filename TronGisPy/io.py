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
    """Get the geo-information of the raster file including rows, cols, bands, geo_transform, 
    projection, gdaldtype, no_data_value, metadata attributes.

    Parameters
    ----------
    fp: str
        File path of the raster file.
    attributes: list or str
        Which attributes to get e.g. ["rows", "cols", "bands", "geo_transform"].

    Returns
    -------
    geo_info : list or attribute
        Attributes of the raster.

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> raster_fp = tgp.get_testing_fp()
    >>> rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = tgp.get_raster_info(raster_fp)
    >>> rows, cols, bands
    (677, 674, 3)
    >>> tgp.get_raster_info(raster_fp, 'projection')
    'PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3826"]]'
    >>> tgp.get_raster_info(raster_fp, ['rows', 'cols'])
    [677, 674]
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
    """Get the digital number of the raster file.

    Parameters
    ----------
    fp: str
        File path of the raster file.

    Returns
    -------
    data : ndarray
        Digital number of the raster file whose shape will be (rows, cols, bnads).

    Examples
    --------
    >>> import TronGisPy as tgp 
    >>> raster_fp = tgp.get_testing_fp()
    >>> data = tgp.get_raster_data(raster_fp)
    >>> type(data) 
    <class 'numpy.ndarray'>
    >>> data.shape 
    (677, 674, 3)
    """
    ds = gdal.Open(fp)
    data = ds.ReadAsArray()
    ds = None 
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    return np.transpose(data, axes=[1,2,0])

def get_raster_extent(fp, return_type='poly'):
    """Get the boundary of the raster file.

    Parameters
    ----------
    fp: str
        File path of the raster file.
    return_type: {'poly', 'plot', 'gdal'}
        If 'poly', return four corner coordinates. If plot, return (xmin, xmax, ymin, ymax). If 'gdal', return (xmin, ymin, xmax, ymax). 

    Returns
    -------
    extent: ndarray or tuple
        Depends on return_type. If 'poly', return four corner coordinates. If plot, return (xmin, xmax, ymin, ymax). If 'gdal', return (xmin, ymin, xmax, ymax). 

    Examples
    --------
    >>> import TronGisPy as tgp 
    >>> from shapely.geometry import Polygon
    >>> raster_fp = tgp.get_testing_fp() 
    >>> extent = tgp.get_raster_extent(raster_fp) 
    >>> Polygon(extent).area
    570976.9697303267
    """
    rows, cols, geo_transform = get_raster_info(fp, ['rows', 'cols', 'geo_transform'])
    return tgp.get_extent(rows, cols, geo_transform, return_type)

def update_raster_info(fp, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
    """Update the geo-information of the raster file including geo_transform, projection, gdaldtype, 
    no_data_value, metadata attributes.

    Parameters
    ----------
    fp: str
        File path of the raster file.
    geo_transform: tuple or list
        Affine transform parameters (c, a, b, f, d, e = geo_transform). 
    projection: str, optional
        The well known text (WKT) of the raster which can be generated from 
        `TronGisPy.epsg_to_wkt(<epsg_code>)`
    gdaldtype: int, optional
        The type of the cell defined in gdal which will affect the information 
        to be stored when saving the file. This can be generate from `gdal.GDT_XXX` 
        such as `gdal.GDT_Int32` equals 5 and `gdal.GDT_Float32` equals 6.
    no_data_value: int or float, optional
        Define which value to replace nan in numpy array when saving a raster file.
    metadata: dict, optional
        Define the metadata of the raster file.
    """
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
def read_raster(fp, fill_na=False):
    """Read raster file as `TronGisPy.Raster` object.

    Parameters
    ----------
    fp: str 
        File path of the raster file.

    Returns
    -------
    raster: Raster
        output raster.

    Examples
    --------
    >>> import TronGisPy as tgp 
    >>> from shapely.geometry import Polygon
    >>> raster_fp = tgp.get_testing_fp() 
    >>> raster = tgp.read_raster(raster_fp) 
    >>> raster
    shape: (677, 674, 3)
    geo_transform: (271982.8783, 1.1186219584569888, 0.0, 2769973.0653, 0.0, -1.1186305760705852)
    projection: PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3826"]]
    no_data_value: -32768.0
    metadata: {'AREA_OR_POINT': 'Area'}
    """
    from TronGisPy import Raster
    data = get_raster_data(fp)
    rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = get_raster_info(fp)
    raster = Raster(data, geo_transform, projection, gdaldtype, no_data_value, metadata)
    if fill_na and np.sum(np.isnan(raster.data)):
        raster.fill_na()
    return raster

def write_raster(fp, data, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
    """Write raster file.

    Parameters
    ----------
    fp: str
        File path of the raster file.
    geo_transform: tuple or list, optional
        Affine transform parameters (c, a, b, f, d, e = geo_transform). 
    projection: str, optional
        The well known text (WKT) of the raster which can be generate from `TronGisPy.epsg_to_wkt(<epsg_code>)`
    gdaldtype: int, optional
        The type of the cell defined in gdal which will affect the information 
        to be stored when saving the file. This can be generate from `gdal.GDT_XXX` 
        such as `gdal.GDT_Int32` equals 5 and `gdal.GDT_Float32` equals 6.
    no_data_value: int or float, optional
        Define which value to replace nan in numpy array when saving a raster file.
    metadata: dict, optional
        Define the metadata of the raster file.
    """
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
    """Read gdal DataSource as `TronGisPy.Raster` object.

    Parameters
    ----------
    ds: gdal.DataSource.

    Returns
    -------
    raster: Raster.
        output raster.
    """
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
    """Build the gdal DataSource from geo-information attributes.

    Parameters
    ----------
    data: array_like, optional
        The digital number for each cell of the raster.  Data is in 
        (n_rows, n_cols, n_bands) shape.
    bands: int, optional
        Number of bands.
    cols: int, optional
        Number of cols.
    rows: int, optional
        Number of rows.
    geo_transform: tuple or list, optional
        Affine transform parameters (c, a, b, f, d, e = geo_transform). 
    projection: str, optional
        The well known text (WKT) of the raster which can be generate 
        from `TronGisPy.epsg_to_wkt(<epsg_code>)`
    gdaldtype: int, optional
        The type of the cell defined in gdal which will affect the information 
        to be stored when saving the file. This can be generate from `gdal.GDT_XXX` 
        such as `gdal.GDT_Int32` equals 5 and `gdal.GDT_Float32` equals 6.
    no_data_value: int or float, optional
        Define which value to replace nan in numpy array when saving a raster file.
    metadata: dict, optional
        Define the metadata of the raster file.

    Returns
    -------
    raster: Raster.
        output raster.

    Examples
    --------
    >>> import TronGisPy as tgp 
    >>> raster_fp = tgp.get_testing_fp() 
    >>> data = tgp.get_raster_data(raster_fp)
    >>> geo_transform, projection, gdaldtype, no_data_value = tgp.get_raster_info(raster_fp, ["geo_transform", "projection", "gdaldtype", "no_data_value"])
    >>> ds = tgp.write_gdal_ds(data, geo_transform=geo_transform, projection=projection, gdaldtype=gdaldtype, no_data_value=no_data_value)
    >>> raster = tgp.read_gdal_ds(ds)
    >>> raster
    shape: (677, 674, 3)
    geo_transform: (271982.8783, 1.1186219584569888, 0.0, 2769973.0653, 0.0, -1.1186305760705852)
    projection: PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3826"]]
    no_data_value: -32768.0
    metadata: {}
    """
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
    """Remove shapefile and all its related files.

    Parameters
    ----------
    shp_fp: str 
        File path of the shapefile.
    """
    assert shp_fp.endswith('.shp'), 'shp_fp should be ends with ".shp"'
    dst_shp_fp = os.path.abspath(shp_fp)
    base_dir = os.path.split(dst_shp_fp)[0]
    shp_fn = os.path.split(dst_shp_fp)[-1].split(".")[0]
    del_fps = [os.path.join(base_dir, f) for f in  os.listdir(base_dir) if shp_fn == f.split('.')[0]]
    for fp in del_fps:
        os.remove(fp)

# testing data
# ===========================
def get_testing_fp(fn=None):
    """Get the testing file built-in TronGisPy.

    Parameters
    ----------
    fn: str
        Choice candidates includes satellite_tif, satellite_tif_clipper,
        satellite_tif_kmeans, rasterized_image, rasterized_image_1,
        poly_to_be_clipped, point_to_be_clipped, line_to_be_clipped,
        multiline_to_be_clipped, shp_clipper, remap_rgb_clipper_path,
        remap_ndvi_path, dem_process_path, tif_forinterpolation,
        aero_triangulation_PXYZs

    Returns
    -------
    fp: str
        The path of testing file.

    Examples
    --------
    >>> import TronGisPy as tgp 
    >>> raster_fp = tgp.get_testing_fp() 
    >>> raster = tgp.read_raster(raster_fp)
    >>> raster.plot()
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    if fn is None:
        fp = os.path.join(data_dir, 'remap', 'rgb_3826_clipper.tif')
    elif fn == 'satellite_tif':
        fp = os.path.join(data_dir, 'satellite_tif', 'satellite_tif.tif')
    elif fn == 'satellite_tif_clipper':
        fp = os.path.join(data_dir, 'satellite_tif_clipper', 'satellite_tif_clipper.shp')
    elif fn == 'aereo_tif':
        fp = os.path.join(data_dir, 'aereo_tif', '131129d_29_0263_refined.tif') # 131129d_29_0263\ras_aerial_img_refined.tif
    elif fn == 'aereo_tif_clipper':
        fp = os.path.join(data_dir, 'aereo_tif_clipper', '131129d_29_0263_clipper.shp')
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
    elif fn == 'flipped_gt':
        fp = os.path.join(data_dir, 'flipped_gt', 'flipped_gt.tif')
    elif fn == 'rotate_tif':
        fp = os.path.join(data_dir, 'rotate_tif', 'rotate_tif.tif')
    elif fn == 'multiple_poly_clipper':
        fp = os.path.join(data_dir, 'multiple_poly_clipping', 'df_farm_clipped.shp')
    elif fn == 'multiple_poly_clip_ras':
        fp = os.path.join(data_dir, 'multiple_poly_clipping', 'ras_sat_clipped.tif')
    elif fn == 'norm':
        fp = os.path.join(data_dir, 'norm', '100131i_39_0049_1527_052_019.tif')
    else:
        assert False, "cannot find the file!!"
    return os.path.abspath(fp)


# def create_temp_dir():
#     base_dir = os.path.dirname(os.path.realpath(__file__))
#     temp_dir = os.path.abspath(os.path.join(base_dir, 'temp'))
#     if not os.path.isdir(temp_dir):
#         os.mkdir(temp_dir)
#     else:
#         shutil.rmtree(temp_dir)
#         time.sleep(0.5)
#         os.mkdir(temp_dir)
#     return temp_dir

# def remove_temp_dir():
#     base_dir = os.path.dirname(os.path.realpath(__file__))
#     temp_dir = os.path.abspath(os.path.join(base_dir, 'temp'))
#     shutil.rmtree(temp_dir)
#     return temp_dir


def create_temp_dir_when_not_exists():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.abspath(os.path.join(base_dir, 'temp'))
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    return temp_dir

