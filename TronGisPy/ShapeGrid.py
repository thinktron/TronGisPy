import os
import cv2
import ogr
import gdal
import pyproj
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import TronGisPy as tgp
from TronGisPy import CRS
def get_rasterize_layer_params(src_vector, res=5):
    """Get params for rasterize_layer if you don't have a grid system. 

    Parameters
    ----------
    src_vector: Geopandas.GeoDataFrame
        The vector data to be rasterize.
    res: resolution
        The resolution (in meters) of the grid.

    Returns
    -------
    rows: int
        Target rasterized image's rows.
    cols: int
        Target rasterized image's cols.
    geo_transform: tuple
        Target rasterized image's geo_transform which is the affine parameter.
    """
    xmin, ymin, xmax, ymax = src_vector.total_bounds 
    geo_transform = (xmin, res, 0, ymax, 0, -res)
    cols = int((xmax - xmin) / res) + 1
    rows = int((ymax - ymin) / res) + 1
    return rows, cols, geo_transform

def rasterize_layer(src_vector, rows, cols, geo_transform, use_attribute, all_touched=False, no_data_value=0):
    """Rasterize vector data. Get the cell value in defined grid (rows, cols, geo_transform)
    from its overlapped polygon.

    Parameters
    ----------
    src_vector: Geopandas.GeoDataFrame
        Which vector data to be rasterize.
    rows: int
        Target rasterized image's rows.
    cols: int
        Target rasterized image's cols.
    geo_transform: tuple
        Target rasterized image's geo_transform which is the affine parameter.
    use_attribute: str
        The column to use as rasterized image value.
    all_touched: bool, optioonal, default: False
        Pixels that touch (not overlap over 50%) the polygon will be assign the use_attribute value of the polygon.
    no_data_value: int or float
        The pixels not covered by any polygon will be filled no_data_value.

    Returns
    -------
    raster: Raster. 
        Rasterized result.

    Examples
    -------- 
    >>> import geopandas as gpd
    >>> import TronGisPy as tgp 
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> ref_raster_fp = tgp.get_testing_fp('satellite_tif') # get the geoinfo from the raster
    >>> src_vector_fp = tgp.get_testing_fp('satellite_tif_clipper') # read source shapefile as GeoDataFrame
    >>> src_vector = gpd.read_file(src_vector_fp)
    >>> src_vector['FEATURE'] = 1 # make the value to fill in the raster cell
    >>> rows, cols, geo_transform = tgp.get_raster_info(ref_raster_fp, ['rows', 'cols', 'geo_transform'])
    >>> raster = ShapeGrid.rasterize_layer(src_vector, rows, cols, geo_transform, use_attribute='FEATURE', no_data_value=99)
    >>> fig, (ax1, ax2) = plt.subplots(1,2) # plot the result
    >>> tgp.read_raster(ref_raster_fp).plot(ax=ax1)
    >>> src_vector.plot(ax=ax1)
    >>> ax1.set_title('polygon with ref_raster')
    >>> raster.plot(ax=ax2)
    >>> ax2.set_title('rasterized image')
    >>> plt.show()
    """
    # Open your shapefile
    assert type(src_vector) is gpd.GeoDataFrame, "src_vector should be GeoDataFrame type."
    assert use_attribute in src_vector.columns, "attribute not exists in src_vector."
    gdaldtype = tgp.npdtype_to_gdaldtype(src_vector[use_attribute].dtype)
    # projection = src_vector.crs.to_wkt() if src_vector.crs is not None else None
    projection = pyproj.CRS(src_vector.crs).to_wkt() if src_vector.crs is not None else None
    src_shp_ds = ogr.Open(src_vector.to_json())
    src_shp_layer = src_shp_ds.GetLayer()

    # Create the destination raster data source
    ds = tgp.write_gdal_ds(bands=1, cols=cols, rows=rows, geo_transform=geo_transform, gdaldtype=gdaldtype, no_data_value=no_data_value)

    # set it to the attribute that contains the relevant unique
    options = ["ATTRIBUTE="+use_attribute]
    if all_touched:
        options.append('ALL_TOUCHED=TRUE')
    gdal.RasterizeLayer(ds, [1], src_shp_layer, options=options) # target_ds, band_list, source_layer, options = options

    data = ds.GetRasterBand(1).ReadAsArray()
    raster = tgp.Raster(data, geo_transform, projection, gdaldtype, no_data_value)
    return raster


def rasterize_layer_from_ref_raster(src_vector, ref_raster, use_attribute, all_touched=False, no_data_value=0):
    """Rasterize vector data. Get the cell value in defined grid of ref_raster
    from its overlapped polygon.

    Parameters
    ----------
    src_vector: Geopandas.GeoDataFrame
        Which vector data to be rasterize.
    ref_raster: Raster
        Target rasterized image's rows, cols, and geo_transform.
    use_attribute: str
        The column to use as rasterized image value.
    all_touched: bool, optioonal, default: False
        Pixels that touch (not overlap over 50%) the polygon will be assign the use_attribute value of the polygon.
    no_data_value: int or float
        The pixels not covered by any polygon will be filled no_data_value.

    Returns
    -------
    raster: Raster. 
        Rasterized result.

    Examples
    -------- 
    >>> import geopandas as gpd
    >>> import TronGisPy as tgp 
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> ref_raster_fp = tgp.get_testing_fp('satellite_tif') # get the geoinfo from the raster
    >>> src_vector_fp = tgp.get_testing_fp('satellite_tif_clipper') # read source shapefile as GeoDataFrame
    >>> src_vector = gpd.read_file(src_vector_fp)
    >>> src_vector['FEATURE'] = 1 # make the value to fill in the raster cell
    >>> ref_raster = tgp.read_raster(ref_raster_fp)
    >>> raster = ShapeGrid.rasterize_layer_from_ref_raster(src_vector, ref_raster, use_attribute='FEATURE', no_data_value=99)
    >>> fig, (ax1, ax2) = plt.subplots(1,2) # plot the result
    >>> tgp.read_raster(ref_raster_fp).plot(ax=ax1)
    >>> src_vector.plot(ax=ax1)
    >>> ax1.set_title('polygon with ref_raster')
    >>> raster.plot(ax=ax2)
    >>> ax2.set_title('rasterized image')
    >>> plt.show()
    """
    # Open your shapefile
    assert type(src_vector) is gpd.GeoDataFrame, "src_vector should be GeoDataFrame type."
    assert use_attribute in src_vector.columns, "attribute not exists in src_vector."
    rows, cols, geo_transform = ref_raster.rows, ref_raster.cols, ref_raster.geo_transform
    raster = rasterize_layer(src_vector, rows, cols, geo_transform, use_attribute=use_attribute, all_touched=all_touched, no_data_value=no_data_value)
    return raster

def vectorize_layer(src_raster, band_num=1, field_name='value', multipolygon=False):
    """Vectorize raster data to achieve an acceptable raster-to-vector conversion.

    Parameters
    ----------
    src_raster: Raster
        Which raster data to be vectorize.
    band_num: int
        Which band to be vectorized.
    field_name: str, optional, default: value
        Field to be generated in output vector data.
    multipolygon: bool, optional, default: False
        Combine the polygon with the same value to be a `shapely.geometry.MultiPolygon`.

    Returns
    -------
    vector: Geopandas.GeoDataFrame
        Vectorized result.

    Examples
    -------- 
    >>> import TronGisPy as tgp
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> src_raster_fp = tgp.get_testing_fp('rasterized_image_1')
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> df_shp = ShapeGrid.vectorize_layer(src_raster)
    >>> fig, ax = plt.subplots(1, 1) # plot the result
    >>> src_raster.plot(ax=ax)
    >>> df_shp.boundary.plot(ax=ax, color='red', linewidth=5)
    >>> plt.show()
    """
    src_ds = src_raster.to_gdal_ds()
    src_band = src_ds.GetRasterBand(band_num)

    drv = ogr.GetDriverByName("MEMORY")
    dst_ds = drv.CreateDataSource('memData')
    dst_layer = dst_ds.CreateLayer('vectorize_layer')
    dst_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex(field_name)
    gdal.Polygonize(src_band, None, dst_layer, dst_field, [], callback=None)

    df_vectorized_rows = []
    for feature in dst_layer:
        field_value = feature.GetField(field_name)
        geometry = shapely.wkt.loads(feature.GetGeometryRef().ExportToWkt())
        df_vectorized_rows.append([field_value, geometry])
    df_vectorized = gpd.GeoDataFrame(df_vectorized_rows, columns=[field_name, 'geometry'], geometry='geometry', crs=src_raster.projection)
    src_ds = None
    dst_ds = None

    if multipolygon:
        from shapely.geometry import MultiPolygon
        multi_polygons = df_vectorized.groupby(field_name)['geometry'].apply(list).apply(MultiPolygon)
        values = df_vectorized.groupby(field_name)[field_name].first()
        df_vectorized = gpd.GeoDataFrame(geometry=multi_polygons)
        df_vectorized[field_name] = values

    return df_vectorized

# def clip_raster_with_polygon(src_raster, src_poly):
#     """Clip raster with polygon.

#     Parameters
#     ----------
#     src_raster: Raster
#         Which raster data to be clipped.
#     src_poly: Geopandas.GeoDataFrame
#         The clipper(clipping boundary).

#     Returns
#     -------
#     dst_raster: Raster 
#         Clipped result.

#     Examples
#     -------- 
#     >>> import geopandas as gpd
#     >>> import TronGisPy as tgp
#     >>> from TronGisPy import ShapeGrid
#     >>> from matplotlib import pyplot as plt
#     >>> src_raster_fp = tgp.get_testing_fp('satellite_tif')
#     >>> src_poly_fp = tgp.get_testing_fp('satellite_tif_clipper')
#     >>> src_raster = tgp.read_raster(src_raster_fp)
#     >>> src_poly = gpd.read_file(src_poly_fp)
#     >>> dst_raster = ShapeGrid.clip_raster_with_polygon(src_raster, src_poly)
#     >>> fig, (ax1, ax2) = plt.subplots(1, 2) # plot the result
#     >>> src_raster.plot(ax=ax1)
#     >>> src_poly.boundary.plot(ax=ax1)
#     >>> ax1.set_title('original image and clipper')
#     >>> dst_raster.plot(ax=ax2)
#     >>> ax2.set_title('clipped image')
#     >>> plt.show()
#     """
#     assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
#     src_ds = src_raster.to_gdal_ds()
#     temp_dir = tgp.create_temp_dir_when_not_exists()
#     src_shp_fp = os.path.join(temp_dir, 'src_poly.shp')
#     src_poly.to_file(src_shp_fp)
#     dst_ds = gdal.Warp('', src_ds, format= 'MEM', cutlineDSName=src_shp_fp, cropToCutline=True)
#     dst_raster = tgp.read_gdal_ds(dst_ds)
#     return dst_raster

def clip_raster_with_polygon(src_raster, src_poly, all_touched=False, no_data_value=0):
    """Clip raster with polygon.

    Parameters
    ----------
    src_raster: Raster
        Which raster data to be clipped.
    src_poly: Geopandas.GeoDataFrame
        The clipper(clipping boundary).

    Returns
    -------
    dst_raster: Raster 
        Clipped result.

    Examples
    -------- 
    >>> import geopandas as gpd
    >>> import TronGisPy as tgp
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> src_raster_fp = tgp.get_testing_fp('satellite_tif')
    >>> src_poly_fp = tgp.get_testing_fp('satellite_tif_clipper')
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> src_poly = gpd.read_file(src_poly_fp)
    >>> dst_raster = ShapeGrid.clip_raster_with_polygon(src_raster, src_poly)
    >>> fig, (ax1, ax2) = plt.subplots(1, 2) # plot the result
    >>> src_raster.plot(ax=ax1)
    >>> src_poly.boundary.plot(ax=ax1)
    >>> ax1.set_title('original image and clipper')
    >>> dst_raster.plot(ax=ax2)
    >>> ax2.set_title('clipped image')
    >>> plt.show()
    """
    assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
    src_poly_copy = src_poly.copy()
    src_poly_copy['value'] = 1
    src_poly_raster = rasterize_layer_from_ref_raster(src_poly_copy, src_raster, use_attribute='value', all_touched=all_touched, no_data_value=no_data_value)
    dst_raster = src_raster.copy()
    dst_raster.data[~(src_poly_raster.data[:, :, 0].astype(bool))] = no_data_value
    
    row_idxs, col_idxs, bands_idxs = np.where(src_poly_raster.data!=0)
    rmin, rmax, cmin, cmax = np.min(row_idxs), np.max(row_idxs), np.min(col_idxs), np.max(col_idxs)
    dst_raster.data = dst_raster.data[rmin:rmax+1, cmin:cmax+1]

    coords = tgp.npidxs_to_coords([(rmin, cmin)], src_raster.geo_transform)[0]
    geo_transform = np.array(dst_raster.geo_transform)
    geo_transform[[0, 3]] = coords
    dst_raster.geo_transform = geo_transform

    # src_ds = src_raster.to_gdal_ds()
    # temp_dir = tgp.create_temp_dir_when_not_exists()
    # src_shp_fp = os.path.join(temp_dir, 'src_poly.shp')
    # src_poly.to_file(src_shp_fp)
    # dst_ds = gdal.Warp('', src_ds, format= 'MEM', cutlineDSName=src_shp_fp, cropToCutline=True)
    # dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster

# split partitions
def __split_idxs_partitions(idxs, partitions, seed=None):
    if seed:
        np.random.seed(seed)
    partition_len = int(len(idxs) // partitions)
    parts = []
    for i in range(partitions - 1):
        part = np.random.choice(idxs, size=partition_len, replace=False)
        parts.append(part)
        idxs = list(set(idxs) - set(part))
    parts.append(idxs)
    return parts 

def clip_raster_with_multiple_polygons(src_raster, src_poly, partitions=10, return_raster=False, no_data_value=None, seed=None):
    """Clip raster with multiple polygons in the same shp as independent image.

    Parameters
    ----------
    src_raster: Raster
        Which raster data to be clipped.
    src_poly: Geopandas.GeoDataFrame
        The clipper(clipping boundary).
    partitions: int, default: 10
        The number of partitions used to split all polygons in diferent parts 
        and rasterize them in different iterations in order to avoid overlapping 
        when rasterizing.
    return_raster:bool, optional, default: False
        Return np.array if return_raster == False, else return tgp.Raster.
    no_data_value:int, optional
        Set no_data_value for clipped image. If None, use src_raster.no_data_value 
        as default. If src_raster.no_data_value is None, use 0 as default.
    seed: int, optional
        Seed to split partitions.

    Returns
    -------
    dst_raster: Raster 
        Clipped result.

    Examples
    -------- 
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> import TronGisPy as tgp
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> src_raster_fp = tgp.get_testing_fp('multiple_poly_clip_ras')
    >>> src_poly_fp = tgp.get_testing_fp('multiple_poly_clipper')
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> src_shp = gpd.read_file(src_poly_fp)
    >>> clipped_imgs = ShapeGrid.clip_raster_with_multiple_polygons(src_raster, src_shp, return_raster=True)
    >>> fig, axes = plt.subplots(2, 5, figsize=(9, 6))
    >>> axes = axes.flatten()
    >>> for idx, ax in zip(np.arange(100, 100+10, 1), axes):
    >>>     clipped_imgs[idx].plot(ax=ax)
    >>> fig.suptitle("TestShapeGrid" + ": " + "test_clip_raster_with_multiple_polygons")
    >>> plt.show()
    """
    # init resource
    df_poly_for_rasterize = src_poly.copy()
    partitions = len(src_poly) if len(src_poly) < partitions else partitions   
    df_poly_for_rasterize.loc[:, 'id'] = range(len(df_poly_for_rasterize))
    parts = __split_idxs_partitions(df_poly_for_rasterize['id'].values, partitions=partitions, seed=seed)
    if no_data_value is None:
        no_data_value = 0 if src_raster.no_data_value is None else src_raster.no_data_value

    # rasterize by its id and clipping
    clipped_imgs = []
    for ps_idx, ps in enumerate(parts): # deal with one part of poly in shp per loop: 1. rasterize => 2. find each poly in the shp
        # 1. rasterize: rasterize only df_plot['id'].isin(ps) (only id in the splitted shp)
        df_poly_for_rasterize_ps = pd.concat([df_poly_for_rasterize[df_poly_for_rasterize['id'] == p].copy() for p in ps])
        df_poly_for_rasterize_ps.loc[:, 'id_ps'] = range(len(df_poly_for_rasterize_ps))
        raster_poly_part = rasterize_layer(df_poly_for_rasterize_ps, src_raster.rows, src_raster.cols, src_raster.geo_transform, use_attribute='id_ps', all_touched=True, no_data_value=-1)
        
        for id_p in range(len(df_poly_for_rasterize_ps)):
            # 2. find each the location (in the raster) of each poly in the shp 
            coords = df_poly_for_rasterize_ps[df_poly_for_rasterize_ps['id_ps'] == id_p].total_bounds.reshape(2,2)
            npidxs = CRS.coords_to_npidxs(coords, src_raster.geo_transform)
            row_idxs_st, row_idxs_end, col_idxs_st, col_idxs_end = np.min(npidxs[:, 0]), np.max(npidxs[:, 0])+1, np.min(npidxs[:, 1]), np.max(npidxs[:, 1])+1
            clipped_img = src_raster.data[row_idxs_st:row_idxs_end, col_idxs_st:col_idxs_end].copy()
            ploy_mask = raster_poly_part.data[row_idxs_st:row_idxs_end, col_idxs_st:col_idxs_end, 0] == id_p
            if np.sum(ploy_mask) > 0:
                # generate clipped image
                clipped_img[~ploy_mask] = no_data_value
                if return_raster:
                    gt = np.array(src_raster.geo_transform)
                    gt[[0, 3]] = CRS.npidxs_to_coords([(row_idxs_st, col_idxs_st)], src_raster.geo_transform)[0]
                    clipped_img = tgp.Raster(clipped_img, tuple(gt), src_raster.projection, src_raster.gdaldtype, no_data_value, src_raster.metadata)
                clipped_imgs.append(clipped_img)
            else:
                clipped_imgs.append(None)
        
        # na_percentage = np.sum([c is None for c in clipped_imgs[-len(df_poly_for_rasterize_ps):]]) / len(df_poly_for_rasterize_ps)
        # if na_percentage != 0 : 
        #     print(ps_idx, na_percentage)
                
    clipped_imgs = [clipped_imgs[i] for i in np.argsort(np.hstack(parts))]
    return clipped_imgs

def clip_raster_with_extent(src_raster, extent):
    """Clip raster with extent.

    Parameters
    ----------
    src_raster: Raster
        Which raster data to be clipped.
    extent: tuple
        extent to clip the data with (xmin, ymin, xmax, ymax) format.

    Returns
    -------
    dst_raster: Raster. 
        Clipped result.

    Examples
    -------- 
    >>> import geopandas as gpd
    >>> import TronGisPy as tgp
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> from shapely.geometry import Polygon
    >>> src_raster_fp = tgp.get_testing_fp('satellite_tif')
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> ext = xmin, ymin, xmax, ymax = [329454.39272725, 2746809.43272727, 331715.57090906, 2748190.90181818]
    >>> dst_raster = ShapeGrid.clip_raster_with_extent(src_raster, ext)
    >>> fig, (ax1, ax2) = plt.subplots(1, 2) # plot the result
    >>> src_raster.plot(ax=ax1) 
    >>> ext_poly = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    >>> df_ext_poly = gpd.GeoDataFrame([Polygon(ext_poly)], columns=['geom'], geometry='geom')
    >>> df_ext_poly.boundary.plot(ax=ax1)
    >>> ax1.set_title('original image and clipper')
    >>> dst_raster.plot(ax=ax2)
    >>> ax2.set_title('clipped image')
    >>> plt.show()
    """
    assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
    src_ds = src_raster.to_gdal_ds()
    dst_ds = gdal.Warp('', src_ds, format= 'MEM', outputBounds=extent, cropToCutline=True)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster

def refine_resolution(src_raster, dst_resolution, resample_alg='near', extent=None, rotate=True):
    """Refine the resolution of the raster.

    Parameters
    ----------
    src_raster: Raster 
        Which raster data to be refined.
    dst_resolution: int
        Target Resolution.
    resample_alg: {'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'mode'}.
        ``near``: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).
        ``bilinear``: bilinear resampling.
        ``cubic``: cubic resampling.
        ``cubicspline``: cubic spline resampling.
        ``lanczos``: Lanczos windowed sinc resampling.
        ``average``: average resampling, computes the weighted average of all non-NODATA contributing pixels.
        ``mode``: mode resampling, selects the value which appears most often of all the sampled points.
    extent: tuple
        extent to clip the data with (xmin, ymin, xmax, ymax) format.
    rotate: bool
        If True, the function will rotate the raster and adds noData values around it to make a new rectangular image 
        matrix if the rotation in Raster.geo_transform is not zero, else it will keep the original rotation angle of 
        Raster.geo_transform. Gdal will rotate the image by default . Please refer to the issue
        https://gis.stackexchange.com/questions/256081/why-does-gdalwarp-rotate-the-data and 
        https://github.com/OSGeo/gdal/issues/1601.

    Returns
    -------
    dst_raster: Raster
        Refined result.

    Examples
    -------- 
    >>> import numpy as np
    >>> import TronGisPy as tgp
    >>> from TronGisPy import ShapeGrid
    >>> from matplotlib import pyplot as plt
    >>> src_raster_fp = tgp.get_testing_fp('dem_process_path')
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> src_raster.data[src_raster.data == -999] = np.nan
    >>> dst_raster = ShapeGrid.refine_resolution(src_raster, dst_resolution=10, resample_alg='bilinear')
    >>> fig, (ax1, ax2) = plt.subplots(1, 2) # plot the result
    >>> src_raster.plot(ax=ax1)
    >>> ax1.set_title('original dem ' + str(src_raster.shape))
    >>> dst_raster.plot(ax=ax2)
    >>> ax2.set_title('refined image ' + str(dst_raster.shape))
    >>> plt.show()
    """
    src_ds = src_raster.to_gdal_ds()

    if rotate:
        dst_ds = gdal.Warp('', src_ds, xRes=dst_resolution, yRes=dst_resolution, outputBounds=extent, format='MEM', resampleAlg=resample_alg)
        dst_raster = tgp.read_gdal_ds(dst_ds)
    else:
        assert extent is None, "you cannot set the extent when rotate == False"
        zoom_in = dst_resolution / src_raster.pixel_size[0]
        dst_ds = gdal.Warp('', src_ds, xRes=zoom_in, yRes=zoom_in, format='MEM', resampleAlg=resample_alg, transformerOptions=['SRC_METHOD=NO_GEOTRANSFORM', 'DST_METHOD=NO_GEOTRANSFORM'])
        dst_geo_transform = np.array(src_raster.geo_transform)
        dst_geo_transform[[1,2,4,5]] *= zoom_in
        dst_raster = tgp.read_gdal_ds(dst_ds)
        dst_raster.geo_transform = tuple(dst_geo_transform)
        dst_raster.projection = src_raster.projection
    
    return dst_raster

def reproject(src_raster, dst_crs='EPSG:4326', src_crs=None):
    """Reproject the raster data.

    Parameters
    ----------
    src_raster: Raster 
        Which raster data to be refined.
    dst_crs: str, optional, default: EPSG:4326
        The target crs to transform the raster to.
    src_crs: str, optional
        The source crs to transform the raster from. If None, 
        get the projection from src_raster.

    Returns
    -------
    dst_raster: Raster
        Reprojected result.

    Examples
    -------- 
    >>> import TronGisPy as tgp
    >>> src_raster_fp = tgp.get_testing_fp()
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> print("project(before)", src_raster.projection)
    >>> dst_raster = ShapeGrid.reproject(src_raster, dst_crs='EPSG:4326', src_crs=None)
    >>> print("project(after)", tgp.wkt_to_epsg(dst_raster.projection))
    """
    src_ds = src_raster.to_gdal_ds()

    if src_crs:
        dst_ds = gdal.Warp('', src_ds, srcSRS=src_crs, dstSRS=dst_crs, format='MEM')
    else:
        dst_ds = gdal.Warp('', src_ds, dstSRS=dst_crs, format='MEM')
    dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster

def zonal_stats(src_poly, src_raster, operator=['mean']):
    """Calculate the statistic value for each zone defined by src_poly, base on values from src_raster. 

    Parameters
    ----------
    src_poly: Geopandas.GeoDataFrame
        The Zone dataset to calculated statistic values.
    src_raster: Raster
        Which value dataset to be calculated statistic values.
    operator: list of {'mean', 'max', 'min', 'median', 'std', 'sum', 'count'}, optional, defalut: ['mean']
        The statistic operator to be used.

    Returns
    -------
    vector: Geopandas.GeoDataFrame 
        Vectorize result.

    Examples
    -------- 
    >>> import geopandas as gpd
    >>> import TronGisPy as tgp
    >>> from TronGisPy import ShapeGrid
    >>> src_raster_fp = tgp.get_testing_fp('satellite_tif')
    >>> src_poly_fp = tgp.get_testing_fp('satellite_tif_clipper')
    >>> src_raster = tgp.read_raster(src_raster_fp)
    >>> src_poly = gpd.read_file(src_poly_fp)
    >>> df_shp = ShapeGrid.zonal_stats(src_poly, src_raster, operator=['mean'])
    >>> df_shp.head()
    """
    assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
    assert isinstance(operator, list), "operator should be a list of string. ex: ['mean']"
    df_shp = src_poly.copy()
    df_shp['poly_idx'] = list(range(len(df_shp)))
    df_shp['poly_idx'] = df_shp['poly_idx'].astype('float')
    poly_rst = tgp.ShapeGrid.rasterize_layer(df_shp, src_raster.rows, src_raster.cols, src_raster.geo_transform, 'poly_idx', all_touched=True, no_data_value=np.nan)
    X_combine = np.concatenate([poly_rst.data, src_raster.data], axis=-1)
    X_combine_df = pd.DataFrame(X_combine.reshape(-1, 2))
    X_groupby = X_combine_df.groupby(0, as_index=False)
    for op in operator:
        if op == 'mean':
            df_shp = df_shp.merge(X_groupby.mean().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        elif op == 'max':
            df_shp = df_shp.merge(X_groupby.max().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        elif op == 'min':
            df_shp = df_shp.merge(X_groupby.min().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        elif op == 'median':
            df_shp = df_shp.merge(X_groupby.median().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        elif op == 'sum':
            df_shp = df_shp.merge(X_groupby.sum().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        elif op == 'std':
            df_shp = df_shp.merge(X_groupby.std().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        elif op == 'count':
            df_shp = df_shp.merge(X_groupby.count().rename(columns={0:'poly_idx', 1:f'zonal_{op}'}), on='poly_idx', how='left')
        else:
            assert False, "no this operator"
    return df_shp