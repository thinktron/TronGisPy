import os
import ogr
import gdal
import pyproj
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import TronGisPy as tgp

def rasterize_layer(src_vector, rows, cols, geo_transform, use_attribute, all_touched=False, no_data_value=0):
    """Rasterize vector data. Get the cell value in defined grid (rows, cols, geo_transform)
    from its overlapped polygon.

    Parameters
    ----------
    src_vector: `Geopandas.GeoDataFrame`. Which vector data to be rasterize.

    rows: int. Target rasterized image's rows.

    cols: int. Target rasterized image's cols.
    
    geo_transform: tuple or list. Target rasterized image's geo_transform which is 
    the affine parameter.

    use_attribute: str. The column to use as rasterized image value.

    all_touched: bool. Pixels that touch (not overlap over 50%) the polygon will be 
    assign the use_attribute value of the polygon.

    no_data_value: int or float. The pixels not covered by any polygon will be filled
    no_data_value.

    Returns
    -------
    raster: `TronGisPy.Raster`. Rasterized result.

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

def vectorize_layer(src_raster, band_num=1, field_name='value', multipolygon=False):
    """Vectorize raster data to achieve an acceptable raster-to-vector conversion.

    Parameters
    ----------
    src_raster: `TronGisPy.Raster`. Which raster data to be vectorize.
    
    band_num: int. Which band to be vectorized.

    field_name: str. Field to be generated in output vector data.

    multipolygon: bool. Combine the polygon with the same value to be a 
    `shapely.geometry.MultiPolygon`.

    Returns
    -------
    vector: `Geopandas.GeoDataFrame`. Vectorize result.

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

def clip_raster_with_polygon(src_raster, src_poly):
    """Clip raster with polygon.

    Parameters
    ----------
    src_raster: `TronGisPy.Raster`. Which raster data to be clipped.
    
    src_poly: `Geopandas.GeoDataFrame`. The clipper(clipping boundary).

    Returns
    -------
    dst_raster: `TronGisPy.Raster`. Clipped result.

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
    src_ds = src_raster.to_gdal_ds()
    temp_dir = tgp.create_temp_dir()
    src_shp_fp = os.path.join(temp_dir, 'src_poly.shp')
    src_poly.to_file(src_shp_fp)
    dst_ds = gdal.Warp('', src_ds, format= 'MEM', cutlineDSName=src_shp_fp, cropToCutline=True)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    tgp.remove_temp_dir()
    return dst_raster

def clip_raster_with_extent(src_raster, extent):
    """Clip raster with extent.

    Parameters
    ----------
    src_raster: `TronGisPy.Raster`. Which raster data to be clipped.
    
    extent: tuple or list. Output bounds as (xmin, ymin, xmax, ymax).

    Returns
    -------
    dst_raster: `TronGisPy.Raster`. Clipped result.

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
    """
    extent --- output bounds as (minX, minY, maxX, maxY) in target SRS
    """
    assert src_raster.geo_transform is not None, "src_raster.geo_transform should not be None"
    src_ds = src_raster.to_gdal_ds()
    dst_ds = gdal.Warp('', src_ds, format= 'MEM', outputBounds=extent, cropToCutline=True)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster

def refine_resolution(src_raster, dst_resolution, resample_alg='near', extent=None):
    """Clip raster with polygon.

    Parameters
    ----------
    src_raster: `TronGisPy.Raster` . Which raster data to be refined.
    
    dst_resolution: int. Target Resolution.

    resample_alg: str. Should be in {'near', 'bilinear', 'cubic', 
    'cubicspline', 'lanczos', 'average', 'mode'}.
        ``near``: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).
        ``bilinear``: bilinear resampling.
        ``cubic``: cubic resampling.
        ``cubicspline``: cubic spline resampling.
        ``lanczos``: Lanczos windowed sinc resampling.
        ``average``: average resampling, computes the weighted average of all non-NODATA contributing pixels.
        ``mode``: mode resampling, selects the value which appears most often of all the sampled points.

    extent: extent to clip the data.

    Returns
    -------
    dst_raster: `TronGisPy.Raster`. Refined result.

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
    dst_ds = gdal.Warp('', src_ds, xRes=dst_resolution, yRes=dst_resolution, outputBounds=extent, format='MEM', resampleAlg=resample_alg)
    dst_raster = tgp.read_gdal_ds(dst_ds)
    return dst_raster

def zonal_stats(src_poly, src_raster, operator=['mean']):
    """Calculate the statistic value for each zone defined by src_poly, base on values from src_raster. 

    Parameters
    ----------
    src_poly: `Geopandas.GeoDataFrame`. The Zone dataset to calculated statistic values.

    src_raster: `TronGisPy.Raster`. Which value dataset to be calculated statistic values.
    
    operator: list of str. Which statistic to be used. Including mean, max, min, median, std, sum, count...

    Returns
    -------
    vector: `Geopandas.GeoDataFrame`. Vectorize result.

    Examples
    -------- 
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