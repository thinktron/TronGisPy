import unittest

# basic
import os
import time
import shutil
import numpy as np

# data
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# gis
import gdal 
import pyproj
import geopandas as gpd # should be put before gdal: https://blog.csdn.net/u014656611/article/details/106450006
from shapely.geometry import Polygon, Point

# main
import TronGisPy as tgp
from TronGisPy import SplittedImage, Normalizer
from TronGisPy import GisIO, Algorithm, Interpolation, ShapeGrid, AeroTriangulation, DEMProcessor
os.environ['TGPDYPEWARNING'] = str(False)
# print(os.environ['TGPDYPEWARNING'])

satellite_tif_path = tgp.get_testing_fp('satellite_tif')
satellite_tif_clipper_path = tgp.get_testing_fp('satellite_tif_clipper')
satellite_tif_kmeans_path = tgp.get_testing_fp('satellite_tif_kmeans')
rasterized_image_path = tgp.get_testing_fp('rasterized_image')
rasterized_image_1_path = tgp.get_testing_fp('rasterized_image_1')
poly_to_be_clipped_path = tgp.get_testing_fp('poly_to_be_clipped')
point_to_be_clipped_path = tgp.get_testing_fp('point_to_be_clipped')
line_to_be_clipped_path = tgp.get_testing_fp('line_to_be_clipped')
multiline_to_be_clipped_path = tgp.get_testing_fp('multiline_to_be_clipped')
remap_rgb_clipper_path = tgp.get_testing_fp('remap_rgb_clipper_path')
remap_ndvi_path = tgp.get_testing_fp('remap_ndvi_path')
tif_forinterpolation_path = tgp.get_testing_fp('tif_forinterpolation')
aero_triangulation_PXYZs_path = tgp.get_testing_fp('aero_triangulation_PXYZs')
flipped_gt_path = tgp.get_testing_fp('flipped_gt')
shp_clipper_path = tgp.get_testing_fp('shp_clipper')
dem_process_path = tgp.get_testing_fp('dem_process_path')

# show_image = True
show_image = False

# operation on gis data
class Testio(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_get_geo_info(self):
        rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = tgp.get_raster_info(satellite_tif_path)
        self.assertTrue((rows, cols, bands) == (512, 512, 4))
        self.assertTrue(geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))
        self.assertTrue(no_data_value == -99.0)

        no_data_value = tgp.get_raster_info(satellite_tif_path, 'no_data_value')
        self.assertTrue(no_data_value == -99.0)

    def test_get_raster_data(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        GisIO.clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path) 
        clip_image_arr = tgp.get_raster_data(dst_image_path)
        no_data_value = tgp.get_raster_info(dst_image_path, 'no_data_value')
        self.assertTrue(np.sum(clip_image_arr == no_data_value) == 42144)

    def test_get_raster_extent(self):
        extent = tgp.get_raster_extent(satellite_tif_path, 'plot')
        self.assertTrue(extent == (328530.0, 333650.0, 2745670.0, 2750790.0))
        if show_image:
            fig, ax = plt.subplots(1, 1)
            poly_extent = [Polygon(tgp.get_raster_extent(satellite_tif_path, True))]
            df = gpd.GeoDataFrame(poly_extent, columns=['geometry'], geometry='geometry')
            tgp.read_raster(satellite_tif_path).plot(ax=ax)
            df.exterior.plot(ax=ax, linewidth=20)
            plt.show()

    def test_update_raster_info(self):
        dst_tif_path = os.path.join(self.output_dir, 'X_geo_info_updated.tif')
        shutil.copyfile(satellite_tif_path, dst_tif_path)
        rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = tgp.get_raster_info(dst_tif_path)
        projection = tgp.epsg_to_wkt(3826)
        geo_transform = list(geo_transform)
        geo_transform[0] += 10
        geo_transform[3] -= 10
        tgp.update_raster_info(dst_tif_path, projection=projection, geo_transform=geo_transform, metadata={"Test":"Test"})
        rows, cols, bands, geo_transform, projection, gdaldtype, no_data_value, metadata = tgp.get_raster_info(dst_tif_path)
        self.assertTrue(geo_transform == (328540.0, 10.0, 0.0, 2750780.0, 0.0, -10.0))
        self.assertTrue(tgp.wkt_to_epsg(projection) == 3826)
        self.assertTrue(metadata['Test'] == "Test")
        ## with warning
        # tgp.update_raster_info(dst_tif_path, projection=projection, geo_transform=geo_transform, no_data_value=99, metadata={"Test":"Test"})
        # self.assertTrue(no_data_value == 99)

    def test_read_raster(self):
        raster = tgp.read_raster(satellite_tif_path)
        self.assertTrue(raster.shape == (512, 512, 4))
        self.assertTrue(raster.geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))
        raster = tgp.read_raster(tif_forinterpolation_path, fill_na=True)
        self.assertTrue(np.sum(raster.data == raster.no_data_value) == 34915)

    def test_write_raster(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        GisIO.clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)
        geo_transform, projection = tgp.get_raster_info(dst_image_path, ['geo_transform', 'projection'])
        clip_image_arr = tgp.get_raster_data(dst_image_path)
        padded_image_arr = np.pad(clip_image_arr, ((0,62), (0,75), (0,0)), mode='constant', constant_values=0)
        dst_tif_path = os.path.join(self.output_dir, 'padded_image.tif')
        tgp.write_raster(dst_tif_path, padded_image_arr, geo_transform, projection)
        padded_image_arr = tgp.get_raster_data(dst_tif_path)
        if show_image:
            plt.imshow(tgp.Normalizer().fit_transform(padded_image_arr[:, :, :3]))
            plt.title("Testio" + ": " + "test_write_raster")
            plt.show()
        self.assertTrue(padded_image_arr.shape == (200, 300, 4))

        # test write output without projection & geotransform
        X = np.random.rand(10000).reshape(100,100)
        dst_tif_path = os.path.join(self.output_dir, "test_output.tif")
        tgp.write_raster(dst_tif_path, X, gdaldtype=gdal.GDT_Float32)
        test_output = tgp.get_raster_data(dst_tif_path)
        self.assertTrue(test_output.shape == (100, 100, 1))

    def test_read_gdal_ds(self):
        data = tgp.get_raster_data(satellite_tif_path)
        geo_transform, projection, gdaldtype, no_data_value = tgp.get_raster_info(satellite_tif_path, ["geo_transform", "projection", "gdaldtype", "no_data_value"])
        ds = tgp.write_gdal_ds(data, geo_transform=geo_transform, projection=projection, gdaldtype=gdaldtype, no_data_value=no_data_value)
        raster = tgp.read_gdal_ds(ds)
        self.assertTrue(raster.shape == (512, 512, 4))
        self.assertTrue(type(raster) == tgp.Raster)
        self.assertTrue(raster.geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))

    def test_write_gdal_ds(self):
        data = tgp.get_raster_data(satellite_tif_path)
        geo_transform, projection, gdaldtype, no_data_value = tgp.get_raster_info(satellite_tif_path, ["geo_transform", "projection", "gdaldtype", "no_data_value"])
        ds = tgp.write_gdal_ds(data, geo_transform=geo_transform, projection=projection, gdaldtype=gdaldtype, no_data_value=no_data_value)
        self.assertTrue(type(ds) == gdal.Dataset)
        self.assertTrue(ds.GetGeoTransform() == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))
        self.assertTrue(ds.GetRasterBand(1).DataType == 5)

    def test_get_testing_fp(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(base_dir, 'TronGisPy', 'data')

        fn = 'satellite_tif'
        fp = tgp.get_testing_fp(fn)
        self.assertTrue(fp == os.path.join(data_dir, 'satellite_tif', 'satellite_tif.tif'))
        
        fn = 'satellite_tif_clipper'
        fp = tgp.get_testing_fp(fn)
        self.assertTrue(fp == os.path.join(data_dir, 'satellite_tif_clipper', 'satellite_tif_clipper.shp'))

        fn = 'satellite_tif_kmeans'
        fp = tgp.get_testing_fp(fn)
        self.assertTrue(fp == os.path.join(data_dir, 'satellite_tif_kmeans', 'satellite_tif_kmeans.tif'))

        fn = 'rasterized_image'
        fp = tgp.get_testing_fp(fn)
        self.assertTrue(fp == os.path.join(data_dir, 'rasterized_image', 'rasterized_image.tif'))

class TestRaster(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.raster = tgp.read_raster(satellite_tif_path)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test___init__(self):
        raster = tgp.Raster(np.zeros((3,3)))
        self.assertTrue(raster.shape == (3, 3, 1))
        self.assertTrue(raster.geo_transform == (0, 1, 0, 0, 0, -1))
        self.assertTrue(raster.metadata is None)

    def test_get_properties(self):
        self.assertTrue(self.raster.rows == 512)
        self.assertTrue(self.raster.cols == 512)
        self.assertTrue(self.raster.bands == 4)
        self.assertTrue(self.raster.gdaldtype_name == 'GDT_Int32')
        self.assertTrue(self.raster.shape == (512, 512, 4))
        self.assertTrue(self.raster.data.shape == (512, 512, 4))
        self.assertTrue(self.raster.geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))
        self.assertTrue(self.raster.metadata == {'AREA_OR_POINT': 'Area'})
        self.assertTrue(tgp.gdaldtype_to_npdtype(self.raster.gdaldtype) == np.int32)

    def test_set_properties(self):
        self.raster.data = np.random.randint(0, 100, self.raster.shape, dtype=np.uint8)
        self.assertTrue(self.raster.data.dtype == np.uint8)
        self.raster.geo_transform = (0, 1, 0, 0, 0, -1)
        self.assertTrue(self.raster.geo_transform == (0, 1, 0, 0, 0, -1))
        self.raster.metadata = {"Test":"Test"}
        self.assertTrue(self.raster.metadata == {"Test":"Test"})

    def test_pixel_size(self):
        flipped_raster = tgp.read_raster(flipped_gt_path)
        self.assertTrue(flipped_raster.pixel_size == (0.4327509283098073, 0.4327509283098073))

    def test_extent(self):
        self.assertTrue(self.raster.extent.tolist() == [[328530.0, 2750790.0], [333650.0, 2750790.0], [333650.0, 2745670.0], [328530.0, 2745670.0]])
        self.assertTrue(self.raster.extent_for_plot == (328530.0, 333650.0, 2745670.0, 2750790.0))

    def test_astype(self):
        self.raster.astype(np.int16)
        self.assertTrue(self.raster.gdaldtype == gdal.GDT_Int16)
        self.assertTrue(self.raster.data.dtype == np.int16)

    def test_get_values_by_coords(self):
        xmin, xmax, ymin, ymax = self.raster.extent_for_plot
        xmax -= 1
        ymin += 1
        extent = np.array([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
        npidxs_row, npidxs_col = np.array([(0, 0), (0, -1), (-1, -1), (-1, 0)]).T
        calcul_values = self.raster.get_values_by_coords(extent)
        target_values = self.raster.data[npidxs_row, npidxs_col]
        self.assertTrue(np.sum(calcul_values == target_values) / np.product(target_values.shape) == 1)

    def test_update_gdaldtype_by_npdtype(self):
        self.raster.data = np.random.randint(0, 100, self.raster.shape, dtype=np.uint8)
        self.raster.update_gdaldtype_by_npdtype()
        self.assertTrue(self.raster.gdaldtype == tgp.npdtype_to_gdaldtype(np.uint8))
        
    def test_to_file(self):
        dst_raster_path = os.path.join(self.output_dir, 'clipped_image.tif')
        self.raster.to_file(dst_raster_path)
        rows, cols, bands, geo_transform = tgp.get_raster_info(dst_raster_path, ['rows', 'cols', 'bands', 'geo_transform'])
        self.assertTrue(rows == 512)
        self.assertTrue(cols == 512)
        self.assertTrue(bands == 4)
        self.assertTrue(geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))        

    def test_to_gdal_ds(self):
        ds = self.raster.to_gdal_ds()
        self.assertTrue(type(ds) == gdal.Dataset)
        self.assertTrue(ds.GetGeoTransform() == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))
        self.assertTrue(ds.GetRasterBand(1).DataType == 5)

    def test_fill_no_data(self):
        raster1 = tgp.read_raster(tif_forinterpolation_path)
        raster1.fill_no_data(mode='constant', constant=0, fill_na=True)
        self.assertTrue(np.sum(raster1.data==0) == 34915)

        raster2 = tgp.read_raster(tif_forinterpolation_path)
        raster2.fill_no_data(mode='neighbor_mean', window_size=3, loop_to_fill_all=True, loop_limit=5, fill_na=True)
        self.assertTrue(np.mean(np.unique(raster2.data)).astype(np.int) == 63)
        self.assertTrue(np.std(np.unique(raster2.data)).astype(np.int) == 12)

        raster3 = tgp.read_raster(tif_forinterpolation_path)
        raster3.fill_no_data(mode='constant', constant=raster3.no_data_value, fill_na=True)
        raster3.astype(np.int)
        raster3.fill_no_data(mode='neighbor_majority', window_size=3, loop_to_fill_all=True, loop_limit=5, fill_na=True)
        self.assertTrue(np.sum(np.isin(np.unique(raster3.data), np.array([ 47, 48, 69, 75, 82]))) == 5)

        if show_image:
            fig , (ax1, ax2, ax3) = plt.subplots(1, 3)
            raster1.plot(ax=ax1)
            raster2.plot(ax=ax2)
            raster3.plot(ax=ax3)
            plt.show()

    def test_copy(self):
        raster_copy = self.raster.copy()
        raster_copy.geo_transform = [0, 1, 0, 0, 0, -1]
        self.assertTrue(self.raster.geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))

    def test_refine_resolution(self):
        src_raster = tgp.read_raster(satellite_tif_path)
        dst_raster = src_raster.refine_resolution(dst_resolution=5, resample_alg='bilinear', rotate=True)
        if show_image:
            dst_raster.plot(title="TestRaster" + ": " + "test_refine_resolution")
        self.assertTrue(dst_raster.shape == (1024, 1024, 4))
        self.assertTrue(dst_raster.geo_transform == (328530.0, 5.0, 0.0, 2750790.0, 0.0, -5.0))

        # src_raster = tgp.read_raster(tgp.get_testing_fp('rotate_tif'))
        src_raster = tgp.Raster(
                        data=np.random.randint(0, 255, (100, 100)), 
                        geo_transform=(286646.9886015, -0.031985, -0.57805, 2694943.1946525, -0.57805, 0.031985),
                        projection=tgp.epsg_to_wkt(3826),
                        gdaldtype=gdal.GDT_Int16)
        dst_raster1 = src_raster.refine_resolution(dst_resolution=2.315736920204883, resample_alg='cubic', rotate=True)
        dst_raster2 = src_raster.refine_resolution(dst_resolution=2.315736920204883, resample_alg='cubic', rotate=False)
        self.assertTrue(np.sum(dst_raster1.data == 0) == 51)
        self.assertTrue(np.sum(dst_raster2.data == 0) == 0)
        if show_image:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle("TestRaster" + ": " + "test_refine_resolution")
            ax1.hist(src_raster.data.flatten())
            ax2.hist(dst_raster1.data.flatten())
            ax3.hist(dst_raster2.data.flatten())
            plt.show()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle("TestShapeGrid" + ": " + "test_refine_resolution")
            src_raster.plot(ax=ax1)
            dst_raster1.plot(ax=ax2)
            dst_raster2.plot(ax=ax3)
            plt.show()

    def test_reproject(self):
        src_raster = tgp.read_raster(satellite_tif_path)
        dst_raster = src_raster.reproject(dst_crs='EPSG:4326', src_crs=None)
        self.assertTrue(tgp.wkt_to_epsg(dst_raster.projection) == 4326)


    def test_remap_by_ref_raster(self): 
        src_tif_path = tgp.get_testing_fp('remap_ndvi_path')
        ref_tif_path = tgp.get_testing_fp('remap_rgb_clipper_path')
        src_raster = tgp.read_raster(src_tif_path)
        ref_raster = tgp.read_raster(ref_tif_path)
        dst_raster = src_raster.remap_by_ref_raster(ref_raster)

        self.assertTrue(ref_raster.shape[:2], dst_raster.shape[:2])
        self.assertTrue(ref_raster.geo_transform, dst_raster.geo_transform)
        self.assertTrue(tgp.wkt_to_epsg(dst_raster.projection), 3826)
        if show_image:
            fig, (ax1 ,ax2) = plt.subplots(1, 2)
            fig.suptitle("TestRaster" + ": " + "test_remap_by_ref_raster")
            src_raster.plot(title="source", ax=ax1)
            dst_raster.plot(title="remapped", ax=ax2, cmap='gray')
            plt.show()

    def test_plot(self):
        flipped_raster = tgp.read_raster(flipped_gt_path)
        if show_image:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
            flipped_raster.plot(ax=ax1, flush_cache=True)
            ax1.set_title('original')
            flipped_raster.plot(ax=ax2, flush_cache=True, clip_percentage=(0.1, 0.9))
            ax2.set_title('clip_percentage')
            flipped_raster.plot(ax=ax3, flush_cache=True, log=True)
            ax3.set_title('log')
            flipped_raster.plot(ax=ax4, flush_cache=True, rescale_percentage=0.5)
            ax4.set_title('rescale')
            plt.show()

    def test_plot1(self):
        ras_norm = tgp.read_raster(tgp.get_testing_fp('norm'))
        if show_image:
            ras_norm.plot(clip_percentage=(0.1, 0.9))
            plt.show()

    def test_hist(self):
        flipped_raster = tgp.read_raster(flipped_gt_path)
        if show_image:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
            flipped_raster.hist(ax=ax1)
            ax1.set_title('original')
            flipped_raster.hist(ax=ax2, clip_percentage=(0.1, 0.9))
            ax2.set_title('clip_percentage')
            flipped_raster.hist(ax=ax3, log=True)
            ax3.set_title('log')
            flipped_raster.hist(ax=ax4, bands=[1,2])
            ax4.set_title('bands')
            plt.show()


class TestShapeGrid(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_get_rasterize_layer_params(self):
        src_shp = gpd.read_file(satellite_tif_clipper_path)
        src_shp['FEATURE'] = 1
        rows, cols ,geo_transform = ShapeGrid.get_rasterize_layer_params(src_shp, res=2)
        self.assertTrue(rows == 690)
        self.assertTrue(cols == 1130)
        self.assertTrue(geo_transform == (329405.4857111082, 2, 0, 2748181.2743610824, 0, -2))
        dst_raster = ShapeGrid.rasterize_layer(src_shp, rows, cols, geo_transform, use_attribute='FEATURE', no_data_value=99)
        self.assertTrue(np.sum(dst_raster.data==1) == 512776)
        if show_image:
            dst_raster.plot(title="TestShapeGrid" + ": " + "test_get_rasterize_layer_params", cmap='gray')

    def test_rasterize_layer(self):
        src_shp = gpd.read_file(satellite_tif_clipper_path)
        src_shp['FEATURE'] = 1
        rows, cols, geo_transform = tgp.get_raster_info(satellite_tif_path, ['rows', 'cols', 'geo_transform'])
        dst_raster = ShapeGrid.rasterize_layer(src_shp, rows, cols, geo_transform, use_attribute='FEATURE', no_data_value=99)
        self.assertTrue(np.sum(dst_raster.data==1) == 20512)
        if show_image:
            dst_raster.plot(title="TestShapeGrid" + ": " + "test_rasterize_layer", cmap='gray')

        dst_raster = ShapeGrid.rasterize_layer(src_shp, rows, cols, geo_transform, use_attribute='FEATURE', no_data_value=-99, all_touched=True)
        self.assertTrue(np.sum(dst_raster.data==1) == 20876)
        self.assertTrue(tgp.wkt_to_epsg(dst_raster.projection) == 3826)
        if show_image:
            dst_raster.plot(title="TestShapeGrid" + ": " + "test_rasterize_layer", cmap='gray')


    def test_rasterize_layer_by_ref_raster(self):
        src_shp = gpd.read_file(satellite_tif_clipper_path)
        src_shp['FEATURE'] = 1
        ref_raster = tgp.read_raster(satellite_tif_path)
        dst_raster = ShapeGrid.rasterize_layer_by_ref_raster(src_shp, ref_raster, use_attribute='FEATURE', no_data_value=99)
        self.assertTrue(np.sum(dst_raster.data==1) == 20512)
        if show_image:
            dst_raster.plot(title="TestShapeGrid" + ": " + "test_rasterize_layer", cmap='gray')

        dst_raster = ShapeGrid.rasterize_layer_by_ref_raster(src_shp, ref_raster, use_attribute='FEATURE', no_data_value=-99, all_touched=True)
        self.assertTrue(np.sum(dst_raster.data==1) == 20876)
        self.assertTrue(tgp.wkt_to_epsg(dst_raster.projection) == 3826)
        if show_image:
            dst_raster.plot(title="TestShapeGrid" + ": " + "test_rasterize_layer", cmap='gray')

    def test_vectorize_layer(self):
        src_raster = tgp.read_raster(rasterized_image_path)
        df_shp = ShapeGrid.vectorize_layer(src_raster)
        if show_image:
            df_shp[df_shp['value']==1].plot()
            plt.show()
        self.assertTrue(df_shp.loc[0, 'geometry'].area == 2051200)

        src_raster = tgp.read_raster(rasterized_image_1_path)
        df_shp = ShapeGrid.vectorize_layer(src_raster, multipolygon=True)
        if show_image:
            df_shp[df_shp['value']==1].plot()
            plt.show()
        self.assertTrue(df_shp['geometry'].area.values[0] == 3624400.0)

    def test_clip_raster_with_polygon(self):
        src_raster = tgp.read_raster(satellite_tif_path)
        src_shp = gpd.read_file(satellite_tif_clipper_path)
        dst_raster = ShapeGrid.clip_raster_with_polygon(src_raster, src_shp)
        if show_image:
            dst_raster.plot(title="TestShapeGrid" + ": " + "clip_raster_with_polygon")

        self.assertTrue(dst_raster.shape == (138, 226, 4))
        self.assertTrue(dst_raster.geo_transform == (329410.0, 10.0, 0.0, 2748180.0, 0.0, -10.0))
#         self.assertTrue(dst_raster.shape == (138, 225, 4))
#         self.assertTrue(dst_raster.geo_transform == (329460.0, 10.0, 0.0, 2748190.0, 0.0, -10.0))

    def test_clip_raster_with_multiple_polygons(self):
        multiple_poly_clipper_path = tgp.get_testing_fp('multiple_poly_clipper')
        multiple_poly_clip_ras_path = tgp.get_testing_fp('multiple_poly_clip_ras')

        src_raster = tgp.read_raster(multiple_poly_clip_ras_path)
        src_shp = gpd.read_file(multiple_poly_clipper_path)

        clipped_imgs = ShapeGrid.clip_raster_with_multiple_polygons(src_raster, src_shp, seed=2020)
        shapes = np.array([r.shape for r in clipped_imgs] )
        
        if show_image:
            fig, axes = plt.subplots(2, 5, figsize=(9, 6))
            axes = axes.flatten()
            for idx, ax in zip(np.arange(100, 100+10, 1), axes):
                ax.imshow(Normalizer().fit_transform(clipped_imgs[idx][:, :, :3]))
            fig.suptitle("TestShapeGrid" + ": " + "test_clip_raster_with_multiple_polygons")
            plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9 ,3))
            ax1.hist(shapes[:, 0])
            ax2.hist(shapes[:, 1])
            fig.suptitle("TestShapeGrid" + ": " + "test_clip_raster_with_multiple_polygons")
            plt.show()

        self.assertTrue(np.sum([c is None for c in clipped_imgs]) == 0)
        self.assertTrue(Counter(pd.cut(shapes[:, 0], bins=5, labels=False)) == {4: 167, 2: 54, 3: 46, 1: 45, 0: 13})
        self.assertTrue(Counter(pd.cut(shapes[:, 1], bins=5, labels=False)) == {2: 179, 3: 75, 1: 33, 0: 24, 4: 14})

        # try return raster
        clipped_imgs = ShapeGrid.clip_raster_with_multiple_polygons(src_raster, src_shp, return_raster=True)
        shapes = np.array([r.shape for r in clipped_imgs] )
        self.assertTrue(np.sum([c is None for c in clipped_imgs]) == 0)
        self.assertTrue(Counter(pd.cut(shapes[:, 0], bins=5, labels=False)) == {4: 167, 2: 54, 3: 46, 1: 45, 0: 13})
        self.assertTrue(Counter(pd.cut(shapes[:, 1], bins=5, labels=False)) == {2: 179, 3: 75, 1: 33, 0: 24, 4: 14})

        if show_image:
            fig, axes = plt.subplots(2, 5, figsize=(9, 6))
            axes = axes.flatten()
            for idx, ax in zip(np.arange(100, 100+10, 1), axes):
                clipped_imgs[idx].plot(ax=ax)
            fig.suptitle("TestShapeGrid" + ": " + "test_clip_raster_with_multiple_polygons")
            plt.show()

    def test_clip_raster_with_extent(self):
        src_raster = tgp.read_raster(satellite_tif_path)
        src_gdf = gpd.read_file(satellite_tif_clipper_path).to_crs(src_raster.projection)
        dst_raster = ShapeGrid.clip_raster_with_extent(src_raster, extent=src_gdf.total_bounds)
        if show_image:
            fig, (ax1 ,ax2) = plt.subplots(1, 2, figsize=(10, 5))
            src_raster.plot(ax=ax1)
            dst_raster.plot(ax=ax2)
            plt.title("TestShapeGrid" + ": " + "clip_raster_with_extent")
            plt.show()
        self.assertTrue(dst_raster.shape == (138, 226, 4))
        self.assertTrue(dst_raster.geo_transform == (329454.3927272463, 10.005213193877433, 0.0, 2748190.9018181805, 0.0, -10.010645586288655))
    
    def test_zonal_stats(self):
        src_raster = tgp.read_raster(satellite_tif_path)
        src_shp = gpd.read_file(satellite_tif_clipper_path)
        dst_shp = ShapeGrid.zonal_stats(src_shp, src_raster, operator=['mean', 'max', 'min', 'median', 'std', 'sum', 'count'])
        self.assertTrue(dst_shp.shape == (1, 10))
        self.assertTrue(dst_shp[f'zonal_mean'].values[0] == 57.394941075021556)
        self.assertTrue(dst_shp[f'zonal_max'].values[0] == 219.0)
        self.assertTrue(dst_shp[f'zonal_min'].values[0] == 16.0)
        self.assertTrue(dst_shp[f'zonal_median'].values[0] == 46.0)
        self.assertTrue(dst_shp[f'zonal_std'].values[0] == 33.44787374465527)
        self.assertTrue(dst_shp[f'zonal_sum'].values[0] == 599031.0	)
        self.assertTrue(dst_shp[f'zonal_count'].values[0] == 10437)

# gis tool
class TestCRS(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_npidxs_to_coord_polygons(self):
        geo_transform = tgp.get_raster_info(satellite_tif_path, 'geo_transform')
        npidxs = [[50, 50], [100, 100]]
        poly_points = tgp.npidxs_to_coord_polygons(npidxs, geo_transform)
        polygons = [Polygon(poly) for poly in poly_points]
        centroid = polygons[0].centroid.x, polygons[0].centroid.y
        self.assertTrue(centroid == (329035.0, 2750285.0))

        if show_image:
            df_poly = gpd.GeoDataFrame([Polygon(poly) for poly in poly_points], columns=['geom'], geometry='geom')
            fig ,ax = plt.subplots(1, 1)
            df_poly.plot(ax=ax)
            plt.title("Testcrs" + ": " + "test_npidxs_to_coords_polygon")
            plt.show()

    def test_wkt_to_epsg(self):
        wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        epsg = tgp.wkt_to_epsg(wkt)
        self.assertTrue(epsg == 4326)

        projection = tgp.get_raster_info(satellite_tif_path, 'projection')
        self.assertRaises(AssertionError, tgp.wkt_to_epsg, wkt=projection)

    def test_epsg_to_wkt(self):
        wkt = tgp.epsg_to_wkt(4326)
        epsg = tgp.wkt_to_epsg(wkt)
        self.assertTrue(epsg == 4326)

    def test_coords_to_npidxs(self):
        geo_transform = tgp.get_raster_info(satellite_tif_path, 'geo_transform')
        coords = np.array([(328560.0+9, 2750780.0-9)]) # resolution is 10 meter, add 9 will be in the same cell
        npidxs = tgp.coords_to_npidxs(coords, geo_transform)
        self.assertTrue(np.sum(npidxs == np.array([(1, 3)])) == 2)
        coords = np.array([(328560.0+11, 2750780.0-11)]) # resolution is 10 meter, add 11 will be in the next cell
        npidxs = tgp.coords_to_npidxs(coords, geo_transform)
        self.assertTrue(np.sum(npidxs == np.array([(2, 4)])) == 2)

    def test_npidxs_to_coords(self):
        geo_transform = tgp.get_raster_info(satellite_tif_path, 'geo_transform')
        npidxs = [(1,3)]
        coords = tgp.npidxs_to_coords(npidxs, geo_transform)
        self.assertTrue(np.sum(coords == np.array([(328560.0, 2750780.0)])) == 2)

    def test_get_extent(self):
        rows, cols, geo_transform = tgp.get_raster_info(remap_rgb_clipper_path, ['rows', 'cols', 'geo_transform'])
        extent = tgp.get_extent(rows, cols, geo_transform, return_type='plot')
        self.assertTrue(extent == (271982.8783, 272736.8295, 2769215.7524, 2769973.0653))

class TestTypeCast(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_get_gdaldtype_name(self):
        self.assertTrue(tgp.get_gdaldtype_name(5) == 'GDT_Int32')

    def test_gdaldtype_to_npdtype(self):
        self.assertTrue(tgp.gdaldtype_to_npdtype(5) == np.int32)

    def test_npdtype_to_gdaldtype(self):
        self.assertTrue(tgp.npdtype_to_gdaldtype(np.int32) == 5)
        self.assertTrue(tgp.npdtype_to_gdaldtype(np.bool) == gdal.GDT_Byte)


class TestAeroTriangulation(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test__convert_imxyzs_to_npidxs(self):
        npidxs = np.array([(0, 0), (2, 10), (3, 100)])
        imxys = AeroTriangulation._convert_npidxs_to_imxys(npidxs, rows=13824, cols=7680, pixel_size=0.012)
        npidxs_new = AeroTriangulation._convert_imxyzs_to_npidxs(imxys, rows=13824, cols=7680, pixel_size=0.012)
        self.assertTrue(np.sum((npidxs_new - npidxs) < 10**-4) == np.product(npidxs_new.shape))

    def test_project_npidxs_to_XYZs(self):
        aerotri_params = [np.array([0.00759914610989079, -0.002376824281950918, 1.561874186205409]),
                            np.array([ 249557.729, 2607778.809,    5826.51 ]),
                            13824, 7680, 120, 0.012]
        P_XYZs = np.load(aero_triangulation_PXYZs_path)
        P_npidxs = AeroTriangulation.project_XYZs_to_npidxs(P_XYZs, aerotri_params)
        P_XYZs_new, ks = AeroTriangulation.project_npidxs_to_XYZs(P_npidxs, P_XYZs[:, 2], aerotri_params, return_k=True)
        self.assertTrue(np.sum(P_XYZs_new - P_XYZs) == 0)

    def test_project_XYZs_to_npidxs(self):
        aerotri_params = [np.array([0.00759914610989079, -0.002376824281950918, 1.561874186205409]),
                            np.array([ 249557.729, 2607778.809,    5826.51 ]),
                            13824, 7680, 120, 0.012]
        gt_aerial = (247240.9472615, 0.0042, 0.332961, 2606525.0328175, 0.332961, -0.0042)
        P_XYZs = np.load(aero_triangulation_PXYZs_path)
        P_npidxs = AeroTriangulation.project_XYZs_to_npidxs(P_XYZs, aerotri_params)
        P_npidxs_coords = tgp.npidxs_to_coords(P_npidxs, gt_aerial)
        self.assertTrue(Polygon(P_npidxs_coords).area == 974.8584352214892)

    def test_get_img_pixels_XYZs(self):
        aerotri_params = [np.array([0.00759914610989079, -0.002376824281950918, 1.561874186205409]),
                            np.array([ 249557.729, 2607778.809,    5826.51 ]),
                            13824, 7680, 120, 0.012]
        P_XYZs = np.load(aero_triangulation_PXYZs_path)
        P_npidxs = AeroTriangulation.project_XYZs_to_npidxs(P_XYZs, aerotri_params)
        P_XYZs = AeroTriangulation.get_img_pixels_XYZs(P_npidxs, aerotri_params)
        self.assertTrue(Polygon(P_XYZs).area == 1.2660021631078202)

# CV
class TestNormalizer(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.rows, self.cols, self.bands, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata = tgp.get_raster_info(satellite_tif_path)
        self.X = tgp.get_raster_data(satellite_tif_path)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_Normalizer(self):
        X_norm = Normalizer().fit_transform(self.X) 
        self.assertTrue(np.sum(X_norm==1) == 289)
        self.assertTrue(np.sum(X_norm==0) == 1)

        X_norm = Normalizer().fit_transform(self.X, min_max_val=(20, 240)) 
        self.assertTrue(np.sum(X_norm==1) == 29)
        self.assertTrue(np.sum(X_norm==0) == 14074)

        X_norm = Normalizer().fit_transform(self.X, clip_percentage=(0.1, 0.9)) 
        self.assertTrue(np.sum(X_norm==0) == 112995)
        self.assertTrue(np.sum(X_norm==1) == 105926)

    def test_clip_by_min_max(self):
        X_norm = Normalizer().clip_by_min_max(self.X) 
        self.assertTrue(X_norm.min() == 10)
        self.assertTrue(X_norm.max() == 100)

    def test_fit_transform_opencv(self):
        X_norm = Normalizer().fit_transform_opencv(self.X) 
        self.assertTrue(np.sum(X_norm==1) == 0)
        self.assertTrue(np.sum(X_norm==0) == 0)

        X_norm = Normalizer().fit_transform_opencv(self.X, min_max_val=(20, 240)) 
        self.assertTrue(np.sum(X_norm==1) == 0)
        self.assertTrue(np.sum(X_norm==0) == 0)

        X_norm = Normalizer().fit_transform_opencv(self.X, clip_percentage=(0.1, 0.9)) 
        self.assertTrue(np.sum(X_norm==0) == 0)
        self.assertTrue(np.sum(X_norm==1) == 105926)

        np.random.seed(2020)
        X_test = self.X.copy().astype(np.float32)
        rand_xs = np.random.choice(range(X_test.shape[1]), 1000)
        rand_ys = np.random.choice(range(X_test.shape[0]), 1000)
        X_test[rand_xs, rand_ys] = np.nan
        X_norm = Normalizer().fit_transform_opencv(X_test, clip_percentage=(0.1, 0.9)) 
        self.assertTrue(np.sum(X_norm==0) == 0)
        self.assertTrue(np.sum(X_norm==1) == 105516)

class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.rows, self.cols, self.bands, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata = tgp.get_raster_info(satellite_tif_path)
        self.X = tgp.get_raster_data(satellite_tif_path)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_kmeans(self):
        X_kmeans = Algorithm.kmeans(self.X, n_clusters=5, no_data_value=0, random_state=2020)
        self.assertTrue(np.bincount(X_kmeans.astype(np.int32).flatten()).max() == 77721)
        if show_image:
            plt.imshow(X_kmeans, cmap='gray')
            plt.title("TestAlgorithm" + ": " + "test_kmeans")
            plt.show()

class TestInterpolation(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        X = tgp.get_raster_data(satellite_tif_path).astype(np.float)
        raw_shape = X.shape
        X = X.flatten()
        rand_idx = np.random.randint(0, len(X), int(len(X)*0.3))
        X[rand_idx] = np.nan
        self.X = X.reshape(raw_shape)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_img_interpolation(self):
        X_band0 = self.X[:, :, 0]
        X_band0_interp_nearest = Interpolation.img_interpolation(X_band0, method='nearest')[:, :, 0]
        X_band0_interp_linear = Interpolation.img_interpolation(X_band0, method='linear')[:, :, 0]
        X_band0_interp_cubic = Interpolation.img_interpolation(X_band0, method='cubic')[:, :, 0]
        if show_image:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle("TestInterpolation" + ": " + "test_img_interpolation")
            axes[0].imshow(X_band0, cmap='gray')
            axes[0].set_title('original')
            axes[1].imshow(X_band0_interp_nearest, cmap='gray')
            axes[1].set_title('nearest')
            axes[2].imshow(X_band0_interp_linear, cmap='gray')
            axes[2].set_title('linear')
            axes[3].imshow(X_band0_interp_cubic, cmap='gray')
            axes[3].set_title('cubic')
            plt.show()
            
        self.assertTrue(np.sum(np.isnan(X_band0_interp_nearest)) < np.product(X_band0_interp_nearest.shape) * 0.05)
        self.assertTrue(np.sum(np.isnan(X_band0_interp_linear)) < np.product(X_band0_interp_nearest.shape) * 0.05)
        self.assertTrue(np.sum(np.isnan(X_band0_interp_cubic)) < np.product(X_band0_interp_nearest.shape) * 0.05)
    
    def test_majority_interpolation(self):
        X = tgp.get_raster_data(tif_forinterpolation_path)[:, :, 0]
        nan_mask = np.isnan(X)
        X[nan_mask] = 999
        X_interp = Interpolation.majority_interpolation(X.astype(np.int), no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=1)
        self.assertTrue(np.mean(X[~nan_mask] == X_interp[~nan_mask]) == 1)
        self.assertTrue(np.sum(X_interp == 999) == 0)
        if show_image:
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))
            fig.suptitle("TestInterpolation" + ": " + "test_majority_interpolation")
            axes[0].imshow(X, cmap='gray')
            axes[0].set_title('original')
            axes[1].imshow(X_interp, cmap='gray')
            axes[1].set_title('interp')
            plt.show()

    def test_mean_interpolation(self):
        X = tgp.get_raster_data(tif_forinterpolation_path)[:, :, 0].astype(np.float)
        X[np.isnan(X)] = 999
        X_interp = Interpolation.mean_interpolation(X, no_data_value=999, window_size=3, loop_to_fill_all=True, loop_limit=1)
        self.assertTrue(np.sum(X_interp == 999) == 0)
        if show_image:
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))
            fig.suptitle("TestInterpolation" + ": " + "test_majority_interpolation")
            axes[0].imshow(X, cmap='gray')
            axes[0].set_title('original')
            axes[1].imshow(X_interp, cmap='gray')
            axes[1].set_title('interp')
            plt.show()

    def test_gdal_fillnodata(self):
        raster = tgp.read_raster(tif_forinterpolation_path)
        raster.data[np.isnan(raster.data)] = 999
        raster_interp = Interpolation.gdal_fillnodata(raster)
        self.assertTrue(np.sum(raster_interp.data == 999) == 0)
        if show_image:
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))
            fig.suptitle("TestInterpolation" + ": " + "test_gdal_fillnodata")
            raster.plot(ax=axes[0], cmap='gray')
            axes[0].set_title('original')
            raster_interp.plot(ax=axes[1], cmap='gray')
            axes[1].set_title('interp')
            plt.show()



class TestSplittedImage(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.raster = tgp.read_raster(satellite_tif_path)
        self.box_size, self.step_size = 254, 127
        self.splitted_image = SplittedImage(self.raster, self.box_size, step_size=self.step_size)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test___getitem__(self):
        self.assertTrue(Counter(pd.cut(self.splitted_image[1].data.flatten(), bins=3, labels=range(3))) == Counter({0: 268790, 1: 247772, 2: 3630}))
        self.assertTrue(Counter(pd.cut(self.splitted_image[:2].data.flatten(), bins=3, labels=range(3))) == Counter({1: 523687, 0: 508289, 2: 8408}))
        self.assertTrue(Counter(pd.cut(self.splitted_image[:2, 2].data.flatten(), bins=3, labels=range(3))) == Counter({0: 285697, 1: 225614, 2: 4817}))
        self.assertTrue(Counter(pd.cut(self.splitted_image[:2, :2].data.flatten(), bins=3, labels=range(3))) == Counter({1: 521447, 0: 502404, 2: 8405}))
        if show_image:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            self.splitted_image[0].plot(ax=ax1)
            self.splitted_image[1:, 1:].plot(ax=ax2)
            plt.show()

    def test_get_padded_image(self):
        shape_test = self.splitted_image.padded_image.shape == (635, 635, 4)
        self.assertTrue(shape_test)

    def test_apply(self):
        splitted_images = self.splitted_image.apply(lambda x:x*1.1, return_raster=True)
        if show_image:
            fig, axes = plt.subplots(*self.splitted_image.shape)
            axes = axes.flatten()
            for idx, ax in enumerate(axes):
                splitted_images[idx].plot(ax=ax)
            plt.show()

    def test_get_splitted_images(self):
        shape_test = self.splitted_image.get_splitted_images().shape == (16, 254, 254, 4)
        self.assertTrue(shape_test)

    def test_get_geo_attribute(self):
        df_attribute = self.splitted_image.get_geo_attribute()
        df_attribute.to_file(os.path.join(self.output_dir, "df_attribute.shp"))
        pol = df_attribute.loc[0, 'geometry']
        self.assertTrue(pol.area == 6451600.0)

    def test_write_splitted_images(self):
        self.splitted_image.write_splitted_images(self.output_dir, 'test_satellite')
        self.assertTrue(len(os.listdir(self.output_dir)) == 16)

    def test_get_combined_image(self):
        X_pred = self.splitted_image.get_splitted_images()
        X_combined = self.splitted_image.get_combined_image(X_pred, padding=3, aggregator='mean')
        self.assertTrue(np.nansum(X_combined - self.splitted_image.src_image) == 0)
        if show_image:
            plt.imshow(tgp.Normalizer().fit_transform(X_combined[:, :, :3]))
            plt.title("TestSplittedImage" + ": " + "test_get_combined_image")
            plt.show()

    def test_write_combined_tif(self):
        X_pred = self.splitted_image.get_splitted_images()
        dst_tif_path = os.path.join(self.output_dir, "combined.tif")
        self.splitted_image.write_combined_tif(X_pred, dst_tif_path)
        rows, cols, bands, geo_transform = tgp.get_raster_info(dst_tif_path, ["rows", "cols", "bands", "geo_transform"])
        self.assertTrue((rows, cols, bands) == (512, 512, 4))
        self.assertTrue(geo_transform == (328530.0, 10.0, 0.0, 2750790.0, 0.0, -10.0))

class TestGisIO(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        time.sleep(1)
        shutil.rmtree(self.output_dir)

    def test_clip_tif_by_shp(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        GisIO.clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)
        clip_image_raster = tgp.read_raster(dst_image_path)
        if show_image:
            clip_image_raster.plot(title="TestGisIO" + ": " + "test_clip_tif_by_shp")
        self.assertTrue(clip_image_raster.shape == (138, 225, 4))

    def test_clip_shp_by_shp(self):
        # polygon clipping
        src_shp_path = poly_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_poly.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_poly = gpd.read_file(dst_shp_path)
        self.assertTrue(gdf_clipped_poly['geometry'].apply(lambda x:x.area).sum() == 3)

        # points clipping
        src_shp_path = point_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_point.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_point = gpd.read_file(dst_shp_path)
        self.assertTrue(gdf_clipped_point['geometry'].iloc[0].coords[0] == (3,3))

        # line clipping
        src_shp_path = line_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_line.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        gdf_clipped_line = gpd.read_file(dst_shp_path)
        self.assertTrue(np.sum([line.length for line in gdf_clipped_line['geometry'] if line != None]) == 4)

        # multiline clipping
        src_shp_path = multiline_to_be_clipped_path
        clipper_shp_path = shp_clipper_path
        dst_shp_path = os.path.join(self.output_dir, 'clipped_multiline.shp')
        GisIO.clip_shp_by_shp(src_shp_path, clipper_shp_path, dst_shp_path)
        self.assertTrue(np.sum([line.length for line in gdf_clipped_line['geometry'] if line != None]) == 4)

    def test_tif_composition(self):
        crs_tif_image = satellite_tif_path
        src_tif_paths = [satellite_tif_path, satellite_tif_kmeans_path]
        dst_tif_path = os.path.join(self.output_dir, 'composited_image.tif')
        GisIO.tif_composition(crs_tif_image, src_tif_paths, dst_tif_path)

        composited_image_raster = tgp.read_raster(dst_tif_path)
        if show_image:
            composited_image_raster.plot(title="TestGisIO" + ": " + "test_tif_composition")
        self.assertTrue(composited_image_raster.shape == (512, 512, 5))

    def test_refine_resolution(self):
        src_tif_path = satellite_tif_path
        dst_tif_path = os.path.join(self.output_dir, 'resolution_refined_image.tif')
        GisIO.refine_resolution(src_tif_path, dst_tif_path, 5, 'bilinear')

        resolution_refined_image_raster = tgp.read_raster(dst_tif_path)
        self.assertTrue(resolution_refined_image_raster.shape == (1024, 1024, 4))
        if show_image:
            resolution_refined_image_raster.plot(title="TestGisIO" + ": " + "test_refine_resolution")

    def test_rasterize_layer(self):
        src_shp_path = satellite_tif_clipper_path
        dst_tif_path = os.path.join(self.output_dir, 'rasterized_image.tif')
        ref_tif_path = satellite_tif_path
        GisIO.rasterize_layer(src_shp_path, dst_tif_path, ref_tif_path)
        rasterized_image_raster = tgp.read_raster(dst_tif_path)
        if show_image:
            rasterized_image_raster.plot(title="TestGisIO" + ": " + "test_rasterize_layer")
        self.assertTrue(np.sum(rasterized_image_raster.data==1) == 20512)

        GisIO.rasterize_layer(src_shp_path, dst_tif_path, ref_tif_path, all_touched=True)
        rasterized_image_raster = tgp.read_raster(dst_tif_path)
        if show_image:
            rasterized_image_raster.plot(title="TestGisIO" + ": " + "test_rasterize_layer")
        self.assertTrue(np.sum(rasterized_image_raster.data==1) == 20876)

    def test_vectorize_layer(self):
        src_tif_path = rasterized_image_path
        dst_shp_path = os.path.join(self.output_dir, 'polygonized_layer.shp')
        GisIO.vectorize_layer(src_tif_path, dst_shp_path)
        df_shp = gpd.read_file(dst_shp_path)
        if show_image:
            df_shp.plot()
            plt.show()
        self.assertTrue(df_shp.loc[0, 'geometry'].area == 2051200)

        src_tif_path = rasterized_image_1_path
        dst_shp_path = os.path.join(self.output_dir, 'polygonized_layer.shp')
        GisIO.vectorize_layer(src_tif_path, dst_shp_path, multipolygon=True)
        df_shp = gpd.read_file(dst_shp_path)
        if show_image:
            df_shp.plot()
            plt.show()
        self.assertTrue(df_shp.loc[0, 'geometry'].area == 3624400.0)

    def test_raster_pixel_to_polygon(self):
        src_tif_path = satellite_tif_path
        dst_shp_path = os.path.join(self.output_dir, 'raster_pixel_to_polygon.shp')
        GisIO.raster_pixel_to_polygon(src_tif_path, dst_shp_path, all_bands_as_feature=True, crs='epsg:3826')

    def test_reproject(self):
        src_tif_path = satellite_tif_path
        dst_tif_path = os.path.join(self.output_dir, "X_reprojected.tif")
        GisIO.reproject(src_tif_path, dst_tif_path, dst_crs='EPSG:3826')
        self.assertTrue(tgp.wkt_to_epsg(tgp.get_raster_info(dst_tif_path, 'projection')) == 3826)

        geo_transform, gdaldtype = tgp.get_raster_info(dst_tif_path, ['geo_transform', 'gdaldtype'])
        tif_without_projection_path = os.path.join(self.output_dir, "X_without_projection.tif")
        tgp.write_raster(tif_without_projection_path, tgp.get_raster_data(dst_tif_path), geo_transform=geo_transform, gdaldtype=gdaldtype)

        src_tif_path = tif_without_projection_path
        dst_tif_path = os.path.join(self.output_dir, "X_reprojected_2.tif")
        GisIO.reproject(src_tif_path, dst_tif_path, dst_crs='EPSG:4326', src_crs='EPSG:3826')
        self.assertTrue(tgp.wkt_to_epsg(tgp.get_raster_info(dst_tif_path, 'projection')) == 4326)
    
    def test_remap_tif(self): 
        src_tif_path = tgp.get_testing_fp('remap_ndvi_path')
        ref_tif_path = tgp.get_testing_fp('remap_rgb_clipper_path')
        dst_tif_path = os.path.join(self.output_dir, 'X_remapped.tif')
        GisIO.remap_tif(src_tif_path, dst_tif_path, ref_tif_path)
        ref_rows, ref_cols, ref_bands, ref_geo_transform, ref_projection, ref_gdaldtype, ref_no_data_value, ref_metadata = tgp.get_raster_info(ref_tif_path)
        dst_rows, dst_cols, dst_bands, dst_geo_transform, dst_projection, dst_gdaldtype, dst_no_data_value, dst_metadata = tgp.get_raster_info(dst_tif_path)

        self.assertTrue((ref_cols, ref_rows) == (dst_cols, dst_rows))
        self.assertTrue(ref_geo_transform == dst_geo_transform)
        self.assertTrue(tgp.wkt_to_epsg(dst_projection) == 3826)
        source_image_raster = tgp.read_raster(src_tif_path)
        remapped_image_raster = tgp.read_raster(dst_tif_path)
        if show_image:
            fig, (ax1 ,ax2) = plt.subplots(1, 2)
            fig.suptitle("TestGisIO" + ": " + "test_remap_tif")
            source_image_raster.plot(title="source", ax=ax1)
            remapped_image_raster.plot(title="remapped", ax=ax2, cmap='gray')
            plt.show()

class TestDEMProcessor(unittest.TestCase):
    def setUp(self):
        time.sleep(1)
        self.dem_rows, self.dem_cols, self.dem_bands, self.dem_geo_transform, self.dem_projection, self.dem_gdaldtype, self.dem_no_data_value, self.dem_metadata = tgp.get_raster_info(dem_process_path)
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        time.sleep(1)

    def test_dem_to_hillshade(self):
        dem_process_raster = tgp.read_raster(dem_process_path)
        out_raster = DEMProcessor.dem_to_hillshade(dem_process_raster, azimuth=0, altitude=30)
        
        dst_array = out_raster.data
        dst_geo_transform = out_raster.geo_transform
        dst_rows, dst_cols, dst_bands = out_raster.shape
        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertGreaterEqual(np.nanmin(dst_array), 0)
        self.assertLessEqual(np.nanmax(dst_array), 255)

    def test_dem_to_slope(self):
        dem_process_raster = tgp.read_raster(dem_process_path)
        out_raster = DEMProcessor.dem_to_slope(dem_process_raster, slope_format='degree')

        dst_array = out_raster.data
        dst_geo_transform = out_raster.geo_transform
        dst_rows, dst_cols, dst_bands = out_raster.shape
        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertGreaterEqual(np.nanmin(dst_array[dst_array != -9999]), 0)
        self.assertLessEqual(np.nanmax(dst_array[dst_array != -9999]), 90)
    
    def test_dem_to_aspect(self):
        dem_process_raster = tgp.read_raster(dem_process_path)
        out_raster = DEMProcessor.dem_to_aspect(dem_process_raster)

        dst_array = out_raster.data
        dst_geo_transform = out_raster.geo_transform
        dst_rows, dst_cols, dst_bands = out_raster.shape
        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertGreaterEqual(np.nanmin(dst_array[dst_array != -9999]), 0)
        self.assertLessEqual(np.nanmax(dst_array[dst_array != -9999]), 360)

    def test_dem_to_TRI(self):
        dem_process_raster = tgp.read_raster(dem_process_path)
        out_raster = DEMProcessor.dem_to_TRI(dem_process_raster)
        
        dst_array = out_raster.data
        dst_geo_transform = out_raster.geo_transform
        dst_rows, dst_cols, dst_bands = out_raster.shape
        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        # self.assertTrue(self.dem_no_data_value == dst_no_data_value)
        self.assertGreaterEqual(np.nanmin(dst_array[dst_array != -9999]), 0)

    def test_dem_to_TPI(self):
        dem_process_raster = tgp.read_raster(dem_process_path)
        out_raster = DEMProcessor.dem_to_TPI(dem_process_raster)
        
        dst_array = out_raster.data
        dst_geo_transform = out_raster.geo_transform
        dst_rows, dst_cols, dst_bands = out_raster.shape
        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
    
    def test_dem_to_roughness(self):
        dem_process_raster = tgp.read_raster(dem_process_path)
        out_raster = DEMProcessor.dem_to_roughness(dem_process_raster)
        
        dst_array = out_raster.data
        dst_geo_transform = out_raster.geo_transform
        dst_rows, dst_cols, dst_bands = out_raster.shape
        self.assertTrue((self.dem_cols, self.dem_rows) == (dst_cols, dst_rows))
        self.assertTrue(self.dem_geo_transform == dst_geo_transform)
        self.assertGreaterEqual(np.nanmin(dst_array[dst_array != -9999]), 0)


if __name__ == "__main__":
    unittest.main()
#  python -m unittest -v test.py