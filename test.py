import unittest

# basic
import os
import shutil
import numpy as np

# data
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# gis
import geopandas as gpd
from shapely.geometry import Polygon
import gdal

# main
from PySatellite.SplittedImage import SplittedImage
from PySatellite.SatelliteIO import get_geo_info, get_nparray, get_extend, write_output_tif, clip_tif_by_shp, tif_composition, refine_resolution, rasterize_layer, polygonize_layer, raster_pixel_to_polygon, get_testing_fp
from PySatellite.Algorithm import kmeans
from PySatellite.CRS import transfer_npidx_to_coord, transfer_npidx_to_coord_polygon
# from PySatellite.Interpolation import inverse_distance_weighted

data_dir = os.path.join('PySatellite', 'data')
satellite_tif_path = os.path.join(data_dir, 'satellite_tif', 'satellite_tif.tif')
satellite_tif_clipper_path = os.path.join(data_dir, 'satellite_tif_clipper', 'satellite_tif_clipper.shp')
satellite_tif_kmeans_path = os.path.join(data_dir, 'satellite_tif_kmeans', 'satellite_tif_kmeans.tif')
rasterized_image_path = os.path.join(data_dir, 'rasterized_image', 'rasterized_image.tif')
# interpolation_points_path = os.path.join(data_dir, 'interpolation', 'climate_points.shp')

# show_image = True
show_image = False


class TestSplittedImage(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # window_size_h = window_size_w = step_size_h = step_size_w = 256
        self.box_size = 128
        
        cols, rows, bands, geo_transform, projection, dtype_gdal, no_data_value = get_geo_info(satellite_tif_path)
        self.geo_transform = geo_transform
        self.projection = projection
        self.dtype_gdal = dtype_gdal
        self.X = get_nparray(satellite_tif_path)

        self.splitted_image = SplittedImage(self.X, self.box_size, self.geo_transform, self.projection)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test___getitem__(self):
        slice_test1 = Counter(pd.cut(self.splitted_image[1].flatten(), bins=3, labels=range(3))) == Counter({1: 137343, 0: 122742, 2: 2059})
        slice_test2 = Counter(pd.cut(self.splitted_image[:2].flatten(), bins=3, labels=range(3))) == Counter({1: 412579, 0: 366496, 2: 7357})
        slice_test3 = Counter(pd.cut(self.splitted_image[:2, 2].flatten(), bins=3, labels=range(3))) == Counter({0: 97945, 1: 95721, 2: 2942})
        slice_test4 = Counter(pd.cut(self.splitted_image[:2, :2].flatten(), bins=3, labels=range(3))) == Counter({1: 333569, 0: 250291, 2: 5964})

        self.assertTrue(slice_test1)
        self.assertTrue(slice_test2)
        self.assertTrue(slice_test3)
        self.assertTrue(slice_test4)

    def test_get_padded_image(self):
        shape_test = self.splitted_image.padded_image.shape == (512, 512, 4)
        self.assertTrue(shape_test)

    def test_get_splitted_images(self):
        shape_test = self.splitted_image.get_splitted_images().shape == (16, 128, 128, 4)
        self.assertTrue(shape_test)

    def test_get_geo_attribute(self):
        df_attribute = self.splitted_image.get_geo_attribute()
        df_attribute.to_file(os.path.join(self.output_dir, "df_attribute.shp"))
        pol = df_attribute.loc[0, 'geometry']
        area_test = pol.area == 1638400.0
        self.assertTrue(area_test)

    def test_write_splitted_images(self):
        self.splitted_image.write_splitted_images(self.output_dir, 'test_satellite')

    def test_write_combined_tif(self):
        box_size = 101
        splitted_image = SplittedImage(self.X, box_size, self.geo_transform, self.projection)
        X_pred = splitted_image.get_splitted_images()
        dst_tif_path = os.path.join(self.output_dir, "combined.tif")
        splitted_image.write_combined_tif(X_pred, dst_tif_path, self.dtype_gdal)

class TestCRS(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_transfer_coord_to_npidx(self):
        cols, rows, bands, geo_transform, projection, dtype_gdal, no_data_value = get_geo_info(satellite_tif_path)
        xy = [1,3]
        coord_xy = transfer_npidx_to_coord(xy, geo_transform)
        self.assertTrue(coord_xy == (328560.0, 2750780.0))

    def test_transfer_npidx_to_coord_polygon(self):
        cols, rows, bands, geo_transform, projection, dtype_gdal, no_data_value = get_geo_info(satellite_tif_path)
        npidx = [0,2]
        polygon = transfer_npidx_to_coord_polygon(npidx, geo_transform)
        # df_lands_boundry = gpd.GeoDataFrame([{'geometry':polygon}], geometry='geometry')
        # df_lands_boundry.crs = {'init' :'epsg:3826'}
        # dst_shp_path = os.path.join(self.output_dir, 'df_lands_boundry.shp')
        # df_lands_boundry.to_file(dst_shp_path)
        centroid = polygon.centroid.x, polygon.centroid.y
        self.assertTrue(centroid == (328555.0, 2750785.0))


class TestSatelliteIO(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    # def tearDown(self):
    #     shutil.rmtree(self.output_dir)

    def test_clip_tif_by_shp(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)
        clip_image_arr = get_nparray(dst_image_path)
        if show_image:
            plt.imshow(clip_image_arr)
            plt.title("TestSatelliteIO" + ": " + "test_clip_tif_by_shp")
            plt.show()
        self.assertTrue(clip_image_arr.shape == (138, 225, 4))

    def test_tif_composition(self):
        crs_tif_image = satellite_tif_path
        src_tif_paths = [satellite_tif_path, satellite_tif_kmeans_path]
        dst_tif_path = os.path.join(self.output_dir, 'composited_image.tif')
        tif_composition(crs_tif_image, src_tif_paths, dst_tif_path)

        composited_image_arr = get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(composited_image_arr[:, :, 4], cmap='gray')
            plt.title("TestSatelliteIO" + ": " + "test_tif_composition")
            plt.show()

        self.assertTrue(composited_image_arr.shape == (512, 512, 5))

    def test_refine_resolution(self):
        src_tif_path = satellite_tif_path
        dst_tif_path = os.path.join(self.output_dir, 'resolution_refined_image.tif')
        dst_resolution = 5
        refine_resolution(src_tif_path, dst_tif_path, dst_resolution)

        resolution_refined_image_arr = get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(resolution_refined_image_arr)
            plt.title("TestSatelliteIO" + ": " + "test_refine_resolution")
            plt.show()
        self.assertTrue(resolution_refined_image_arr.shape == (1024, 1024, 4))

    def test_write_output_tif(self):
        dst_image_path = os.path.join(self.output_dir, 'clipped_image.tif')
        clip_tif_by_shp(satellite_tif_path, satellite_tif_clipper_path, dst_image_path)

        cols, rows, bands, geo_transform, projection, dtype_gdal, no_data_value = get_geo_info(dst_image_path)
        clip_image_arr = get_nparray(dst_image_path)
        padded_image_arr = np.pad(clip_image_arr, ((0,62), (0,75), (0,0)), mode='constant', constant_values=0)
        dst_tif_path = os.path.join(self.output_dir, 'padded_image.tif')
        write_output_tif(padded_image_arr,dst_tif_path,4,300,200,geo_transform, projection)

        padded_image_arr = get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(padded_image_arr)
            plt.title("TestSatelliteIO" + ": " + "test_write_output_tif")
            plt.show()
        self.assertTrue(padded_image_arr.shape == (200, 300, 4))

    def test_rasterize_layer(self):
        src_shp_path = satellite_tif_clipper_path
        dst_tif_path = os.path.join(self.output_dir, 'rasterized_image.tif')
        ref_tif_path = satellite_tif_path
        rasterize_layer(src_shp_path, dst_tif_path, ref_tif_path)

        rasterized_image = get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(rasterized_image[:,:,0], cmap='gray')
            plt.title("TestSatelliteIO" + ": " + "test_rasterize_layer")
            plt.show()
        
        self.assertTrue(np.sum(rasterized_image==1) == 20512)

    def test_polygonize_layer(self):
        src_tif_path = rasterized_image_path
        dst_shp_path = os.path.join(self.output_dir, 'polygonized_layer.shp')
        polygonize_layer(src_tif_path, dst_shp_path)

        df_shp = gpd.read_file(dst_shp_path)
        if show_image:
            df_shp.plot()
            plt.show()

        self.assertTrue(df_shp.loc[0, 'geometry'].area == 2051200)

    def test_raster_pixel_to_polygon(self):
        src_tif_path = satellite_tif_path
        dst_shp_path = os.path.join(self.output_dir, 'raster_pixel_to_polygon.shp')
        raster_pixel_to_polygon(src_tif_path, dst_shp_path, all_bands_as_feature=True, crs={'init' :'epsg:3826'})

    def test_get_testing_fp(self):
        fn = 'satellite_tif'
        fp = get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\PySatellite\\PySatellite\\data\\satellite_tif\\satellite_tif.tif')
        
        fn = 'satellite_tif_clipper'
        fp = get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\PySatellite\\PySatellite\\data\\satellite_tif_clipper\\satellite_tif_clipper.shp')

        fn = 'satellite_tif_kmeans'
        fp = get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\PySatellite\\PySatellite\\data\\satellite_tif_kmeans\\satellite_tif_kmeans.tif')

        fn = 'rasterized_image'
        fp = get_testing_fp(fn)
        self.assertTrue(fp == 'C:\\Users\\Thinktron\\Projects\\PySatellite\\PySatellite\\data\\rasterized_image\\rasterized_image.tif')


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join('test_output')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # window_size_h = window_size_w = step_size_h = step_size_w = 256
        self.box_size = 128
        self.cols, self.rows, self.bands, self.geo_transform, self.projection, self.dtype_gdal, self.no_data_value = get_geo_info(satellite_tif_path)
        self.X = get_nparray(satellite_tif_path)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_kmeans(self):
        X_kmeans = kmeans(self.X, n_clusters=5, no_data_value=0)
        dst_tif_path = os.path.join(self.output_dir, "X_kmeans.tif")
        bands = 1
        write_output_tif(X_kmeans.reshape(*X_kmeans.shape, -1), dst_tif_path, bands, self.cols, self.rows, self.geo_transform, self.projection)

        kmeans_image_arr = get_nparray(dst_tif_path)
        if show_image:
            plt.imshow(kmeans_image_arr[:, :, 0], cmap='gray')
            plt.title("TestAlgorithm" + ": " + "test_kmeans")
            plt.show()
        self.assertTrue(Counter(list(np.hstack(kmeans_image_arr[:, :, 0])))[4] == 9511)

# class TestInterpolation(unittest.TestCase):
#     def setUp(self):
#         self.output_dir = os.path.join('test_output')
#         if not os.path.isdir(self.output_dir):
#             os.mkdir(self.output_dir)

#     # def tearDown(self):
#     #     shutil.rmtree(self.output_dir)

#     def test_inverse_distance_weighted(self):
#         POINTS = os.path.abspath(interpolation_points_path)
#         FIELD = "TEMP" # Field used for analysis
#         DOWNLOAD_DIR = self.output_dir
#         TARGET_TEMPLATE = None
#         CV_METHOD = 0
#         CV_SAMPLES = 10
#         TARGET_DEFINITION = 0
#         TARGET_USER_SIZE = 0.053 #Depends on the input shape
#         TARGET_USER_XMIN = 118.19 #Changes depends on the extent of shp
#         TARGET_USER_XMAX = 122.165 #Changes depends on the extent of shp
#         TARGET_USER_YMIN = 21.677 #Changes depends on the extent of shp
#         TARGET_USER_YMAX = 26.606 #Changes depends on the extent of shp
#         TARGET_USER_FITS = 0
#         SEARCH_RANGE = 1
#         SEARCH_RADIUS = 1000.0
#         SEARCH_POINTS_ALL = 1
#         SEARCH_POINTS_MIN = 0
#         SEARCH_POINTS_MAX = 20
#         SEARCH_DIRECTION = 0
#         DW_WEIGHTING = 1
#         DW_IDW_POWER = 2.0
#         DW_IDW_OFFSET = False
#         DW_BANDWIDTH = 1.0
#         out, err = inverse_distance_weighted(POINTS, FIELD, DOWNLOAD_DIR, TARGET_TEMPLATE, CV_METHOD, CV_SAMPLES, TARGET_DEFINITION, TARGET_USER_SIZE, TARGET_USER_XMIN, TARGET_USER_XMAX, TARGET_USER_YMIN, TARGET_USER_YMAX, TARGET_USER_FITS, SEARCH_RANGE, SEARCH_RADIUS, SEARCH_POINTS_ALL, SEARCH_POINTS_MIN, SEARCH_POINTS_MAX, SEARCH_DIRECTION, DW_WEIGHTING, DW_IDW_POWER, DW_IDW_OFFSET, DW_BANDWIDTH)
#         self.assertEqual(err.decode(), "")

if __name__ == "__main__":
    unittest.main()
#  python -m unittest -v test.py