import os
import numpy as np
import gdal
import geopandas as gpd
from shapely.geometry import Polygon
epsilon = 10**-6

class SplittedImage():
    def __init__(self, source_image, box_size, geo_transform, projection, padding='right'):
        """padding: ['right', 'left', 'center']"""
        self.source_image = source_image
        self.window_size_h = self.window_size_w = self.step_size_h = self.step_size_w = box_size
        # self.window_size_h = window_size_h
        # self.window_size_w = window_size_w
        # self.step_size_h = step_size_h
        # self.step_size_w = step_size_w
        self.geo_transform = geo_transform
        self.projection = projection
        self.padding = padding
        self.img_h, self.img_w = self.source_image.shape[:2]
        self.is_single_band = len(self.source_image.shape) == 2
        self.num_bands = 1 if self.is_single_band else self.source_image.shape[2]

        # how many img will be splitted in one row & column
        self.n_steps_h = int((self.img_h - self.window_size_h) / (self.step_size_h + epsilon)) + 1
        self.n_steps_w = int((self.img_w - self.window_size_w) / (self.step_size_w + epsilon)) + 1
 
        # resize image into fit for splitting size 
        self.img_h_resized = self.window_size_h + self.step_size_h * self.n_steps_h
        self.img_w_resized = self.window_size_w + self.step_size_w * self.n_steps_w

        self.shape = ((self.n_steps_h + 1), (self.n_steps_w + 1))
        self.padded_image = self.get_padded_image()
        self.n_splitted_images = (self.n_steps_h + 1) * (self.n_steps_w + 1)

    def __getitem__(self, slice_value):
        if type(slice_value) in [int, slice]:
            h_start_inner, h_stop_inner = self.get_inner_idx(slice_value)
            return self.source_image[h_start_inner:h_stop_inner, :]

        elif type(slice_value) == tuple:
            h_start_inner, h_stop_inner = self.get_inner_idx(slice_value[0])
            w_start_inner, w_stop_inner = self.get_inner_idx(slice_value[1])
            return self.source_image[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner]

    def get_inner_idx(self, slice_value):
        if type(slice_value) == int:
            h_start, h_stop = slice_value, slice_value
            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(h_start, h_stop)
        elif type(slice_value) == slice:
            h_start, h_stop = slice_value.start, slice_value.stop
            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(h_start, h_stop)
        return h_start_inner, h_stop_inner

    def get_padded_image(self):
        if self.padding == 'right':
            if self.is_single_band: # s`ingle band (gray scale)
                img_resized = np.pad(self.source_image, ((0, self.img_h_resized-self.img_h), (0, self.img_w_resized-self.img_w)), 'constant', constant_values=0)
            else: # multiple bands
                img_resized = np.pad(self.source_image, ((0, self.img_h_resized-self.img_h), (0, self.img_w_resized-self.img_w), (0,0)), 'constant', constant_values=0)
        return img_resized

    def convert_to_inner_index_h(self, h_start, h_stop):
        h_start = 0 if h_start == None else h_start
        h_stop = 0 if h_stop == None else h_stop
        h_start_inner = self.step_size_h * h_start
        h_stop_inner = self.step_size_h * h_stop + self.window_size_h
        return (h_start_inner, h_stop_inner)

    def convert_to_inner_index_w(self, w_start, w_stop):
        w_start = 0 if w_start == None else w_start
        w_stop = 0 if w_stop == None else w_stop
        w_start_inner = self.step_size_w * w_start
        w_stop_inner = self.step_size_w * w_stop + self.window_size_w
        return (w_start_inner, w_stop_inner)

    def convert_location_to_order_index(self, idx_h, idx_w):
        return (idx_h * self.n_steps_w) + idx_w

    def convert_order_to_location_index(self, order_index):
        idx_h = order_index // (self.n_steps_w + 1)
        idx_w = order_index % (self.n_steps_w + 1)
        return (idx_h, idx_w)

    def apply(self, apply_fun): # apply functino to all images:
        return_objs = []
        for i in range(self.n_splitted_images):
            idx_h , idx_w = self.convert_order_to_location_index(i)
            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
            w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
            splitted_img = self.padded_image[h_start_inner:h_stop_inner,w_start_inner:w_stop_inner].copy()
            return_objs.append(apply_fun(splitted_img))
        return return_objs

    def get_splitted_images(self):
        return np.array(self.apply(lambda x:x))

    def transfer_xy_to_coord(self, xy, geo_transform):
        xoffset, px_w, rot1, yoffset, rot2, px_h = geo_transform
        x, y = xy
        posX = px_w * x + rot1 * y + xoffset
        posY = rot2 * x + px_h * y + yoffset
        posX += px_w / 2.0
        posY += px_h / 2.0
        return (posX, posY)
        
    def get_geo_attribute(self, return_geo_transform=False):
        rows = []
        for i in range(self.n_splitted_images):
            idx_h , idx_w = self.convert_order_to_location_index(i)

            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
            w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
            x_start, y_start, x_stop, y_stop = w_start_inner, h_start_inner, w_stop_inner, h_stop_inner

            left_top_coord = self.transfer_xy_to_coord((x_start, y_start), self.geo_transform)
            left_buttom_coord = self.transfer_xy_to_coord((x_start, y_stop), self.geo_transform)
            right_buttom_coord = self.transfer_xy_to_coord((x_stop, y_stop), self.geo_transform)
            right_top_coord = self.transfer_xy_to_coord((x_stop, y_start), self.geo_transform)

            x_min, y_max = left_top_coord
            pixel_size = self.geo_transform[1]
            row = {
                "idx":i,
                "idx_h":idx_h,
                "idx_w":idx_w,
                "geo_transform":(x_min, pixel_size, 0, y_max, 0, -pixel_size),
                "geometry": Polygon([left_top_coord, 
                                    left_buttom_coord, 
                                    right_buttom_coord, 
                                    right_top_coord, 
                                    left_top_coord]),
            }
            rows.append(row)
        df_attribute = gpd.GeoDataFrame(rows, geometry='geometry')
        if return_geo_transform:
            return df_attribute[["idx", "idx_h", "idx_w", "geometry", "geo_transform"]]
        else:
            return df_attribute[["idx", "idx_h", "idx_w", "geometry"]]

    def write_output_tif(self, X, dst_tif_path, bands, cols, rows, geo_transform, projection):
        dst_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, cols, rows, bands, gdal.GDT_Int32) # dst_filename, xsize=512, ysize=512, bands=1, eType=gdal.GDT_Byte
        dst_ds.SetGeoTransform(geo_transform)
        dst_ds.SetProjection(projection)

        for b in range(bands):
            band = dst_ds.GetRasterBand(b+1)
            band.WriteArray(X[:, :, b], 0, 0)

        band.FlushCache()
        band.SetNoDataValue(-99)
        dst_ds = None

    def write_splitted_images(self, target_dir, filename):
        df_attribute = self.get_geo_attribute(return_geo_transform=True)
        splitted_images = self.get_splitted_images()
        for idx, row in df_attribute.iterrows():
            target_img = splitted_images[idx].copy()
            idx_str = "_" + ("%3i"%idx).replace(" ", "0")
            idx_h_str = "_" + ("%3i"%row['idx_h']).replace(" ", "0")
            idx_w_str = "_" + ("%3i"%row['idx_w']).replace(" ", "0")
            path = os.path.join(target_dir, filename + idx_str + idx_h_str + idx_w_str + ".tif")
            if target_img.std() != 0:
                self.write_output_tif(target_img, path, self.num_bands, self.window_size_w, self.window_size_h, row['geo_transform'], self.projection)

    def write_combined_tif(self, X, dst_tif_path, dtype_gdal=gdal.GDT_Int32):
        rows = self.source_image.shape[0]
        cols = self.source_image.shape[1]
        bands = X.shape[3]
        dst_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, cols, rows, bands, dtype_gdal)
        dst_ds.SetGeoTransform(self.geo_transform)
        dst_ds.SetProjection(self.projection)
        
        for b in range(bands):
            X_combined = np.zeros((rows, cols))
            for i in range(len(X)):
                idx_h , idx_w = self.convert_order_to_location_index(i)
                h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
                w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
                h_length, w_length = X_combined[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner].shape
                X_combined[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner] = X[i, :h_length, :w_length, b]
                
            band = dst_ds.GetRasterBand(b+1)
            band.WriteArray(X_combined, 0, 0)
            band.FlushCache()
            band.SetNoDataValue(-99)

        dst_ds = None
