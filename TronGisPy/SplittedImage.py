import os
import gdal
import numpy as np
import geopandas as gpd
import TronGisPy as tgp
from shapely.geometry import Polygon
epsilon = 10**-6

class SplittedImage():
    def __init__(self, src_raster, box_size, step_size=None, pad_val=0):
        """padding: ['right', 'left']"""
        self.src_raster = src_raster
        self.src_image = src_raster.data
        self.src_gt = src_raster.geo_transform
        self.src_rows, self.src_cols, self.src_bands = src_raster.shape
        self.proj = src_raster.projection
        self.gdaldtype = src_raster.gdaldtype
        self.no_data_value = src_raster.no_data_value

        self.window_size_h = self.window_size_w = box_size
        self.step_size_h = self.step_size_w = step_size if step_size is not None else box_size
        self.pad_val = pad_val

    @property
    def n_steps_h(self):
        """how many img will be splitted in one column"""
        return int((self.src_rows - self.window_size_h) / (self.step_size_h + epsilon)) + 1
        
    @property
    def n_steps_w(self):
        """how many img will be splitted in one row"""
        return int((self.src_cols - self.window_size_w) / (self.step_size_w + epsilon)) + 1
 
    @property
    def padded_rows(self):
        """resize image into fit for splitting size"""
        return self.window_size_h + self.step_size_h * self.n_steps_h

    @property
    def padded_cols(self):
        """resize image into fit for splitting size"""
        return self.window_size_w + self.step_size_w * self.n_steps_w

    @property
    def padded_shape(self):
        """number of splitted images by rows and cols"""
        return ((self.n_steps_h + 1), (self.n_steps_w + 1))

    @property
    def n_splitted_images(self):
        """number of splitted images"""
        return (self.n_steps_h + 1) * (self.n_steps_w + 1)

    @property
    def padded_image(self):
        """pad image for splitting"""
        padded_image = np.pad(self.src_image, ((0, self.padded_rows-self.src_rows), (0, self.padded_cols-self.src_cols), (0,0)), 'constant', constant_values=self.pad_val)
        return padded_image

    def __getitem__(self, slice_value):
        if type(slice_value) in [int, slice]:
            h_start_inner, h_stop_inner = self.get_inner_idx(slice_value)
            return self.src_image[h_start_inner:h_stop_inner, :]

        elif type(slice_value) == tuple:
            h_start_inner, h_stop_inner = self.get_inner_idx(slice_value[0])
            w_start_inner, w_stop_inner = self.get_inner_idx(slice_value[1])
            return self.src_image[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner]

    def get_inner_idx(self, slice_value):
        if type(slice_value) == int:
            h_start, h_stop = slice_value, slice_value
            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(h_start, h_stop)
        elif type(slice_value) == slice:
            h_start, h_stop = slice_value.start, slice_value.stop
            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(h_start, h_stop)
        return h_start_inner, h_stop_inner

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
        padded_image = self.padded_image
        for i in range(self.n_splitted_images):
            idx_h , idx_w = self.convert_order_to_location_index(i)
            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
            w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
            splitted_img = padded_image[h_start_inner:h_stop_inner,w_start_inner:w_stop_inner].copy()
            return_objs.append(apply_fun(splitted_img))
        return return_objs

    def get_splitted_images(self):
        return np.array(self.apply(lambda x:x))
        
    def get_geo_attribute(self, return_geo_transform=False, crs=None):
        """crs={'init' :'epsg:xxxx'} e.g.{'init' :'epsg:4326'} """
        rows = []
        for i in range(self.n_splitted_images):
            idx_h , idx_w = self.convert_order_to_location_index(i)

            h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
            w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
            w_start_inner, h_start_inner, w_stop_inner, h_stop_inner

            left_top_coord = tgp.npidxs_to_coords([(h_start_inner, w_start_inner)], self.src_gt)[0]
            left_buttom_coord = tgp.npidxs_to_coords([(h_start_inner, w_stop_inner)], self.src_gt)[0]
            right_buttom_coord = tgp.npidxs_to_coords([(h_stop_inner, w_stop_inner)], self.src_gt)[0]
            right_top_coord = tgp.npidxs_to_coords([(h_stop_inner, w_start_inner)], self.src_gt)[0]

            x_min, y_max = left_top_coord
            row = {
                "idx":i,
                "idx_h":idx_h,
                "idx_w":idx_w,
                "geo_transform":(x_min, self.src_gt[1], self.src_gt[2], y_max, self.src_gt[4], self.src_gt[5]),
                "geometry": Polygon([left_top_coord, 
                                    left_buttom_coord, 
                                    right_buttom_coord, 
                                    right_top_coord, 
                                    left_top_coord]),
            }
            rows.append(row)
        df_attribute = gpd.GeoDataFrame(rows, geometry='geometry')

        if crs is not None:
            df_attribute.crs = crs

        if not return_geo_transform:
            df_attribute.drop('geo_transform', axis=1, inplace=True)

        return df_attribute


    def write_splitted_images(self, target_dir, filename, filter_fun=lambda x:True):
        """
        target_dir: where you want to store all aplitted images; 
        filename: index number will be followed by the output filename you defined, e.g. <filename>_idx_idxh_idxw;
        filter_fun(x_splitted): if return True, the image will be stored.
        """
        df_attribute = self.get_geo_attribute(return_geo_transform=True)
        splitted_images = self.get_splitted_images()
        for idx, row in df_attribute.iterrows():
            target_img = splitted_images[idx].copy()
            idx_str = "_" + ("%3i"%idx).replace(" ", "0")
            idx_h_str = "_" + ("%3i"%row['idx_h']).replace(" ", "0")
            idx_w_str = "_" + ("%3i"%row['idx_w']).replace(" ", "0")
            path = os.path.join(target_dir, filename + idx_str + idx_h_str + idx_w_str + ".tif")
            gt = row['geo_transform']
            if filter_fun(target_img):
                tgp.write_raster(path, target_img, gt, self.proj, self.gdaldtype, self.no_data_value)

    def get_combined_image(self, X, padding=3, aggregator='mean'):
        """
        padding:segmentation may wrong result at the boundry for each splitted image, it can be resolve to pad each image when combining it.
        aggregator: if multuple predicting result is overlapped in the combined image, aggregator is necessary to combine them into one band. "mean", "median", "max" and "min" is available.
        """
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=3) 
        rows, cols = self.src_rows, self.src_cols
        bands = X.shape[3]
        
        X_combined_bands = np.zeros((rows, cols, bands))
        for b in range(bands):
            overlapped_count = np.max([
                int(self.window_size_w/(self.step_size_w + epsilon)) + 1, 
                int(self.window_size_h/(self.step_size_h + epsilon)) + 1
                ])

            X_combined_overlap = np.full((rows, cols, overlapped_count), np.nan)
            for i in range(len(X)):
                idx_h , idx_w = self.convert_order_to_location_index(i)
                h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
                w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
                h_start_inner, h_stop_inner = h_start_inner + padding, h_stop_inner - padding
                w_start_inner, w_stop_inner = w_start_inner + padding, w_stop_inner - padding

                h_length, w_length, b_length = X_combined_overlap[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner].shape
                X_combined_inner = X_combined_overlap[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner]
                next_idx = np.argmax(np.isnan(X_combined_inner), axis=2).flatten() # find nan location (where to fill in) for each cell
                img_idx_x, img_idx_y = np.where(np.ones_like(X_combined_inner[:,:,0]))
                X_combined_inner[img_idx_x, img_idx_y, next_idx] = X[i, padding:padding+h_length, padding:padding+w_length, b].flatten()

            X_combined = np.full((rows, cols), np.nan)
            row_idxs_not_na, col_idxs_not_na = np.where(~(np.sum(np.isnan(X_combined_overlap), axis=2) == overlapped_count)) # for filtering all nan cell
            if aggregator=='mean': 
                X_combined[row_idxs_not_na, col_idxs_not_na] = np.nanmean(X_combined_overlap[row_idxs_not_na, col_idxs_not_na], axis=1)
            elif aggregator=='median': 
                X_combined[row_idxs_not_na, col_idxs_not_na] = np.nanmedian(X_combined_overlap[row_idxs_not_na, col_idxs_not_na], axis=1)
            elif aggregator=='max': 
                X_combined[row_idxs_not_na, col_idxs_not_na] = np.nanmax(X_combined_overlap[row_idxs_not_na, col_idxs_not_na], axis=1)
            elif aggregator=='min': 
                X_combined[row_idxs_not_na, col_idxs_not_na] = np.nanmin(X_combined_overlap[row_idxs_not_na, col_idxs_not_na], axis=1)
            X_combined_bands[:, :, b] = X_combined
        return X_combined_bands

    def write_combined_tif(self, X, dst_tif_path, gdaldtype=None, no_data_value=None):
        X_combined_bands = self.get_combined_image(X)
        gdaldtype = gdaldtype if gdaldtype is not None else self.gdaldtype
        no_data_value = no_data_value if no_data_value is not None else self.no_data_value
        tgp.write_raster(dst_tif_path, X_combined_bands, geo_transform=self.src_gt, projection=self.proj, gdaldtype=gdaldtype, no_data_value=no_data_value)
