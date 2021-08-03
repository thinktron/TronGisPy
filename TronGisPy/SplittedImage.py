import os
import numpy as np
from osgeo import gdal
import geopandas as gpd
import TronGisPy as tgp
from shapely.geometry import Polygon
epsilon = 10**-6

class SplittedImage():
    """SplittedImage helps to splitting big remote sensing images into tiny 
    pieces for AI training purpose. SplittedImage supports not only images 
    splitting, but also combination of predicted results on the splitted images.

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> raster = tgp.read_raster(tgp.get_testing_fp())
    >>> box_size, step_size = 254, 127
    >>> splitted_image = tgp.SplittedImage(raster, box_size, step_size=step_size)
    >>> splitted_image
    window_size: (254, 254)
    step_size: (127, 127)
    pad_val: 0
    src_raster:
        shape: (677, 674, 3)
        gdaldtype: GDT_Int16
        geo_transform: (271982.8783, 1.1186219584569888, 0.0, 2769973.0653, 0.0, -1.1186305760705852)
        projection: PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3826"]]
        no_data_value: -32768.0
        metadata: {'AREA_OR_POINT': 'Area'}
    """

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

    def __repr__(self):
        desc = ""
        desc += "window_size: ({window_size_h}, {window_size_w})\n".format(window_size_h=self.window_size_h, window_size_w=self.window_size_w)
        desc += "step_size: ({step_size_h}, {step_size_w})\n".format(step_size_h=self.step_size_h, step_size_w=self.step_size_w)
        desc += "pad_val: {pad_val}\n".format(pad_val=self.pad_val)
        desc += "src_raster: \n {src_raster}\n".format(src_raster="\n".join(["\t"+line for line in str(self.src_raster).split("\n")]))
        return desc

    @property
    def n_steps_h(self):
        """The number of the images will be splitted in one column"""
        return int((self.src_rows - self.window_size_h) / (self.step_size_h + epsilon)) + 2
        
    @property
    def n_steps_w(self):
        """The number of the images will be splitted in one row"""
        return int((self.src_cols - self.window_size_w) / (self.step_size_w + epsilon)) + 2
 
    @property
    def padded_rows(self):
        """The padded image rows. The original image is padded 
        in order to fit for splitting size."""
        return self.window_size_h + self.step_size_h * (self.n_steps_h - 1)

    @property
    def padded_cols(self):
        """The padded image columns. The original image is padded 
        in order to fit for splitting size."""
        return self.window_size_w + self.step_size_w * (self.n_steps_w - 1)

    @property
    def shape(self):
        """The number of splitted images by rows and columns."""
        return ((self.n_steps_h), (self.n_steps_w))

    @property
    def n_splitted_images(self):
        """The number of splitted images."""
        return (self.n_steps_h) * (self.n_steps_w)

    @property
    def padded_image(self):
        """The padded image. The original image is padded 
        in order to fit for splitting size."""
        padded_image = np.pad(self.src_image, ((0, self.padded_rows-self.src_rows), (0, self.padded_cols-self.src_cols), (0,0)), 'constant', constant_values=self.pad_val)
        return padded_image

    def __getitem__(self, slice_value):
        if type(slice_value) in [int, slice]:
            h_start, h_stop = self.__process_slice_value(slice_value)
            w_start, w_stop = 0, self.n_steps_w
        elif type(slice_value) == tuple:
            h_start, h_stop = self.__process_slice_value(slice_value[0])
            w_start, w_stop = self.__process_slice_value(slice_value[1])

        h_start_inner, h_stop_inner = self.__convert_to_inner_index_h(h_start, h_stop)
        w_start_inner, w_stop_inner = self.__convert_to_inner_index_w(w_start, w_stop)

        data = self.src_image[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner]
        gt = np.array(self.src_gt).copy()
        gt[[0, 3]] = tgp.npidxs_to_coords([(h_start_inner, w_start_inner)], self.src_gt)[0]
        raster = tgp.Raster(data, gt, self.proj, self.gdaldtype, self.no_data_value)
        return raster

    def __process_slice_value(self, slice_value):
        if slice_value is None:
            start, stop = None, None
        elif type(slice_value) == int:
            start, stop = slice_value, slice_value
        elif type(slice_value) == slice:
            start, stop = slice_value.start, slice_value.stop
        return start, stop

    def __convert_to_inner_index_h(self, h_start, h_stop):
        """convert order index of splitted images into source image index"""
        h_start = 0 if h_start == None else h_start
        h_stop = self.n_steps_h if h_stop == None else h_stop
        h_start_inner = self.step_size_h * h_start
        h_stop_inner = self.step_size_h * h_stop + self.window_size_h
        return (h_start_inner, h_stop_inner)

    def __convert_to_inner_index_w(self, w_start, w_stop):
        """convert order index of splitted images into source image index"""
        w_start = 0 if w_start == None else w_start
        w_stop = self.n_steps_w if w_stop == None else w_stop
        w_start_inner = self.step_size_w * w_start
        w_stop_inner = self.step_size_w * w_stop + self.window_size_w
        return (w_start_inner, w_stop_inner)

    def convert_location_to_order_index(self, idx_h, idx_w):
        """Convert location index of the splitted image to the order index. 
        The order index:

        | 0 1 2 |  
        | 3 4 5 |  
        | 6 7 8 |
    
        The location index:

        | (0, 0) (0, 1) (0, 2) |  
        | (1, 0) (1, 1) (1, 2) |  
        | (2, 0) (2, 1) (2, 2) |  

        Parameters
        ----------
        idx_h: int
            The row index of the splitted image.
        idx_w: int
            The column index of the splitted image.

        Returns
        -------
        order_index: int
            The order index of the aplitted index.
        """
        return (idx_h * self.n_steps_w) + idx_w

    def convert_order_to_location_index(self, order_index):
        """Convert the order index of the splitted image to location index.
        The order index:

        | 0 1 2 |  
        | 3 4 5 |  
        | 6 7 8 |
    
        The location index:

        | (0, 0) (0, 1) (0, 2) |  
        | (1, 0) (1, 1) (1, 2) |  
        | (2, 0) (2, 1) (2, 2) |  

        Parameters
        ----------
        order_index: int
            The order index of the aplitted index.

        Returns
        -------
        location_index: tuple of int
            (idx_h, idx_w)
        """
        idx_h = order_index // self.n_steps_w
        idx_w = order_index % self.n_steps_w
        return (idx_h, idx_w)

    def apply(self, apply_fun, return_raster=False, gdaldtype=None, no_data_value=None): # apply functino to all images:
        """Apply a function to all splitted images.

        Parameters
        ----------
        apply_fun: function
            The function used to apply to all splitted images.

        Returns
        -------
        return_objs: nparray
            splitted images that have applied the function.
        """
        return_objs = []
        padded_image = self.padded_image
        geo_transforms = self.get_geo_attribute(True)['geo_transform']
        for i in range(self.n_splitted_images):
            idx_h , idx_w = self.convert_order_to_location_index(i)
            h_start_inner, h_stop_inner = self.__convert_to_inner_index_h(idx_h, idx_h)
            w_start_inner, w_stop_inner = self.__convert_to_inner_index_w(idx_w, idx_w)
            splitted_img = padded_image[h_start_inner:h_stop_inner,w_start_inner:w_stop_inner].copy()
            data = apply_fun(splitted_img)
            if return_raster:
                gt = geo_transforms[i]
                pj = self.proj
                gdaldtype = self.gdaldtype if gdaldtype is None else gdaldtype
                no_data_value = self.no_data_value if no_data_value is None else no_data_value
                raster = tgp.Raster(data, gt, pj, gdaldtype, no_data_value)
                return_objs.append(raster)
            else:
                return_objs.append(data)
        return return_objs

    def get_splitted_images(self, return_raster=False):
        """Get all splitted images.

        Returns
        -------
        splitted_images: ndarray
            All splitted images.
        """
        return np.array(self.apply(lambda x:x, return_raster=return_raster))
        
    def get_geo_attribute(self, return_geo_transform=False, crs=None):
        """Get geo_attributes (idx, idx_h, idx_w, geo_transform, geometry) 
        of all splitted images.

        Parameters
        ----------
        return_geo_transform: bool, optional, default: False
            Return gdal geo_transform for each geometry in the output GeoDataFrame.
        crs: str, optional
            The crs for the output GeoDataFrame e.g. 'epsg:4326'.

        Returns
        -------
        df_attribute: gpd.GeoDataFrame
            The geo_attributes of all splitted images, which can be output as a shapefile.
        """
        rows = []
        for i in range(self.n_splitted_images):
            idx_h , idx_w = self.convert_order_to_location_index(i)

            h_start_inner, h_stop_inner = self.__convert_to_inner_index_h(idx_h, idx_h)
            w_start_inner, w_stop_inner = self.__convert_to_inner_index_w(idx_w, idx_w)

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
        elif self.proj is not None: 
            try:
                df_attribute.crs = 'init:' + str(tgp.wkt_to_epsg(self.proj))
            except:
                df_attribute.crs = self.proj

        if not return_geo_transform:
            df_attribute.drop('geo_transform', axis=1, inplace=True)

        return df_attribute


    def write_splitted_images(self, target_dir, filename, filter_fun=lambda x:True, idxs_to_be_kept=None):
        """Write all splitted images as tif file.

        Parameters
        ----------
        target_dir: str
            The directory to save the splitted images.
        filename: str
            The prefix od the filename. The index number will be followed by 
            the output filename you defined, e.g. <filename>_idx_idxh_idxw;.
        filter_fun: function, optional, default: lambda x:True
            Filter specific spllitted images and not save it. The input of the function is 
            a splitted image. If the function output is True, the splitted image will be 
            saved.
        idxs_to_be_kept: list, optional
            list of indexs of splitted image to save as file.
        """
        df_attribute = self.get_geo_attribute(return_geo_transform=True)
        splitted_images = self.get_splitted_images()
        idxs_to_be_kept = range(len(df_attribute)) if idxs_to_be_kept is None else idxs_to_be_kept
        for idx, row in df_attribute.iterrows():
            if idx in idxs_to_be_kept:
                target_img = splitted_images[idx].copy()
                idx_str = "_" + ("%3i"%idx).replace(" ", "0")
                idx_h_str = "_" + ("%3i"%row['idx_h']).replace(" ", "0")
                idx_w_str = "_" + ("%3i"%row['idx_w']).replace(" ", "0")
                path = os.path.join(target_dir, filename + idx_str + idx_h_str + idx_w_str + ".tif")
                gt = row['geo_transform']
                if filter_fun(target_img):
                    tgp.write_raster(path, target_img, gt, self.proj, self.gdaldtype, self.no_data_value)

    def get_combined_image(self, X, padding=3, aggregator='mean'):
        """Combine the model predict result on all splitted images

        Parameters
        ----------
        X: array_like
            The splitted image prediction result. should have the same shape with the 
            SplittedImage.get_splitted_images.
        padding: int
            The number of pixel to remove the edge of each splitted image. Since 
            the segmentation model may not perform well on the edge of the image, 
            it can be resolve to pad each image when combining it.
        aggregator: {"mean", "median", "max", "min"}, optional, default: mean
            The operator perform on the image pixel with multiple results. If 
            multuple predicting results are overlapped in the combined image, 
            aggregator is necessary to combine them into one band.

        Returns
        -------
        X_combined: ndarray
            The combined image. The image will have the same geo-attribute (geo_transform 
            and projection) with the original image.
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
                h_start_inner, h_stop_inner = self.__convert_to_inner_index_h(idx_h, idx_h)
                w_start_inner, w_stop_inner = self.__convert_to_inner_index_w(idx_w, idx_w)
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
        """Combine the model predict result on splitted images and write as tif file.

        Parameters
        ----------
        X: array_like
            The splitted image prediction result. should have the same shape with the 
            SplittedImage.get_splitted_images.
        dst_tif_path: str
            The location to save the tif file.
        gdaldtype: int, optional
            The type of the cell defined in gdal which will affect the information 
            to be stored when saving the file. This can be generate from `gdal.GDT_XXX` 
            such as `gdal.GDT_Int32` equals 5 and `gdal.GDT_Float32` equals 6.
        no_data_value: int or float, optional
            Define which value to replace nan in numpy array when saving a raster file.
            
        """
        X_combined_bands = self.get_combined_image(X)
        gdaldtype = gdaldtype if gdaldtype is not None else self.gdaldtype
        no_data_value = no_data_value if no_data_value is not None else self.no_data_value
        tgp.write_raster(dst_tif_path, X_combined_bands, geo_transform=self.src_gt, projection=self.proj, gdaldtype=gdaldtype, no_data_value=no_data_value)
