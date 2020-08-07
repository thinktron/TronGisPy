import gdal
import numpy as np
import TronGisPy as tgp
from TronGisPy import Interpolation
from matplotlib import pyplot as plt

class Raster():
    """A Raster object contains all required information for a gis raster file such
    as `.tif` file including digital number for each pixel, number of rows,
    number of cols, number of bands, geo_transform, projection, no_data_value
    and metadata. 

    Parameters
    ----------
    data: `numpy.array`. Digital number for each raster cell. Data is in (n_rows, n_cols, n_bands) shape.

    geo_transform: tuple or list. Affine transform parameters (c, a, b, f, d, e
    = geo_transform). 

    projection: string. The well known text (WKT) of the raster which can be
    generate from `TronGisPy.epsg_to_wkt(<epsg_code>)`

    gdaldtype: int. The type of the cell defined in gdal which will affect the
    information to be stored when saving the file. This can be generate from
    `gdal.GDT_XXX` such as `gdal.GDT_Int32` equals 5 and `gdal.GDT_Float32`
    equals 6.

    no_data_value: int or float. Define which value to replace nan in numpy
    array when saving a raster file.

    metadata: dict. Define the metadata of the raster file.

    Attributes
    ----------
    rows: int. Number of rows.

    cols: int. Number of cols.

    bands: int. Number of bands.

    shape: tuple. The shape of the raster data.

    data: `numpy.array`. The digital number for each cell of the raster. Data is in (n_rows, n_cols, n_bands) shape.

    geo_transform: tuple. Affine transform parameters (c, a, b, f, d, e
    = geo_transform).

    metadata: dict. Metadata of the raster file.

    extent: `numpy.array`. Coordinates of Four corner points' of the raster. 

    extent_for_plot: tuple. (xmin, xmax, ymin, ymax) of the raster boundary.

    Examples
    --------
    >>> import TronGisPy as tgp
    >>> raster = tgp.read_raster(tgp.get_testing_fp())
    >>> raster.plot()
    >>> raster
    shape: (677, 674, 3)
    geo_transform: (271982.8783, 1.1186219584569888, 0.0, 2769973.0653, 0.0, -1.1186305760705852)
    projection: PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","1026"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","3824"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3826"]]
    no_data_value: -32768.0
    metadata: {'AREA_OR_POINT': 'Area'}
    >>> raster.data.shape
    (677, 674, 3)
    >>> raster.shape
    (677, 674, 3)
    >>> raster.extent
    array([[ 271982.8783, 2769973.0653],
           [ 272736.8295, 2769973.0653],
           [ 272736.8295, 2769215.7524],
           [ 271982.8783, 2769215.7524]])
    >>> raster.extent_for_plot
    (271982.8783, 272736.8295, 2769215.7524, 2769973.0653)
    """

    def __init__(self, data, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        self.data = data
        self.geo_transform = geo_transform if geo_transform is not None else [0, 1, 0, 0, 0, -1]
        self.gdaldtype = gdaldtype if gdaldtype is not None else tgp.npdtype_to_gdaldtype(data.dtype)
        self.projection = projection
        self.no_data_value = no_data_value
        self.metadata = metadata
        self.cache_data_for_plot = None

    def __repr__(self):
        desc = ""
        desc += "shape: ({rows}, {cols}, {bands})\n".format(rows=self.rows, cols=self.cols, bands=self.bands)
        desc += "gdaldtype: {gdaldtype}\n".format(gdaldtype=tgp.get_gdaldtype_name(self.gdaldtype))
        desc += "geo_transform: {geo_transform}\n".format(geo_transform=self.geo_transform)
        desc += "projection: {projection}\n".format(projection=self.projection)
        desc += "no_data_value: {no_data_value}\n".format(no_data_value=self.no_data_value)
        desc += "metadata: {metadata}".format(metadata=self.metadata)
        return desc

    @property
    def rows(self):
        return self.data.shape[0]
        
    @property
    def cols(self):
        return self.data.shape[1]

    @property
    def bands(self):
        return self.data.shape[2]
        
    @property
    def shape(self):
        return self.data.shape

    @property
    def gdaldtype_name(self):
        return tgp.get_gdaldtype_name(self.gdaldtype)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        assert type(data) == np.ndarray, "data should be numpy.ndarray type"
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        self.__data = data.copy()
        self.update_gdaldtype_by_npdtype()
        self.cache_data_for_plot = None

    @property
    def geo_transform(self):
        return self.__geo_transform

    @geo_transform.setter
    def geo_transform(self, geo_transform):
        if geo_transform is not None:
            assert len(geo_transform) == 6, "length of geo_transform should be 6"
            self.__geo_transform = geo_transform
        else:
            self.__geo_transform = None

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        if metadata is not None:
            assert type(metadata) == dict, "metadata should be in dict type"
            self.__metadata = metadata
        else:
            self.__metadata = None

    @property
    def pixel_size(self):
        """Return pixel resolution on both row and col axes"""
        c, a, b, f, d, e = self.geo_transform 
        res_row = (b**2 + e **2)**(1/2)
        res_col = (a**2 + d**2)**(1/2)
        return (res_row, res_col)

    @property
    def extent(self):
        return tgp.get_extent(self.rows, self.cols, self.geo_transform, return_poly=True)

    @property
    def extent_for_plot(self):
        """get the extent for matplotlib extent

        Returns
        -------
        extent: `numpy.array` or tuple. If return_poly==True, return four corner coordinates, else return
        (xmin, xmax, ymin, ymax)
        """
        return tgp.get_extent(self.rows, self.cols, self.geo_transform, return_poly=False)

    # def __getitem__(self, slice_value):
    #     if type(slice_value) in [int, slice]:
    #         h_start_inner, h_stop_inner = self.get_values_by_coords(slice_value)
    #         return self.data[h_start_inner:h_stop_inner, :]

    #     elif type(slice_value) == tuple:
    #         h_start_inner, h_stop_inner = self.get_values_by_coords(slice_value[0])
    #         w_start_inner, w_stop_inner = self.get_values_by_coords(slice_value[1])
    #         return self.data[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner]

    def astype(self, dtype, update_gdaldtype=True):
        """Change dtype of self.data.

        Parameters
        ----------
        dtype: type. Target dtype.

        update_gdaldtype: bool. Change gdaldtype according to `self.data.dtype`.
        """
        assert type(dtype) is type, "dtype should type type"
        self.data = self.data.astype(dtype)
        self.update_gdaldtype_by_npdtype()

    def get_values_by_coords(self, coords):
        """get the data values be the coordinates

        Returns
        -------
        coords: `numpy.array`. The coordinates with shape (n_points, 2). The order of
        last dimension is (lng, lat).
        """
        npidxs_row, npidxs_col = tgp.coords_to_npidxs(coords, self.geo_transform).T
        return self.data[npidxs_row, npidxs_col]

    def update_gdaldtype_by_npdtype(self):
        """Update gdaldtype according to gdaldtype using `TronGisPy.npdtype_to_gdaldtype`.
        For memory operation, numpy dtype will be used. For saving the file,
        gdal dtype will be used. If the data of raster object have being
        changed, its recomended to update the dtype before saveing the file.
        """
        self.gdaldtype = tgp.npdtype_to_gdaldtype(self.data.dtype)

    def to_file(self, fp):
        """Save the file. It is recommended to save the file using tif format, that is 
        use '.tif' as its extension.

        Parameters
        ----------
        fp: str. file path.
        """
        tgp.write_raster(fp, self.data, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata)

    def to_gdal_ds(self):
        """Convert raster object to `gdal.DataSource`.

        Returns
        -------
        ds: gdal.DataSource.
        """
        ds = tgp.write_gdal_ds(self.data, geo_transform=self.geo_transform, projection=self.projection, 
                                gdaldtype=self.gdaldtype, no_data_value=self.no_data_value)
        return ds

    def fill_na(self, no_data_value=None):
        """Fill np.nan with self.no_data_value

        Parameters
        ----------
        no_data_value: int. If None, `self.no_data_value` will be used, else self.no_data_value 
        will be re-assigned.
        """
        data = self.data.copy()
        self.no_data_value =  self.no_data_value if no_data_value is None else no_data_value
        data[np.isnan(data)] = self.no_data_value
        self.data = data

    def fill_no_data(self, mode='constant', no_data_value=None, constant=0, window_size=3, loop_to_fill_all=True, loop_limit=5, fill_na=True):
        """Fill no_data_value
        
        Parameters
        ----------
        mode: str. Should be in {'constant', 'neighbor_mean', 'neighbor_majority'}

        no_data_value: int. If None, `self.no_data_value` will be used, else self.no_data_value 
        will be re-assigned.

        constant: int. If `constant` mode is used, use the constant to fill the cell with values
        no_data_value.

        window_size: int. If `neighbor_mean` or `neighbor_majority` mode is used, the size of the 
        window of the convolution to calculate the mean value. window_size should be odd number. 
        See also `TronGisPy.Interpolation.mean_interpolation` or 
        `TronGisPy.Interpolation.majority_interpolation`.

        loop_to_fill_all: bool. If `neighbor_mean` or `neighbor_majority` mode is used, fill all 
        no_data_value until there is no no_data_value value in the data. See also 
        `TronGisPy.Interpolation.mean_interpolation` or `TronGisPy.Interpolation.majority_interpolation`.

        loop_limit: bool.  If `neighbor_mean` or `neighbor_majority` mode is used, the maximum 
        limitation on loop. if loop_to_fill_all==True, loop_limit will be considered. `-1` means 
        no limitation. See also `TronGisPy.Interpolation.mean_interpolation` or 
        `TronGisPy.Interpolation.majority_interpolation`.
        """
        self.no_data_value =  self.no_data_value if no_data_value is None else no_data_value
        if fill_na and np.sum(np.isnan(self.data)) > 0:
            self.fill_na(self.no_data_value)

        data = self.data.copy()
        if mode == 'constant':
            data[data == self.no_data_value] = constant
            self.data = data
            self.no_data_value = constant
        elif mode == 'neighbor_mean':
            for i in range(self.bands):
                self.data[:, :, i] = Interpolation.mean_interpolation(data[:, :, i], no_data_value=self.no_data_value, window_size=window_size, loop_to_fill_all=loop_to_fill_all, loop_limit=loop_limit) 
        elif mode == 'neighbor_majority':
            for i in range(self.bands):
                self.data[:, :, i] = Interpolation.majority_interpolation(data[:, :, i], no_data_value=self.no_data_value, window_size=window_size, loop_to_fill_all=loop_to_fill_all, loop_limit=loop_limit)

    def copy(self):
        """copy raster object."""
        return Raster(self.data, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata)

    def hist(self, norm=False, clip_percentage=None, log=False, bands=None, ax=None, title=None, figsize=None):
        """plot digital value histgram (distribution) of raster object.

        Parameters
        ----------
        norm: bool. Normalize the image for showing.

        clip_percentage: tuple of float. The percentage to cut the data in head and tail e.g. (0.02, 0.98)

        log: bool. Get the log value of data to show the image.

        bands: list. Which bands to plot. 

        ax: `matplotlib.axes._subplots.AxesSubplot`. On which ax the raster will
        be plot.

        figsize: tuple of float, Width and height of the histgram. 
        """
        if bands is None:
            data = self.data[self.data != self.no_data_value].flatten()
        else:
            data = self.data[:, :, bands]
            data = data[data != self.no_data_value].flatten()

        # clip_percentage
        if clip_percentage is not None:
            assert len(clip_percentage) == 2, "clip_percentage two element tuple"
            idx_st = int(len(data.flatten()) * clip_percentage[0])
            idx_end = int(len(data.flatten()) * clip_percentage[1])
            X_sorted = np.sort(data.flatten())
            data_min = X_sorted[idx_st]
            data_max = X_sorted[idx_end]
            data[data<data_min] = data_min
            data[data>data_max] = data_max
            
        # log
        if log:
            data = np.log(data)

        # normalize
        if norm:
            data = tgp.Normalizer().fit_transform(data, clip_percentage=clip_percentage)

        if ax is not None:
            ax.hist(data)
            ax.set_title(title)
        else:
            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.hist(data)
            plt.title(title)
            plt.show()

    @property
    def cache_data_for_plot(self):
        """cache the processed data (norm & fill_na) for plotting

        Returns
        -------
        cache_data: `numpy.array`. Processed data.
        """
        return self.__cache_data_for_plot

    @cache_data_for_plot.setter
    def cache_data_for_plot(self, cache_data_for_plot):
        """set value for cache_data_for_plot"""
        if cache_data_for_plot is not None:
            self.__cache_data_for_plot = cache_data_for_plot
        else:
            self.__cache_data_for_plot = None

    def plot(self, flush_cache=True, norm=True, clip_percentage=(0.02, 0.98), log=False, rescale_percentage=None, bands=None, ax=None, title=None, cmap=None, figsize=None):
        """plot raster object.

        Parameters
        ----------
        flush_cache: bool. Cache the processed result for quick plotting.

        norm: bool. Normalize the image for showing.

        clip_percentage: tuple of float. The percentage to cut the data in head and tail e.g. (0.02, 0.98)

        log: bool. Get the log value of data to show the image.

        rescale_percentage: float. The percentage to recale the image for efficient showing.

        bands: list. Which bands to plot. Length of bands should be 1, 3 or 4.
        If 3 bands is used, each of them will be defined as rgb bands. If the
        forth band is used, it will be the opacity value.

        ax: `matplotlib.axes._subplots.AxesSubplot`. On which ax the raster will
        be plot.

        cmap: string, dict or `matplotlib.colors.Colormap`. Color map used to plot
        the raster. 
    
        figsize: tuple of float, Width and height of the histgram. 
        """
        if bands is None:
           bands = [0, 1, 2] if self.bands >= 3 else [0]
        
        assert type(bands) is list, "type of bands should be list"
        assert len(bands) in [1, 3, 4], "length of bands should be 1, 3 or 4"

        if (self.cache_data_for_plot is None) or flush_cache:
            # reshape to valid shape for matplotlib
            if len(bands) == 1:
                data = self.data[:, :, bands[0]]
            else:
                data = self.data[:, :, bands]

            # deal with no data
            data = data.astype(np.float)
            data[data == self.no_data_value] = np.nan

            # detect single value
            if len(np.unique(data[~np.isnan(data)])) == 1:
                norm = False
                log = False
                clip_percentage = None

            # clip_percentage
            if clip_percentage is not None:
                assert len(clip_percentage) == 2, "clip_percentage two element tuple"
                data_notna = data[~np.isnan(data)]
                idx_st = int(len(data_notna.flatten()) * clip_percentage[0])
                idx_end = int(len(data_notna.flatten()) * clip_percentage[1])
                X_sorted = np.sort(data_notna.flatten())
                data_min = X_sorted[idx_st]
                data_max = X_sorted[idx_end]
                data[~np.isnan(data) & (data<data_min)] = data_min
                data[~np.isnan(data) & (data>data_max)] = data_max
                
            # log
            if log:
                if data[~np.isnan(data)].min() < 0: 
                    data -= data[~np.isnan(data)].min()
                data[~np.isnan(data)] = np.log(data[~np.isnan(data)] + 10**-6)

            # normalize
            if norm:
                data = tgp.Normalizer().fit_transform(data)

            if rescale_percentage is not None:
                import cv2
                data = cv2.resize(data, (int(self.cols*rescale_percentage), int(self.rows*rescale_percentage)))

            self.cache_data_for_plot = data
        else:
            data = self.cache_data_for_plot

        # flip xy
        c, a, b, f, d, e = self.geo_transform
        # lng = a * col + b * row + c
        # lat = d * col + e * row + f
        # a = d(lng) / d(col)
        # b = d(lng) / d(row)
        # d = d(lat) / d(col)
        # e = d(lat) / d(row)
        if (np.abs(a) > np.abs(d)) and (np.abs(e) > np.abs(b)):
            lng_res, lat_res = a, e
        elif (np.abs(d) > np.abs(a)) and (np.abs(b) > np.abs(e)): # if d > a, 1 col move contribute more on lat, less on lng
            data = np.flipud(np.rot90(data))
            lng_res, lat_res = d, b
        else:
            assert False, "not acceptable geotransform"
        if lng_res < 0: # general lng_res > 0
            data = np.fliplr(data)
        if lat_res > 0: # general lat_res < 0
            data = np.flipud(data)

        # plotting
        if ax is not None:
            ax.imshow(data, extent=self.extent_for_plot, cmap=cmap)
            ax.set_title(title)
        else:
            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.imshow(data, extent=self.extent_for_plot, cmap=cmap)
            plt.title(title)
            plt.show()
    