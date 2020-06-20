import gdal
import numpy as np
import TronGisPy as tgp
from TronGisPy import GisIO
from matplotlib import pyplot as plt
gdal.GDT_Float32
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
        self.geo_transform = geo_transform# if geo_transform is not None else [0, 1, 0, 0, 0, -1]
        self.gdaldtype = gdaldtype if gdaldtype is not None else tgp.npdtype_to_gdaldtype(data.dtype)
        self.projection = projection
        self.no_data_value = no_data_value
        self.metadata = metadata

    def __repr__(self):
        desc = ""
        desc += "shape: ({rows}, {cols}, {bands})\n".format(rows=self.rows, cols=self.cols, bands=self.bands)
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
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        assert type(data) == np.ndarray, "data should be numpy.ndarray type"
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        self.__data = data

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
    def extent(self):
        return tgp.get_extent(self.rows, self.cols, self.geo_transform, return_poly=True)

    @property
    def extent_for_plot(self):
        return tgp.get_extent(self.rows, self.cols, self.geo_transform, return_poly=False)

    def update_gdaltype_by_npdtype(self):
        """Update gdaltype according to gdaltype using `TronGisPy.npdtype_to_gdaldtype`.
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

    def copy(self):
        """copy raster object."""
        return Raster(self.data, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata)

    def plot(self, ax=None, bands=None, title=None, cmap=None):
        """plot raster object.

        Parameters
        ----------
        ax: `matplotlib.axes._subplots.AxesSubplot`. On which ax the raster will
        be plot.

        bands: list. Which bands to plot. Length of bands should be 1, 3 or 4.
        If 3 bands is used, each of them will be defined as rgb bands. If the
        forth band is used, it will be the opacity value.

        cmap: string, dict or `matplotlib.colors.Colormap`. Color map used to plot
        the raster. 
        """
        if bands is None:
           bands = [0, 1, 2] if self.bands >= 3 else [0]

        assert type(bands) is list, "type of bands should be list"
        assert len(bands) in [1, 3, 4], "length of bands should be 1, 3 or 4"

        # reshape to valid shape for matplotlib
        if len(bands) == 1:
            data = self.data[:, :, bands[0]]
        else:
            data = self.data[:, :, bands]

        # normalize
        data = tgp.Normalizer().fit_transform(data)

        # plotting
        if ax is not None:
            ax.imshow(data, extent=self.extent_for_plot, cmap=cmap)
            ax.set_title(title)
        else:
            plt.imshow(data, extent=self.extent_for_plot, cmap=cmap)
            plt.title(title)
            plt.show()
    