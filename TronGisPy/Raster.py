import gdal
import numpy as np
import TronGisPy as tgp
from TronGisPy import GisIO
from matplotlib import pyplot as plt

class Raster():
    def __init__(self, data, geo_transform=None, projection=None, gdaldtype=None, no_data_value=None, metadata=None):
        """data: should be in shape of (rows, cols, bands)"""
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
        self.gdaldtype = tgp.npdtype_to_gdaldtype(self.data.dtype)

    def to_file(self, fp):
        tgp.write_raster(fp, self.data, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata)

    def to_gdal_ds(self):
        """X should be in (n_rows, n_cols, n_bands) shape"""
        ds = tgp.write_gdal_ds(self.data, geo_transform=self.geo_transform, projection=self.projection, 
                                gdaldtype=self.gdaldtype, no_data_value=self.no_data_value)
        return ds

    def copy(self):
        return Raster(self.data, self.geo_transform, self.projection, self.gdaldtype, self.no_data_value, self.metadata)

    def plot(self, ax=None, bands=None, title=None, cmap=None):
        if bands is None:
           bands = [0, 1, 2] if self.bands >= 3 else [0]

        assert type(bands) is list, "type of bands should be list"
        assert len(bands) in [1, 3], "length of bands should be 1 or 3"

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
    