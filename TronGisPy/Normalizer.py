import numpy as np

class Normalizer():
    """Normalize the digital values of the image into 0~1 range."""

    def clip_by_percentage(self, data, clip_percentage=(0.02, 0.98)):
        data_temp = data.copy()
        data_temp_masked = data_temp[~np.isnan(data_temp)]
        idx_st = int(len(data_temp_masked) * clip_percentage[0])
        idx_end = int((len(data_temp_masked) * clip_percentage[1]) - 10**-6)
        X_sorted = np.sort(data_temp_masked)
        data_min = X_sorted[idx_st]
        data_max = X_sorted[idx_end]
        data_temp_masked[data_temp_masked<data_min] = data_min
        data_temp_masked[data_temp_masked>data_max] = data_max
        data_temp[~np.isnan(data_temp)] = data_temp_masked
        return data_temp

    def clip_by_min_max(self, data, min_max=(0, 100)):
        data_temp = data.copy()
        data_min, data_max = min_max
        data_temp[data_temp<data_min] = data_min
        data_temp[data_temp>data_max] = data_max
        return data_temp

    def fit(self, X, min_max_val=None):
        """find the min and max for the normalizer.

        Parameters
        ----------
        X: array_like
            The image used to normalize.
        min_max_val: tuple, optional
            The min and max limitation on the normalizer e.g. (50, 150).
        """
        if min_max_val is not None:
            self.min, self.max = min_max_val
        else:
            self.min, self.max = np.nanmin(X), np.nanmax(X)
        assert self.min != self.max, "The image has only single value, that cannot be normed!"

    def transform(self, X):
        """Normalize the image.

        Parameters
        ----------
        X: array_like
            The image used to normalize.

        Returns
        -------
        Y: ndarray
            The normalized image.
        """
        X = (X - self.min) / (self.max - self.min)
        X_not_nan = X[~np.isnan(X)]
        X_not_nan[(X_not_nan<0)] = 0
        X_not_nan[(X_not_nan>1)] = 1
        return X

    def fit_transform(self, X, min_max_val=None, clip_percentage=None, clip_min_max=None):
        """Combine the fit and transform funtion.

        Parameters
        -------
        X: ndarray
            The normalized image.
        min_max_val: tuple, optional
            The min and max limitation on the normalizer e.g. (50, 150).
        clip_percentage: tuple, optional
            give a two element tuple represents the start and end percentage of 
            the data to cut in head and tail e.g. (0.02, 0.98)
        clip_min_max: tuple, optional
            give a two element tuple represents the min and max to cut in head 
            and tail e.g. (0, 100)

        Returns
        -------
        Y: ndarray
            The normalized image.
        """
        if clip_percentage is not None:
            assert not (clip_percentage is not None and clip_min_max is not None), "should not set clip_percentage and clip_min_max at the same time!"
            assert len(clip_percentage) == 2, "clip_percentage two element tuple"
            X = self.clip_by_percentage(X, clip_percentage=clip_percentage)
        elif clip_min_max is not None:
            assert len(clip_min_max) == 2, "clip_min_max two element tuple"
            X = self.clip_by_min_max(X, min_max=clip_min_max)
        self.fit(X, min_max_val=min_max_val)
        return self.transform(X)
        
    def reverse_transform(self, Y):
        """Reverse the transform funtion of te normalizer.

        Parameters
        ----------
        Y: ndarray
            The normalized image.

        Returns
        -------
        X: ndarray
            The un-normalized image.
        """
        return (Y * (self.max - self.min)) + self.min
