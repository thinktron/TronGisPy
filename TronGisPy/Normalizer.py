import numpy as np

class Normalizer():
    """Normalize the digital values of the image into 0~1 range.
    """

    def fit(self, X, min_max_val=None, clip_percentage=None):
        """find the min and max for the normalizer.

        Parameters
        ----------
        X: array_like
            The image used to normalize.
        min_max_val: tuple, optional
            The min and max limitation on the normalizer e.g. (50, 150).
        clip_percentage: tuple, optional
            give a two element tuple represents the percentage to cut in head and tail e.g. (0.02, 0.98)
        """
        assert not (min_max_val is not None and clip_percentage is not None), "should not set min_max_val and at the same time!"
        if clip_percentage is not None:
            assert len(clip_percentage) == 2, "clip_percentage two element tuple"
            X_flatten = X.flatten().copy()
            idx_st = int(len(X_flatten) * clip_percentage[0])
            idx_end = int(len(X_flatten) * clip_percentage[1])
            X_sorted = np.sort(X_flatten)
            self.min = X_sorted[idx_st]
            self.max = X_sorted[idx_end]
        elif min_max_val is not None:
            self.min = min_max_val[0]
            self.max = min_max_val[1]
        else:            
            self.min = np.nanmin(X)
            self.max = np.nanmax(X)
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

    def fit_transform(self, X, min_max_val=None, clip_percentage=None):
        """Combine the fit and transform funtion.

        Parameters
        -------
        X: ndarray
            The normalized image.
        min_max_val: tuple, optional
            The min and max limitation on the normalizer e.g. (50, 150).
        clip_percentage: tuple, optional
            give a two element tuple represents the percentage to cut in head and tail e.g. (0.02, 0.98)

        Returns
        -------
        Y: ndarray
            The normalized image.
        """
        self.fit(X, min_max_val=min_max_val, clip_percentage=clip_percentage)
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
