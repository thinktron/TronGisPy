import numpy as np

class Normalizer():
    def fit(self, X, min_max_val=None, clip_percentage=None):
        """
        clip_percentage: give a two element tuple represents the percentage to cut in head and tail e.g. (0.02, 0.98)
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

    def transform(self, X):
        X = (X - self.min) / (self.max - self.min)
        X_not_nan = X[~np.isnan(X)]
        X_not_nan[(X_not_nan<0)] = 0
        X_not_nan[(X_not_nan>1)] = 1
        return X

    def fit_transform(self, X, min_max_val=None, clip_percentage=None):
        self.fit(X, min_max_val=min_max_val, clip_percentage=clip_percentage)
        return self.transform(X)
        
    def reverse_transform(self, Y):
        return (Y * (self.max - self.min)) + self.min
