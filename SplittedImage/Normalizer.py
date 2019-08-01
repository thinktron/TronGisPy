class Normalizer():
    def fit(self, X, min_val=None, max_val=None):
        self.min = X.min() if min_val == None else min_val
        self.max = X.max() if max_val == None else max_val
        
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X, min_val=None, max_val=None):
        self.fit(X, min_val, max_val)
        return self.transform(X)
        
    def reverse_transform(self, Y):
        return (Y * (self.max - self.min)) + self.min
