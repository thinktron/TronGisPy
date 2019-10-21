import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, n_clusters, no_data_value=0):
    shape_ori = X.shape
    X = X.reshape(-1, X.shape[2])
    X_idxs = np.arange(X.shape[0])
    keep_idxs = np.where(np.mean(X, axis=1) != 0)

    kmeans = KMeans(n_clusters, random_state=0, n_jobs=16).fit(X[keep_idxs])
    X_labels = np.zeros(X.shape[0])
    X_labels[:] = -1
    X_labels[keep_idxs] = kmeans.labels_
    X_labels = X_labels.reshape(shape_ori[0], shape_ori[1])
    
    return X_labels