import numpy as np

def _get_M(omega, phi, kappa):
    """Collinearity Equation"""
    cos, sin = np.cos(omega), np.sin(omega)
    Mo = np.array([[    1,    0,    0], 
                   [    0,  cos,  sin],
                   [    0, -sin,  cos]])
    cos, sin = np.cos(phi), np.sin(phi)
    Mp = np.array([[  cos,    0, -sin],
                   [    0,    1,    0],
                   [  sin,    0,  cos]])
    cos, sin = np.cos(kappa), np.sin(kappa)
    Mk = np.array([[  cos,  sin,    0],
                   [ -sin,  cos,    0],
                   [    0,    0,    1]])
    M = np.matmul(np.matmul(Mk, Mp), Mo)
    return M

def _convert_npidxs_to_imxys(npidxs, rows=13824, cols=7680, pixel_size=0.012):
    """
    image space linear transform
    pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    npidxs
    """
    col_zoomout, row_zoomout = pixel_size, -pixel_size
    row_idxs, col_idxs = npidxs.T
    xs = (col_idxs - (cols/2)) * col_zoomout
    ys = (row_idxs - (rows/2)) * row_zoomout
    imxys = np.array([xs, ys]).T
    return imxys

def _convert_imxyzs_to_npidxs(imxyzs, rows=13824, cols=7680, pixel_size=0.012):
    """
    image space linear transform
    pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    """
    col_zoomin, row_zoomin = 1/pixel_size, -1/pixel_size
    xs, ys = imxyzs.T[0], imxyzs.T[1]
    col_idxs = col_zoomin * xs + (cols/2)
    row_idxs = row_zoomin * ys + (rows/2)
    npidxs = np.array([row_idxs, col_idxs]).T
    return npidxs

def project_npidxs_to_XYZs(P_npidxs, P_Z, aerotri_params, return_k=False):
    """
    idxs conversion from image space and object space.
    P_xyzs: xyzs of points in image space.
    P_Z: assume height.
    return_k: assume height.
    """
    # aerotri_params
    opk, L_XYZ, rows, cols, focal_length, pixel_size = aerotri_params
    # L_XYZ = np.expand_dims(L_XYZ, axis=-1)

    # calculate P_imxyzs from P_npidxs
    P_xys = _convert_npidxs_to_imxys(np.array(P_npidxs), rows, cols, pixel_size) # (n points, 2 dim)
    P_xyzs = np.concatenate([P_xys, np.full((len(P_xys), 1), -focal_length)], axis=1).T # add -focal_length as new dim to (3 dim, n points)

    # calculate P_XYZs from P_imxyzs
    M, L_Z = _get_M(*opk), L_XYZ[2]
    mt_xyzs = np.matmul(M.T, P_xyzs)
    ks = mt_xyzs[2] / (P_Z - L_Z) # convert from (L_Z - P_Z) = P_xyzs[2] / k, and query for k
    P_XYZs = ((mt_xyzs / ks) + np.expand_dims(L_XYZ, axis=-1)).T # 4 points, 3 dim
    if not return_k:
        return P_XYZs
    else:
        return P_XYZs, ks


def project_XYZs_to_npidxs(P_XYZs, aerotri_params, return_k=False):
    """
    idxs conversion from object space and image space.
    P_XYZs: XYZs of points in object space.
    aerotri_params: opk, L_XYZ, rows, cols, focal_length, pixel_size
        opk: omega, phi, kappa.
        L_XYZ: the location of the camera.
        focal_length: dmc image is 120 (mm).
        pixel_size: dmc image is sensor_width/image_cols = sensor_height/image_rows = 0.012 (mm).
    """
    # aerotri_params
    opk, L_XYZ, rows, cols, focal_length, pixel_size = aerotri_params
    
    # calculate P_imxyzs from P_XYZs
    M = _get_M(*opk)
    P_XYZs = np.array(P_XYZs).T  # =>P_XYZs.shape==(3, None)
    L_XYZ = np.array(L_XYZ).reshape(3, 1) # =>L_XYZ.shape==(3, 1)
    dXYZs = (P_XYZs-L_XYZ) # =>xyz.shape==(3, None)
    m_XYZs = np.matmul(M, dXYZs)
    ks = -focal_length / m_XYZs[2]
    P_xyzs = (ks * m_XYZs).T # (n points, 3 dim)
    
    # calculate P_npidxs from P_xyzs
    P_npidxs = _convert_imxyzs_to_npidxs(P_xyzs, rows, cols, pixel_size)
    if not return_k:
        return P_npidxs
    else:
        return P_npidxs, ks



def get_img_pixels_XYZs(npidxs, aerotri_params, k=-1):
    """
    idxs conversion from image space and object space.
    P_xyzs: xyzs of points in image space.
    """
    # aerotri_params
    opk, L_XYZ, rows, cols, focal_length, pixel_size = aerotri_params

    # calculate P_imxyzs from npidxs
    P_xys = _convert_npidxs_to_imxys(np.array(npidxs), rows, cols, pixel_size) # (n points, 2 dim)
    P_xyzs = np.concatenate([P_xys, np.full((len(P_xys), 1), -focal_length)], axis=1).T # add -focal_length as new dim to (3 dim, n points)

    # calculate P_XYZs from P_imxyzs
    M, L_Z = _get_M(*opk), L_XYZ[2]
    mt_xyzs = np.matmul(M.T, P_xyzs)
    P_XYZs = (L_XYZ - (mt_xyzs / k).T) # 4 points, 3 dim
    return P_XYZs