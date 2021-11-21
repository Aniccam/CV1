import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    #
    # You code here
    return np.load(path)
    #


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #
    # You code here
    H, W = bayerdata.shape
    r= np.zeros((H, W))
    g = np.zeros((H, W))
    b = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if (i % 2) == 0 and (j % 2) != 0:
                r[i, j] = bayerdata[i, j]
            elif (i % 2) != 0 and (j % 2) == 0:
                b[i, j] = bayerdata[i, j]
            else:
                g[i, j] = bayerdata[i, j]
    return r, g, b
    #


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    H, W = r.shape
    Image = np.zeros((H, W, 3))
    Image[:, :, 0] = b
    Image[:, :, 1] = g
    Image[:, :, 2] = r
    return Image
    #


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #
    # You code here
    H, W = r.shape
    kernel_g = 1/4 * np.array([[1, 0, 1], [1, 0, 1], [0, 0, 0]])
    kernel_b = 1/4 * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    kernel_r = 1/4 * np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    g_interpolated = convolve(g, kernel_g)
    b_interpolated = convolve(b, kernel_b, mode='mirror')
    r_interpolated = convolve(r, kernel_r, mode='mirror')
    img_interpolated = np.zeros((H, W, 3))
    img_interpolated[:, :, 0] = b_interpolated
    img_interpolated[:, :, 1] = g_interpolated
    img_interpolated[:, :, 2] = r_interpolated
    return img_interpolated
    #
