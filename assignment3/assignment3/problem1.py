import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, maximum_filter

def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """

    #
    # You code here
    # smooth the image
    smoothed = convolve(img, gauss, mode='mirror')
    # derivative
    I_x = convolve(smoothed, fx, mode='mirror')
    I_xx = convolve(I_x, fx, mode='mirror')
    I_y = convolve(smoothed, fy, mode='mirror')
    I_yy = convolve(I_y, fy, mode='mirror')
    I_xy = convolve(I_x, fy, mode='mirror')

    return I_xx, I_yy, I_xy
    #



def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """

    #
    # You code here
    H, W = I_xx.shape
    criterion = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            criterion[i, j] = I_xx[i, j] * I_yy[i, j] - I_xy[i, j] ** 2
    return sigma ** 4 * criterion
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold

        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    # You code here

    criterion_allmax = maximum_filter(criterion, size=5, mode='constant')
    criterion_localmax = np.where(criterion == criterion_allmax, criterion, 0)

    # remove the image borders
    criterion_localmax[:5] = 0
    criterion_localmax[-5:] = 0
    criterion_localmax[:, :5] = 0
    criterion_localmax[:, -5:] = 0

    # find local maxima above given threshold
    criterion_thre = np.where(criterion_localmax > threshold, 1, 0)

    # get coordinates of local maxima above given threshold
    f = criterion_thre.nonzero()

    return f[0], f[1]
    #

