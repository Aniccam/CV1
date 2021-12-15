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
    #
    h, w = np.shape(img)
    Ix = np.zeros(h, w)
    Iy = np.zeros(h, w)
    I_xx = np.zeros(h, w)
    I_yy = np.zeros(h, w)
    I_xy = np.zeros(h, w)
    # mirror boundry conditions
    mirror = np.copyMakeBorder(img, 1, 1, 1, 1, np.BORDER_CONSTANT, value=0)
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            Ix[i, j]  = ndimage.convolve(mirror[i, j], fx)
            Iy[i, j] = ndimage.convolve(mirror[i, j], fy)
            I_xx[i, j] = ndimage.convolve(Ix[i, j], fx)
            I_yy[i, j] = ndimage.convolve(Iy[i, j], fy)
            I_xy[i, j] = ndimage.convolve(Iy[i, j], fx)

    return I_xx, I_yy, I_xy


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
    #
    h, w = np.shape(I_xx)
    criterion = np.zeros(h,w)
    for i in range(0,w):
        for j in range(0,h):
            criterion[i,j] = sigma(I_xx[i,j] * I_yy[i,j] - I_xy[i,j] * I_xy[i,j])
    return criterion

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
    #
    h,w = np.shape(criterion)
    rows = np.zeros(h)
    cols = np.zeros(w)
    mirror = np.copyMakeBorder(criterion, 2, 2, 2, 2, np.BORDER_CONSTANT, value=0)
    for i in range(0, w-1):
        for j in range(0, h-1):
            for t in range(-2, 2):
                for s in range(2, 2):
                    #if(i - t <= 0):
                    if(mirror[i - t, j - s] > threshold):
                        rows[j] = j
                        cols[i] = i
    #max = 0
    #if(criterion[i - t, j - s] > max):
       #max = criterion[i - t, j - s]

    return rows, cols
