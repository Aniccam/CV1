import random

import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
    """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """

    #
    # You code here
    middle = fsize // 2
    g = np.zeros((fsize, fsize), dtype=np.float)
    for x in range(-middle, -middle + fsize):
        for y in range(-middle, -middle + fsize):
            g[x+middle, y+middle] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g
    #


def createfilters():
    """
  Returns:
    fx, fy: filters as described in the problem assignment
  """

    #
    # You code here
    fx = np.zeros((3, 3))
    fy = np.zeros((3, 3))
    gy = gauss2d(0.9, 3)[:, 0]
    gx = gauss2d(0.9, 3)[0, :]
    fx[:, 0] = gy
    fx[:, 2] = -gy
    fy[0, :] = gx
    fy[2, :] = -gx
    fx = fx / (2 * gy.sum())
    fy = fy / (2 * gy.sum())
    return fx, fy
    #


def filterimage(I, fx, fy):
    """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

    #
    # You code here
    Ix = ndimage.convolve(I, fx)
    Iy = ndimage.convolve(I, fy)
    return Ix, Iy
    #


def detectedges(Ix, Iy, thr):
    """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

    #
    # You code here
    '''
    for the threshold, we choose 0.09, because the larger threshold is, we lost more details.
    Meanwhile the edages are more clear. But too big threshold causes edges losing. 
    0.09 is almost same as the 0.1, but 0.09 performs better than 0.1 with Non-maximum suppression.
    So at the end we choose 0.09
    '''
    Imagnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    edges = np.where(Imagnitude >= thr, Imagnitude, 0)
    return edges
    #


def nonmaxsupp(edges, Ix, Iy):
    """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """

    # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]

    # You code here
    edges2 = edges
    Itheta = np.arctan2(Iy, Ix) * 180 / np.pi
    H, W = edges.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            if -22.5 < Itheta[i, j] <= 22.5:
                if edges[i - 1, j] > edges[i, j] or edges[i + 1, j] > edges[i, j]:
                    edges2[i, j] = 0


    # handle left-to-right edges: theta in (-22.5, 22.5]

    # You code here
            elif -90 <= Itheta[i, j] <= -67.5 or 67.5 < Itheta[i, j] <= 90:
                if edges[i, j-1] > edges[i, j] or edges[i, j+1] > edges[i, j]:
                    edges2[i, j] = 0

    # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

    # Your code here
            elif -67.5 <= Itheta[i, j] <= -22.5:
                if edges[i-1, j+1] > edges[i, j] or edges[i+1, j-1] > edges[i, j]:
                    edges2[i, j] = 0

    # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

    # Your code here
            elif 22.5 < Itheta[i, j] <= 67.5:
                if edges[i-1, j-1] > edges[i, j] or edges[i+1, j+1] > edges[i, j]:
                    edges2[i, j] = 0

    return edges2
