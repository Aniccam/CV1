import matplotlib.pyplot as plt
import numpy as np


def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    #
    # You code here

    u, s, vh = np.linalg.svd(A)
    s[-1] = 0
    A_hat = u @ np.diag(s) @ vh

    return A_hat
    #



def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    #
    # You code here

    l, _ = p1.shape
    A = np.zeros((2*l, 9))

    # construct A from A*f = 0
    for i in range(l):
        A[i, :] = np.array([p1[i, 0]*p2[i, 0], p1[i, 1]*p2[i, 0], p2[i, 0], p1[i, 0]*p2[i, 1], p1[i, 1]*p2[i, 1], p2[i, 1], p1[i, 0], p1[i, 1], 1])

    # compute F with svd and enforce F rank 2
    _, _, vh = np.linalg.svd(A)
    F = vh[8, :].reshape((3, 3))

    return enforce_rank2(F)

    #



def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    #
    # You code here

    ps1, T1 = condition_points(p1)
    ps2, T2 = condition_points(p2)
    F_hat = compute_fundamental(ps1, ps2)
    F = T2.T @ F_hat @ T1

    return F
    #




def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """

    #
    # You code here

    p1_h = np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1)
    l2 = p1_h @ F.T

    H, W, _ = img.shape
    l, _ = p1.shape

    X1 = np.zeros(l)
    Y1 = (- l2[:, 2] - l2[:, 0] * X1) / l2[:, 1]
    X2 = W * np.ones(l)
    Y2 = (- l2[:, 2] - l2[:, 0] * X2) / l2[:, 1]

    return X1, X2, Y1, Y2
    #



def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    #
    # You code here
    residual = np.absolute(p1 @ F @ p2.T)
    max_residual = np.max(residual)
    avg_residual = np.mean(residual)

    return max_residual, avg_residual
    #


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    #
    # You code here
    u, _, vh = np.linalg.svd(F)
    e2 = vh[-1, :-1] / vh[-1, -1]
    e1 = u[:-1, -1] / u[-1, -1]

    return e1, e2
    #
