import numpy as np
import os
from PIL import Image

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    
    #
    # You code here
    x = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pgm"):
                img = Image.open(os.path.join(root, file))
                img1 = np.asarray(img)
                H, W = img1.shape
                img1 = img1.flatten()
                x.append(img1)
    x = np.array(x)

    #
    
    return x, (H, W)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 1, 2, 3, 4)

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """

    N, M = X.shape
    X = X.astype(np.float64)
    X_mean = np.mean(X, axis=0)
    X_Hut = (X - X_mean).transpose()
    u, s, vh = np.linalg.svd(X_Hut, full_matrices=False)
    lmb = np.power(s, 2) / N

    return u, lmb

#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """

    M = s.size
    total = np.sum(s)
    sum = 0.0
    for i in range(M):
        sum += s[i]
        if sum >= p * total:
            v = u[:, :i+1]
            break

    return v

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """

    a = u.transpose().dot(face_image)
    image_out = u.dot(a)

    return image_out

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (1, 4)


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """

    top = []
    Y1 = []
    N, M = Y.shape
    a = u.transpose().dot(x)
    for i in range(N):
        ai = u.transpose().dot(Y[i, :].transpose())
        dist = np.linalg.norm(a-ai)
        top.append((i, dist))
    top.sort(key=lambda t:t[1])
    [Y1.append(Y[top[i][0], :]) for i in range(top_n)]

    return np.asarray(Y1)

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    Y = []
    A = []
    _, D = u.shape
    a1 = u.transpose().dot(x1)
    a2 = u.transpose().dot(x2)
    [A.append(np.linspace(a1[i], a2[i], N)) for i in range(D)]
    A = np.asarray(A).transpose()
    for i in range(N):
        img = u.dot(A[i, :])
        Y.append(img)

    return np.asarray(Y)
