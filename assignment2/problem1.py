import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import convolve2d

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    #
    # TODO
    imgsdir = path + "/facial_images/"
    ftdir = path + "/facial_features/"
    imgsdirList = sorted(os.listdir(imgsdir))
    ftdirList = sorted(os.listdir(ftdir))
    for name in imgsdirList:
        imgs.append(plt.imread(imgsdir + name))

    for name in ftdirList:
        feats.append(plt.imread(ftdir + name))

    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    #
    # TODO
    middle = fsize // 2
    kernel = np.zeros((fsize, fsize), dtype=np.float)
    for x in range(-middle, -middle + fsize):
        for y in range(-middle, -middle + fsize):
            kernel[x + middle, y+middle] =  1/ (2*sigma) *math.exp(-(x**2 + y**2) / (2 * sigma **2))

    return kernel/ kernel.sum()


def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    #
    # TODO
    #
    H, W = x.shape
    dH, dW = H//factor, W//factor   # downsampled size
    downsample = np.empty((dH, dW))
    for i in range(dH):
        for j in range(dW):
            downsample[i, j] = x[factor*i, factor*j]


    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = [img]

    #
    # TODO
    kernel = gaussian_kernel(fsize, sigma)
    for _ in range(nlevels-1):
        filtered = scipy.signal.convolve2d(img, kernel, mode="same")
        downsampled = downsample_x2(filtered, factor=2)
        img = downsampled
        GP.append(downsampled)

    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    #
    # TODO

    # Dot Product
    # norm_v1 = np.linalg.norm(np.array(v1))
    # norm_v2 = np.linalg.norm(np.array(v2))
    # distance = np.dot(v1.T,v2) / (norm_v1 * norm_v2)

    # SSD
    distance2 = np.linalg.norm(np.array(v1) - np.array(v2))
    #
    # We think with SSD we can anticipate the results are more robust and need less computational cost.
    return distance2


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    min_score = 1000000000000000

    #
    # TODO
    H, W = img.shape
    h, w = feat.shape
    v2 = feat.copy().flatten()
    scores = []
    if H*W < h*w:
        return 100000000000
    else:
        for i in range(0, H-h, step):
            for j in range(0, W-w, step):
                v1 = img[i:i+h, j:j+w].flatten()
                scores.append(template_distance(v1, v2) )
        return np.min(scores)


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (2,1,3,4)  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    # (score, g_im, feat) = (None, None, None)

    #
    # TODO



    for feat in feats:
        scoresortbyimage = []
        gimssortbyimage = []
        for img in imgs:
            GPs = gaussian_pyramid(img, 3, 5, 1.4)
            scores = []
            gims = []
            for pyramid in GPs:
                scores.append(sliding_window(pyramid, feat))
                gims.append(pyramid)
            scoresortbyimage.append(scores)
            gimssortbyimage.append(gims)
        loc = np.unravel_index((np.array(scoresortbyimage)).argmin(), (np.array(scoresortbyimage)).shape)
            # np.argmin(np.array(scoresortbyimage))
        print(loc)
        match.append((np.array(scoresortbyimage), gimssortbyimage[loc[0]][loc[1]], feat ))

    #

    return match