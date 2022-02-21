import numpy as np


def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    #
    # Your code goes here
    cost_ssd = 0.0

    cost_ssd = np.sum((patch1 - patch2) ** 2)
    # m, _, _ = patch1.shape
    #
    # for i in range(m):
    #     for j in range(m):
    #         diff = patch1[i, j] - patch2[i, j]
    #         cost_ssd += diff**2
    #
    # cost_ssd = -1

    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """

    #
    # Your code goes here
    cost_nc = 0.0
    p1 = patch1.flatten()
    p2 = patch2.flatten()

    p1_ = p1 - np.mean(p1)
    p2_ = p2 - np.mean(p2)
    cost_nc = (p1_ @ p2_) / (np.linalg.norm(p1_) * np.linalg.norm(p2_))
    #
    # cost_nc = -1

    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value        
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape 

    #
    # Your code goes here
    m, _ = patch1.shape
    cost_val = 1/m**2 * cost_ssd(patch1, patch2) + alpha * cost_nc(patch1, patch2)
    #
    # cost_val = -1
    
    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    #
    # Your code goes here
    padding_size = window_size // 2
    if padding_mode == 'constant':
        padded_img = np.pad(input_img, padding_size, 'constant', constant_values=0)
    elif padding_mode == 'symmetric':
        padded_img = np.pad(input_img, padding_size, 'symmetric')
    else:
        padded_img = np.pad(input_img, padding_size, 'reflect')
    #
    # padded_img = input_img.copy()

    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2 
    assert padded_img_r.ndim == 2 
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    #
    # Your code goes here
    H, W = padded_img_l.shape
    disparity = np.zeros((H-window_size+1, W-window_size+1))
    max_shift = window_size if window_size > max_disp else max_disp
    for i in range(H-window_size):
        for j in range(max_shift, W-window_size):
            cost_min = float('inf')
            patch1 = padded_img_l[i:i+window_size, j:j+window_size]
            for d in range(max_disp):
                patch2 = padded_img_r[i:i+window_size, j-d:j+window_size-d]
                cost = cost_function(patch1, patch2, alpha)
                if cost < cost_min:
                    cost_min = cost
                    disparity[i, j] = d
    #
    # disparity = padded_img_l.copy()

    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    #
    # Your code goes here
    H, W = disparity_res.shape
    N = H * W
    aepe = np.sum(np.absolute(disparity_gt-disparity_res)) / N
    #
    # aepe = -1

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    alpha = -0.01
    # alpha = -0.06 AEPE = 1.323
    # alpha = -0.01 AEPE = 1.322
    # alpha = 0.04 AEPE = 5.131
    # alpha = 0.1 AEPE = 6.922
    #
    # alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (1, 2, 1, 2)
