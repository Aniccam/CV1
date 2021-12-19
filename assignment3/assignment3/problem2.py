import numpy as np


class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
    #
    # You code here
        _, m = features1.shape
        _, n = features2.shape
        distances = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                distances[i, j] = np.linalg.norm(features1[:, j] - features2[:, i])
        return distances
    #

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        
    #
    # You code here
        n, m = distances.shape
        # initialize pairs
        pairs = np.zeros((min(n, m), 4))

        # find the smallest distance
        if n <= m:
            a = np.argmin(distances, axis=1)
            for i in range(n):
                pairs[i, :] = np.append(p1[a[i], :], p2[i, :])
        else:
            a = np.argmin(distances, axis=0)
            for i in range(m):
                pairs[i, :] = np.append(p1[i, :], p2[a[i], :])

        return pairs
    #


    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        
    #
    # You code here
        m, _ = p1.shape
        sample1 = np.zeros((k, 2))
        sample2 = np.zeros((k, 2))
        index = np.random.choice(m, k)
        for i in range(k):
            sample1[i, :] = p1[index[i], :]
            sample2[i, :] = p2[index[i], :]

        return sample1, sample2
    #


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """

    #
    # You code here
        # initialization
        l, _ = points.shape
        T = np.zeros((3, 3))
        points_homo = np.ones((l, 3))
        points_homo[:, :2] = points

        # construct T
        points_mean = np.mean(points_homo, axis=0)
        max_element = 1/2 * np.max(np.absolute(points_homo), axis=0)
        T[0, 0] = 1 / max_element[0]
        T[0, 2] = - points_mean[0] / max_element[0]
        T[1, 1] = 1 / max_element[1]
        T[1, 2] = - points_mean[1] / max_element[1]
        T[2, 2] = 1

        # construct ps
        ps = points_homo.dot(T.transpose())

        return ps, T
    #


    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices should be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

    #
    # You code here
        l, _ = p1.shape
        A = np.zeros((2*l, 9))

        # construct A from A*HC = 0
        for i in range(l):
            A[2*i, :] = np.array([0, 0, 0, p1[i, 0], p1[i, 1], 1, -p1[i, 0]*p2[i, 1], -p1[i, 1]*p2[i, 1], -p2[i, 1]])
            A[2*i+1, :] = np.array([-p1[i, 0], -p1[i, 1], -1, 0, 0, 0, p1[i, 0]*p2[i, 0], p1[i, 1]*p2[i, 0], p2[i, 0]])

        # compute HC with svd
        _, _, Vh = np.linalg.svd(A)
        HC = Vh[8, :].reshape((3, 3))
        HC /= HC[2, 2]

        # construct H
        H = np.linalg.pinv(T2).dot(HC).dot(T1)
        H /= H[2, 2]

        return H, HC
    #


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

    #
    # You code here
        l, _ = p.shape
        p_homo = np.ones((l, 3))
        p_homo[:, :2] = p
        points_homo = (H.dot(p_homo.T)).T
        points = (points_homo[:, :2] / points_homo[:, -1:])
        np.seterr(divide='ignore', invalid='ignore')

        return points
    #


    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
    #
    # You code here
        p1_trans = self.transform_pts(p1, H)
        p2_trans = self.transform_pts(p2, np.linalg.pinv(H))
        dist1 = np.linalg.norm(p1_trans - p2, axis=1)
        dist2 = np.linalg.norm(p1 - p2_trans, axis=1)
        dist = np.power(dist1, 2) + np.power(dist2, 2)

        return dist

    #


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
    #
    # You code here
        l, _ = pairs.shape
        N = 0
        inliers = []
        for i in range(l):
            if dist[i] < threshold:
                N += 1
                inliers.append(pairs[i, :])

        return N, np.asarray(inliers)
    #


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
    #
    # You code here
        return np.ceil(np.log(1 - z) / np.log(1 - p ** k))
    #



    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
    #
    # You code here
        p1 = pairs[:, :2]
        p2 = pairs[:, -2:]
        max_inliers = 0

        for i in range(int(n_iters)):
            sample1, sample2 = self.pick_samples(p1, p2, k)
            p1_con, T1 = self.condition_points(sample1)
            p2_con, T2 = self.condition_points(sample2)
            H_i, HC_i = self.compute_homography(p1_con, p2_con, T1, T2)
            dist = self.compute_homography_distance(H_i, p1, p2)
            N, N_inliers = self.find_inliers(pairs, dist, threshold)
            if N > max_inliers:
                max_inliers = N
                inliers = N_inliers
                H = H_i

        return H, max_inliers, inliers
    #


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
    #
    # You code here
        l, _ = inliers.shape
        p1 = inliers[:, :2]
        p2 = inliers[:, -2:]
        A = np.zeros((2 * l, 9))

        # construct A from A*HC = 0
        for i in range(l):
            A[2 * i, :] = np.array(
                [0, 0, 0, p1[i, 0], p1[i, 1], 1, -p1[i, 0] * p2[i, 1], -p1[i, 1] * p2[i, 1], -p2[i, 1]])
            A[2 * i + 1, :] = np.array(
                [-p1[i, 0], -p1[i, 1], -1, 0, 0, 0, p1[i, 0] * p2[i, 0], p1[i, 1] * p2[i, 0], p2[i, 0]])

        # compute HC with svd
        _, _, Vh = np.linalg.svd(A)
        H = Vh[8, :].reshape((3, 3))
        H /= H[2, 2]

        return H
    #