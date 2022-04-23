import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image

from graph_cut import GraphCut
from graph_cut_gui import GraphCutGui
from scipy.spatial.distance import cdist


class GraphCutController:

    def __init__(self):
        self.__init_view()
        self.connection = -1

    def __init_view(self):
        root = Tk()
        root.geometry("700x500")
        self._view = GraphCutGui(self, root)
        root.mainloop()

    # TODO: TASK 2.1
    def __get_color_histogram(self, image, seed, hist_res):
        """
        Compute a color histograms based on selected points from an image
        
        :param image: color image
        :param seed: Nx2 matrix containing the the position of pixels which will be
                    used to compute the color histogram
        :param histRes: resolution of the histogram
        :return hist: color histogram
        """
        sample = image[seed[:, 1], seed[:, 0], :]
        H, edges = np.histogramdd(sample, bins = (hist_res, hist_res, hist_res), range=[(0, 256), (0, 256), (0, 256)])
        H = ndimage.gaussian_filter(H, sigma = 0.1)
        H /= H.sum()
        return H, edges


    # TODO: TASK 2.2
    # Hint: Set K very high using numpy's inf parameter
    def __negative_log_probability(self, hist, edges, r, g, b):
        r_index = np.searchsorted(edges[0], r) - 1
        g_index = np.searchsorted(edges[1], g) - 1
        b_index = np.searchsorted(edges[2], b) - 1
        if r_index < 0 or g_index < 0 or b_index < 0 or r_index >= hist.shape[0] or g_index >= hist.shape[1] or b_index >= hist.shape[2]:
            return -np.log(1e-10)
        return -np.log(hist[r_index][g_index][b_index] + 1e-10)

    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        """

        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
        unary = np.zeros((image.shape[0] * image.shape[1], 2))
        hist_fg_H, hist_fg_edge = hist_fg
        hist_bg_H, hist_bg_edge = hist_bg

        # y x; index_fg: Nx1 numpy array 
        index_fg = np.ravel_multi_index(np.c_[seed_fg[:, 1], seed_fg[:, 0]].T, (image.shape[0], image.shape[1]))
        unary[index_fg, 1], unary[index_fg, 0] = np.inf, 0

        index_bg = np.ravel_multi_index(np.c_[seed_bg[:, 1], seed_bg[:, 0]].T, (image.shape[0], image.shape[1]))
        unary[index_bg, 1], unary[index_bg, 0] = 0, np.inf

        index_other = np.setdiff1d(np.arange(image.shape[0] * image.shape[1], dtype=int), np.union1d(index_fg, index_bg))
        
        for idx in index_other:
            i, j = np.unravel_index(idx, (image.shape[0], image.shape[1]))
            r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
            unary[idx, 1] = lambda_param * self.__negative_log_probability(hist_bg_H, hist_bg_edge, r, g, b)
            unary[idx, 0] = lambda_param * self.__negative_log_probability(hist_fg_H, hist_fg_edge, r, g, b)
        return unary
            
    # TODO: TASK 2.3
    # Hint: Use coo_matrix from the scipy.sparse library to initialize large matrices
    # The coo_matrix has the following syntax for initialization: coo_matrix((data, (row, col)), shape=(width, height))
    def __get_neighbors(self, image, point):
        x, y = np.unravel_index(point, (image.shape[0], image.shape[1]))
        neighbors = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                nx, ny = x + i, y + j
                if nx < image.shape[0] and ny < image.shape[1] and nx >= 0 and ny >=0 and not(i == 0 and j == 0):
                    neighbors.append([nx, ny])
        neighbors = np.asarray(neighbors,  dtype=int)
        neighbors = np.ravel_multi_index(neighbors.T, (image.shape[0], image.shape[1]))
        return neighbors

    def __get_ad_hoc(self, image, point, neighbor_idx):
        sigma = 5
        center_pos = np.unravel_index(point, (image.shape[0], image.shape[1]))
        neighbors_pos = np.unravel_index(neighbor_idx, (image.shape[0], image.shape[1]))
        center, neighbors = image[center_pos[0],center_pos[1], :], image[neighbors_pos[0],neighbors_pos[1], :]
        center_pos = np.asarray(center_pos, dtype = int)
        neighbors_pos = np.asarray(neighbors_pos, dtype = int).T
        dist = np.exp( -cdist(center[None, :], neighbors, 'sqeuclidean') / (2 * (sigma**2)) ) / cdist(center_pos[None, :], neighbors_pos, 'euclidean')
        return dist[0]

    def __get_pairwise(self, image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        N = image.shape[0] * image.shape[1]
        num_sparse = (image.shape[0] - 2) * (image.shape[1] - 2) * 8 + (image.shape[0] - 2) * 2 * 5 + (image.shape[1] - 2) * 2 * 5 + 4 * 3
        self.connection = num_sparse
        row, col, data = np.zeros(num_sparse, dtype = int), np.zeros(num_sparse, dtype = int), np.zeros(num_sparse)
        cnt = 0
        for point in range(N):
            neighbor_idx = self.__get_neighbors(image, point)
            row[cnt:cnt + neighbor_idx.shape[0]] = np.ones(neighbor_idx.shape, dtype = int) * point
            col[cnt:cnt + neighbor_idx.shape[0]] = neighbor_idx
            data[cnt:cnt + neighbor_idx.shape[0]] = self.__get_ad_hoc(image, point, neighbor_idx)
            cnt += neighbor_idx.shape[0]
        pairwise = coo_matrix((data, (row, col)), shape=(N, N))
        return pairwise
        
    # TODO TASK 2.4 get segmented image to the view
    def __get_segmented_image(self, image, labels, background=None):
        """
        Return a segmented image, as well as an image with new background 
        :param image: color image as a numpy array
        :param label: labels a numpy array
        :param background: color image as a numpy array
        :return image_segmented: image as a numpy array with red foreground, blue background
        :return image_with_background: image as a numpy array with changed background if any (None if not)
        """
        image_segmented, image_with_background = image.copy(), None
        labels = np.reshape(labels, (image.shape[0], image.shape[1]))

        image_segmented[labels==0, 1] = 0
        image_segmented[labels==0, 2] = 0
        image_segmented[labels==1, 0] = 0
        image_segmented[labels==1, 1] = 0

        if background is not None:
            image_with_background = image.copy()
            background = Image.fromarray(background[:, :, :3], 'RGB').resize((image.shape[1], image.shape[0]))
            background = np.asarray(background)
            image_with_background[labels==1, :] = background[labels==1, :]
        return image_segmented, image_with_background

    def segment_image(self, image, seed_fg, seed_bg, lambda_value, background=None):
        image_array = np.asarray(image)
        background_array = None
        if background:
            background_array = np.asarray(background)
        seed_fg = np.array(seed_fg)
        seed_bg = np.array(seed_bg)
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        # TODO: TASK 2.1 - get the color histogram for the unaries
        hist_res = 32
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TODO: TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        # TODO: TASK 2.4 - perform graph cut
        # Your code here
        optimizer = GraphCut(unaries.shape[0], self.connection)
        optimizer.set_unary(unaries)
        optimizer.set_neighbors(pairwise)
        optimizer.minimize()
        labels = optimizer.get_labeling()

        # TODO TASK 2.4 get segmented image to the view
        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels,
                                                                                      background_array)
        # transform image array to an rgb image
        segmented_image = Image.fromarray(segmented_image, 'RGB')
        self._view.set_canvas_image(segmented_image)
        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            plt.imshow(segmented_image_with_background)
            plt.show()
