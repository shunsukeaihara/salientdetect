# -*- coding: utf-8 -*-
import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.io import imread
import os
from salientdetect.detector import calc_saliency_score


def _load_dist_mat():
    npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_dist.npy")
    return np.load(npy_path)

DIST_MAT = _load_dist_mat()


def saliency_score(path):
    img = imread(path)
    return saliency_score_from_ndarry(img)


def saliency_score_from_ndarry(img):
    segment_labels = slic(img_as_float(img), n_segments=20, compactness=20, sigma=2.0)
    return calc_saliency_score(img, segment_labels, DIST_MAT)

