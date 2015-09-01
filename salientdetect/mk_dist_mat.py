# -*- coding: utf-8 -*-
import numpy as np
from skimage.color import rgb2lab
from skimage.color import deltaE_cie76
import os

if __name__ == "__main__":
    # pre conpute distance matrix for reduced lab color space
    colors = []
    for r in range(12):
        for g in range(12):
            for b in range(12):
                colors.append( (r*21+10, g*21+10, b*21+10))
    base_color = rgb2lab(np.array(colors, dtype=np.uint8).reshape(864,2,3)).reshape(1728,3)
    mat = np.zeros((1728,1728))
    for i in range(1728):
        for j in range(i + 1, 1728):
            mat[i, j] = deltaE_cie76(base_color[i], base_color[j])
            mat[j, i] = mat[i, j]
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_dist"), mat)
