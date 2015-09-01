# -*- coding: utf-8 -*-
cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport sqrt, exp
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map as map
from cython.operator cimport dereference as derefit, preincrement as incit
from libc.stdlib cimport malloc, free


cdef struct Coordinate:
    int x, y

cdef struct DoubleCoordinate:
    double x, y

ctypedef vector[Coordinate*] coordary
ctypedef unsigned char UInt8


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void label_to_regions(int[:, ::1] &labels, map[int, coordary*] &region_map):
    cdef int height, width, y , x, region
    cdef Coordinate *coord
    height = labels.shape[0]
    width = labels.shape[1]
    cdef int[:,::1] regions = -np.ones((height, width), dtype=np.int32)
    label_to_region(width, height, labels, regions)
    for y in range(height):
        for x in range(width):
            region = regions[y, x]
            if region == -1:
                continue
            if region_map.find(region) == region_map.end():
                region_map[region] = new coordary()
            coord = <Coordinate*>malloc(sizeof(Coordinate))
            coord.x = x
            coord.y = y
            region_map[region].push_back(coord)

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void calc_region_centroid(map[int, coordary*] &region_map,  map[int, DoubleCoordinate*] &centroid_map):
    cdef coordary *ary
    cdef vector[Coordinate*].iterator ait
    cdef DoubleCoordinate *centroid
    cdef int count
    cdef map[int, coordary*].iterator it = region_map.begin()
    while it != region_map.end():
        count = 0
        centroid = <DoubleCoordinate *>malloc(sizeof(DoubleCoordinate))
        centroid.x = 0.0
        centroid.y = 0.0
        ary = derefit(it).second
        ait = ary.begin()
        while ait != ary.end():
            centroid.x += derefit(ait).x
            centroid.y += derefit(ait).y
            count += 1
            incit(ait)
        # calc avg
        centroid.x = centroid.x / count
        centroid.y = centroid.y /count
        centroid_map[derefit(it).first] = centroid
        incit(it)


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double calc_normalized_euclidian_distance(DoubleCoordinate *a, DoubleCoordinate *b, int height, int width):
    cdef double a_x = a.x / <double> width
    cdef double a_y = a.y / <double> height
    cdef double b_x = b.x / <double> width
    cdef double b_y = b.y / <double> height
    return sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void calc_region_score(UInt8[:,:,::1] &img, map[int, coordary*] &regions,
                            map[int, DoubleCoordinate*] &region_centroid,
                            map[int, double*] &region_histgram, double[:, ::1] &color_dist,
                            map[int, double] &region_scores):
    cdef double pixels = <double>(img.shape[0] * img.shape[1])
    cdef double score, dist, weight, d
    cdef int regA, regB
    cdef map[int, coordary*].iterator ait, bit
    ait = regions.begin()
    while ait != regions.end():
        score = 0.0
        regA = derefit(ait).first
        if regions[regA].size() < 100:  # ToDo: remove hard-coded magic number
            incit(ait)
            continue
        bit = regions.begin()
        while bit != regions.end():
            regB = derefit(bit).first
            if regA == regB:
                incit(bit)
                continue
            elif regions[regB].size() < 100:  # ToDo: remove hard-coded magic number
                incit(bit)
                continue
            else:
                # ToDo: caching
                dist = calc_normalized_euclidian_distance(region_centroid[regA], region_centroid[regB],
                                                          img.shape[0], img.shape[1])
                weight = exp(-dist/0.4)
                d = weight * calc_region_distance(region_histgram[regA],
                                                  region_histgram[regB], color_dist) * (float(regions[regB].size())/pixels)
                score += d
            incit(bit)
        region_scores[regA] = score
        incit(ait)

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double calc_region_distance(double *histA, double *histB, double[:, ::1] &color_dist_mat):
    cdef double ret = 0.0
    cdef int i, j
    for i in range(1728):
        for j in range(i + 1, 1728):
            ret += histA[i] * histB[j] * color_dist_mat[i, j] 
    return ret

cdef int check_boundary(int x, int y, int width, int height):
    if x < 0 or y < 0:
        return 0
    elif x >= width or y >= height:
        return 0
    else:
        return 1

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef int check_same_label(int x, int y, int width, int height, int label, int[:, ::1] &labeled):
    if check_boundary(x, y, width, height) == 0:
        return 0
    if label == labeled[y, x]:
        return 1
    return 0

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void replace_region_id(int x, int y, int width, int height, int region_id, int to_region_id, int[:, ::1] &regions):
    if check_boundary(x, y, width, height) == 0:
        return 
    if regions[y, x] == region_id:
        regions[y,  x] = to_region_id
        replace_region_id(x, y - 1, width, height, region_id, to_region_id, regions)
        replace_region_id(x - 1, y, width, height, region_id, to_region_id, regions)
        replace_region_id(x - 1, y - 1, width, height, region_id, to_region_id, regions)
        replace_region_id(x + 1, y - 1, width, height, region_id, to_region_id, regions)

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void label_to_region(int width, int height,  int[:, ::1] &labeled, int[:, ::1] &regions):
    regions[0, 0] = 0  # initialize region_id
    cdef int x, y
    cdef int label
    cdef int region_id = 0
    for y in range(height):
        for x in range(width):
            if x == 0 and y == 0:  # skipping
                continue
            label = labeled[y, x]
            if check_same_label(x - 1, y, width, height, label, labeled) == 1:
                # same as the left pixel
                regions[y, x] = regions[y, x - 1]

            if check_same_label(x, y - 1, width, height, label, labeled) == 1:
                # same as the upper pixel
                if regions[y, x] != -1 and regions[y, x] != regions[y - 1, x]:
                    # set region id recursively when label doese not match with upper pixel and region_id is already not -1
                    replace_region_id(x, y - 1,  width, height, regions[y - 1, x], regions[y, x], regions)
                regions[y, x] = regions[y - 1, x]

            if check_same_label(x - 1, y - 1, width, height, label, labeled) == 1:
                # same as the upper left pixel
                if regions[y, x] != -1 and regions[y, x] != regions[y - 1, x - 1]:
                    # set region id recursively when label doese not match with upper left pixel and region_id is already not -1
                    replace_region_id(x - 1, y - 1,  width, height, regions[y - 1, x - 1], regions[y, x], regions)
                regions[y, x] = regions[y - 1, x - 1]
                
            if check_same_label(x + 1, y - 1, width, height, label, labeled) == 1:
                # same as the upper right pixel
                if regions[y, x] != -1 and regions[y, x] != regions[y - 1, x + 1]:
                    # set region id recursively when label doese not match with upper right pixel and region_id is already not -1
                    replace_region_id(x + 1, y - 1,  width, height, regions[y - 1, x + 1], regions[y, x], regions)
                regions[y, x] = regions[y - 1, x + 1]
            
            if regions[y, x] == -1:
                # set new region id when label is not match the neighborhood pixel
                region_id += 1
                regions[y, x] = region_id

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void create_color_idx(UInt8[:, :, ::1] img, int width, int hight, int[:, ::1] &reduced):
    cdef int x, y
    for y in range(hight):
        for x in range(width):
            reduced[y, x] = img[y, x, 0] * 144 + img[y, x, 1] * 12 + img[y, x, 2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef normalize_dvec(double *vec, int l):
    cdef double s = 0.0
    cdef int i
    for i in range(l):
        s += vec[i]
    for i in range(l):
        vec[i] = vec[i] / s

        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_region_histgram(map[int, coordary*] &regions, int[:, ::1] &color_idx, map[int, double*] &hist):
    cdef int i
    cdef int reg_id
    cdef double *pdf
    cdef coordary *coords
    cdef Coordinate *coord
    cdef vector[Coordinate*].iterator cit
    cdef map[int, coordary*].iterator it = regions.begin()
    while it != regions.end():
        reg_id = derefit(it).first
        coords = derefit(it).second
        pdf = <double *>malloc(1728 * sizeof(double))
        for i in range(1728):
            pdf[i] = 0.0
        cit = coords.begin()
        while cit != coords.end():
            coord = derefit(cit)
            pdf[color_idx[coord.y, coord.x]] += 1.0
            incit(cit)
        normalize_dvec(pdf, 1728)
        hist[reg_id] = pdf
        incit(it)


def calc_saliency_score(np.ndarray[np.uint8_t, ndim=3, mode="c"] img,
                        np.ndarray[np.int_t, ndim=2, mode="c"] segment_labels,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] color_dist_mat):
    reduced = np.trunc(img/21.4).astype(np.uint8)
    shape = reduced.shape
    cdef int[:, ::1] color_idx = np.zeros((shape[0], shape[1]), dtype=np.int32)
    cdef map[int, coordary*] regions
    cdef map[int, DoubleCoordinate*] region_centroid
    cdef map[int, double*] region_histgram
    cdef map[int, double] scores
    create_color_idx(reduced, shape[1], shape[0], color_idx)
    label_to_regions(segment_labels.astype(np.int32), regions)
    calc_region_centroid(regions, region_centroid)
    calc_region_histgram(regions, color_idx, region_histgram)

    calc_region_score(img, regions, region_centroid, region_histgram, color_dist_mat, scores)
    ret = {}
    cdef vector[Coordinate*].iterator cvecit
    cdef Coordinate* coor
    cdef double score
    for reg_id, score in scores.items():
        ls = []
        cvecit = regions[reg_id].begin()
        while cvecit != regions[reg_id].end():
            coor = derefit(cvecit)
            ls.append((coor.x, coor.y))
            incit(cvecit)
        ret[score]=ls


    cdef map[int, double*].iterator it2 = region_histgram.begin()
    while it2 != region_histgram.end():
        free(derefit(it2).second)
        incit(it2)

    cdef map[int, DoubleCoordinate*].iterator it3 = region_centroid.begin()
    while it3 != region_centroid.end():
        free(derefit(it3).second)
        incit(it3)
    
    cdef coordary *ary
    cdef map[int, coordary*].iterator it4 = regions.begin()
    while it4 != regions.end():
        ary = derefit(it4).second
        cvecit = ary.begin()
        while cvecit != ary.end():
            free(derefit(cvecit))
            incit(cvecit)
        del ary
        incit(it4)
    return ret
