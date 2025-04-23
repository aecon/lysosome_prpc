import os
import sys
import numpy as np
import argparse
import skimage.io
import skimage.filters
import skimage.morphology
from skimage import restoration
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import watershed


# Segmentation parameters
THRESHOLD_NORMALIZED = 1.04  # used 1.20 for validation figures
MIN_CELL_INTENSITY = 200
FOREGROUND_SIGMA = 2
BACKGROUND_SIGMA = 50
MAX_INTENSITY_PERCENTILE = 99.8
MAX_DISTANCE_FROM_NUCLEI = 400
CLAHE_FIELD_BLUR_SIGMA = 6
CLAHE_KERNEL = 150
MAX_HOLE_AREA = 400

def rescale01(img):
    i = np.asfarray(img)
    l = np.nanmin(i)
    h = np.nanmax(i)
    o = (i-l)/(h-l)
    return o


def segment_propagation_with_pmap(raw_cells, pmap_cells, nuclei_labels):

    # cell mask from probability map
    nuclei_mask = np.zeros(nuclei_labels.shape, dtype=float)
    nuclei_mask[nuclei_labels>0] = 1

    mask_cells = np.zeros(pmap_cells.shape, dtype=float)
    mask_cells[pmap_cells>0.5] = 1
    mask_cells = skimage.morphology.remove_small_holes(mask_cells.astype(bool, copy=True), area_threshold=MAX_HOLE_AREA)

    # float raw cells
    raw_cells = np.asfarray(raw_cells, float)  # float is necessary for next steps

    # Background normalization
    denoised = skimage.filters.gaussian(raw_cells, sigma=FOREGROUND_SIGMA)
    cap_threshold = np.percentile(raw_cells, MAX_INTENSITY_PERCENTILE)
    img_cap = raw_cells.copy()
    img_cap[raw_cells>cap_threshold] = cap_threshold
    background = skimage.filters.gaussian(img_cap, sigma=BACKGROUND_SIGMA)
    normalized = denoised / background
    normalized_01 = rescale01(normalized)

    # Adaptive histogram equalization 
    clahe = skimage.exposure.equalize_adapthist(normalized_01, kernel_size=CLAHE_KERNEL)

    # intensity field
    intensity_field = rescale01( skimage.filters.gaussian(clahe, sigma=CLAHE_FIELD_BLUR_SIGMA) )
    inverted_01_intensity_field = 1.0 - intensity_field
    inverted_01_intensity_field[nuclei_mask==1] = 0

    # distance field
    assert(np.min(nuclei_mask)==0 and np.max(nuclei_mask)<=1)
    distances = ndimage.distance_transform_edt(1.0-nuclei_mask)  # float64
    distances[distances>=MAX_DISTANCE_FROM_NUCLEI] = 0
    distances01 = rescale01(distances)

    # propagation field for watershed : weighted average of distance and intensity
    field = np.zeros(distances.shape, dtype=float)
    field[:,:] = inverted_01_intensity_field * pow(distances01, 0.5)

    # watershed of distance map
    labels = watershed(field, markers=nuclei_labels, mask=mask_cells, watershed_line=False)
    labels[nuclei_mask>0] = 0

#    # plots
#    plt.subplot(1,4,1)
#    plt.imshow(raw_cells, vmin=np.percentile(raw_cells,5), vmax=np.percentile(raw_cells, 99.98))
#    plt.subplot(1,4,2)
#    plt.imshow(nuclei_labels)
#    plt.subplot(1,4,3)
#    plt.imshow(field, cmap='viridis_r')
#    plt.subplot(1,4,4)
#    plt.imshow(labels)
#    plt.show()

    return labels

