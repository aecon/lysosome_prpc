import os
import sys
import glob
import argparse

import numpy as np
import skimage.io
import skimage.filters
import skimage.morphology
from csbdeep.utils import normalize

import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, nargs='+', help="tif 2D image")
args = parser.parse_args()


# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load StarDist model only once
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')


# list of parameters
# - post process: 60x
min_area = 300  # 20x
max_area = 15000  # 20x


def _post_process(labels, img):

    # get number of detected objects
    unique_labels, unique_counts = np.unique(labels, return_counts=True)
    Nlabels = np.shape(unique_labels)[0]
    print("Detected %d labels" % Nlabels)

    # exclude non-cells based on thresholds
    avg_brightness = []
    Nc = 0
    for i in range(1,Nlabels):
        Vol = unique_counts[i]
        idx = (labels==unique_labels[i])

        # remove small objects
        if Vol<min_area:
            labels[idx] = 0
        # remove very large objects
        elif Vol>max_area:
            labels[idx] = 0
        else:
            avg_brightness.append(np.mean(img[idx]))

    # remove very dim objects
    avg_brightness = np.asarray(avg_brightness)
    if len(avg_brightness) < 1:
        return 0, 0

    bmax = np.percentile(avg_brightness, 98)
    bmin = np.percentile(avg_brightness, 2)

    for i in range(1,Nlabels):
        idx = (labels==unique_labels[i])

        bavg = np.mean(img[idx])

        if bavg<bmin:
            labels[idx] = 0
        if bavg>bmax:
            labels[idx] = 0


    # count number of cells after size exlcusion
    Nnuclei = np.shape(np.unique(labels))[0]
    print("After removal of small/large objects, Nlabels=", Nnuclei)

    return Nnuclei, labels



for file in args.i:

    output_folder = os.path.dirname(file) + os.sep + "nuclei_labels"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = output_folder + os.sep + os.path.basename(file) + "_labels.tif"
    if os.path.exists(output_path):
        print(">path exists")
        continue

    print("Processing image:", file)

    # read image
    img = skimage.io.imread(file, plugin='tifffile')
    
    # segment with stardist
    labels, _ = model.predict_instances(normalize(img), predict_kwargs=dict(verbose=False))

    # post-processing
    Nnuclei, labels = _post_process(labels, img)

    if Nnuclei == 0:
        continue

    # save labels
    skimage.io.imsave(output_path, labels, plugin='tifffile', check_contrast=False)
