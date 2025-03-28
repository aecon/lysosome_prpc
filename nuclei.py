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
# - pre process
sigma_denoise = 2
sigma_background = 50
# - post process
min_area = 300
max_area = 15000


def _pre_process(img0):
    """
    Input:
        The original image (I)
    Output:
        The normalized image (In), where:
        * In = G(s=2) / G(s=50), and
        * G(s=a) is the result of the convolution of the original image
          with a Gaussian filter with standard deviation, sigma=a. 
    """
    # takes ~1.2 sec
    img1 = skimage.filters.gaussian(img0, sigma=sigma_denoise)
    back = skimage.filters.gaussian(
        img1, sigma=sigma_background, mode='nearest', preserve_range=True)
    img = img1 / (back + 1.e-7)
    return img


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


#textfile = os.path.dirname(args.i[0]) + os.sep + "counts_%s.txt" % os.path.basename( os.path.dirname(args.i[0]) )
#with open(textfile, 'a') as f:
#    f.write("%20s %15s\n" % ("Image", "Nuclei counts"))


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
    
    # pre-process
    #img = _pre_process(img0)

    # segment with stardist
    labels, _ = model.predict_instances(normalize(img), predict_kwargs=dict(verbose=False))

    # post-processing
    Nnuclei, labels = _post_process(labels, img)

#    # save counts in text file
#    with open(textfile, 'a') as f:
#        f.write("%20s %15d\n" % (os.path.basename(file), Nnuclei))

    if Nnuclei == 0:
        continue

    # save labels
    skimage.io.imsave(output_path, labels, plugin='tifffile', check_contrast=False)


