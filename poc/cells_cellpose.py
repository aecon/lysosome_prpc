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
from cellpose.models import Cellpose
import tensorflow as tf

from nuclei import min_area
min_nucleus_area = min_area

nucleus_area_overlap = 0.8


def exclude_cells_without_nucleus(labels, seeds):
    AllLabels = np.unique(labels[labels>0])
    for l in AllLabels:
        idx = labels==l
        is_seed = np.sum(seeds[idx])
        if is_seed < nucleus_area_overlap * min_nucleus_area:
            labels[idx] = 0
            print("No seed for label", l)
    return labels


def segment_cellpose(image_file, output_files_cells, output_files_nuclei, model):
    """
    Use the pre-trained Deep Neural Network cellpose for segmentation of cells. 

    Model:
        cyto2: For best accuracy and runtime performance, resize images so cells are less than 100 pixels across.

    Settings:
        Using default settings. For more info see:
        https://cellpose.readthedocs.io/en/latest/settings.html

    Additional info:
        Explanation of 'channel' parameter:
        https://forum.image.sc/t/about-the-correct-meaning-of-channel-in-cellpose-2-2/79671/7
    """

    # cellpose parameters
    channels = [1,2]

    # load cell image
    img0 = skimage.io.imread(image_file, plugin='tifffile')

    # load nuclei labels
    allnuclei = skimage.io.imread(output_files_nuclei, plugin='tifffile')
    allnuclei_mask = np.zeros(np.shape(allnuclei))
    allnuclei_mask[allnuclei>0] = 1

    # pixels per direction
    n = np.shape(img0)[0]

    # input to cellpose: a two-channel image [cells, nuclei]
    input_to_cellpose = np.zeros((2,n,n))
    input_to_cellpose[0,:,:] = img0[:,:]
    input_to_cellpose[1,:,:] = allnuclei_mask[:,:]  # using nuclei masks gives MUCH better cell segmentation!

    # run cellpose
    masks, flows, styles, diams = model.eval(input_to_cellpose, diameter=None, channels=channels, resample=True,
                                             flow_threshold=0.8, cellprob_threshold=0., do_3D=False)

    # remove cells that don't contain nucleus
    labels2 = exclude_cells_without_nucleus(masks, allnuclei_mask)

    skimage.io.imsave(output_files_cells, labels2, plugin='tifffile', check_contrast=False)



parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, nargs='+', help="tif 2D image")
args = parser.parse_args()

# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load CellPose model
cellpose_model = Cellpose(gpu=True, model_type='cyto2')

for file in args.i:

    output_folder = os.path.dirname(file) + os.sep + "cell_labels"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_labels_nuclei = os.path.dirname(file) + os.sep + "nuclei_labels" + os.sep + os.path.basename(file).split("CH2.tif")[0] + "CH1.tif_labels.tif"
    if os.path.exists(file_labels_nuclei)==False:
        print("File does not exist!")
        print(file_labels_nuclei)
        sys.exit()

    output_path = output_folder + os.sep + os.path.basename(file) + "_labels.tif"

    segment_cellpose(file, output_path, file_labels_nuclei, cellpose_model)

