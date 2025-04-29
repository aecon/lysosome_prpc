import os
import glob
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import erosion
import numba

from cells_propagation import segment_propagation_with_pmap


def exclude_cells_without_nucleus(cell_labels, nuclei_labels):
    Ncells = np.max(cell_labels)
    for label in range(1, Ncells+1):
        idx_cells  = (cell_labels==label)
        idx_nuclei = (nuclei_labels==label)
        if np.sum(idx_nuclei)==0:
            cell_labels[idx_cells] = 0
    return cell_labels


def distance_from_object_edge(mask):
    # replaces each foreground (non-zero) element,
    # with its shortest distance to the background (zeros)
    dist = distance_transform_edt(mask)
    return dist


def get_cell_edges(cell_labels, nuclei_labels):
    cell_edge_width = 8  # !!! in pixels
    Ncells = np.max(cell_labels)
    edges = np.zeros(cell_labels.shape, dtype=float)
    for label in range(1, Ncells+1):
        mask = (cell_labels==label) + (nuclei_labels==label)
        dist = distance_from_object_edge(mask)
#        plt.imshow(dist)
#        plt.show()
        edges[(mask>0) * (dist<cell_edge_width)] = label

#    plt.imshow(edges)
#    plt.title("surface width: %g pixels" % cell_edge_width)
#    plt.show()

    return edges



def init_mask(x):
    return np.zeros(x.shape, dtype=float)


def process_image(paths):

    fnuc = paths[0]
    fcel = paths[1]
    rcel = paths[2]
    flys = paths[3]
    fprp = paths[4]
    #print(paths)

    # load image files
    img_nuc = skimage.io.imread(fnuc, plugin='tifffile') # nuclei labels
    img_cel = skimage.io.imread(fcel, plugin='tifffile').astype(float)  # probability map from ilastik
    img_lys = skimage.io.imread(flys, plugin='tifffile').astype(float)  # probability map from ilastik
    img_prp = skimage.io.imread(fprp, plugin='tifffile').astype(float)  # probability map from ilastik
    raw_cel = skimage.io.imread(rcel, plugin='tifffile')

    # nuclei mask
    mask_nuc = init_mask(img_nuc)
    mask_nuc[img_nuc>0] = 1
    sumNuclei = np.sum(mask_nuc)

    if sumNuclei>0:

        # MASKS

        # cells mask (excludes nucleus)
        mask_cel = init_mask(img_nuc)
        mask_cel[img_cel>0.5] = 1
        mask_cel = skimage.morphology.remove_small_holes(mask_cel.astype(bool, copy=True), area_threshold=400)
        mask_cel[mask_nuc==1] = 0

        # estimate CellEdge (a layer of XX pixels)
        cell_labels = segment_propagation_with_pmap(raw_cel, img_cel, img_nuc)
        cell_labels = exclude_cells_without_nucleus(cell_labels, img_nuc)
        mask_cel[cell_labels==0] = 0  # update cell mask
        edge = get_cell_edges(cell_labels, img_nuc)
        mask_edge = init_mask(img_nuc)
        mask_edge[(edge>0)] = 1

        # cell cytoplasm
        mask_cytoplasm = init_mask(img_nuc)
        mask_cytoplasm[(mask_cel==1) * (mask_edge==0)] = 1

        # mask of both nucleus and cells
        mask_nucleus_and_cells = init_mask(img_nuc)
        mask_nucleus_and_cells[(mask_nuc+mask_cel)>0] = 1

        # lysosome mask (inside cell or nucleus)
        mask_lys = init_mask(img_nuc)
        mask_lys[img_lys>0.5] = 1
        mask_lys[mask_nucleus_and_cells==0] = 0  # ignore signal outside cells

        # prpc mask (inside cell or nucleus)
        mask_prp = init_mask(img_nuc)
        mask_prp[img_prp>0.5] = 1
        mask_prp[mask_nucleus_and_cells==0] = 0  # ignore signal outside cells
        sumPrPc = np.sum(mask_prp)

        
        # ALL STRUCTURES TOGETHER

        # map of structures
        # 0: empty
        # 1: nucleus
        # 2: cytoplasm
        # 3: cell edge
        # 4: lysosome
        structures_map = init_mask(img_nuc)

        structures_map[mask_nuc==1] = 1
        structures_map[mask_cytoplasm==1] = 2
        structures_map[mask_lys==1] = 4
        structures_map[mask_edge==1] = 3
        structures_map[mask_nucleus_and_cells==0] = 0

        #plt.imshow(structures_map)
        #plt.show()
        #
        #output_array = np.zeros((2, img_nuc.shape[0], img_nuc.shape[1]))
        #output_array[0,:,:] = structures_map[:,:]
        #output_array[1,:,:] = mask_prp[:,:]
        #skimage.io.imsave("overlay_structures_%s.tif" % os.path.basename(fprp), output_array, plugin='tifffile', check_contrast=False)
    
        total_prpc_nucleus   = np.sum( (mask_prp==1) * (structures_map==1) )
        total_prpc_cytoplasm = np.sum( (mask_prp==1) * (structures_map==2) )
        total_prpc_celledge  = np.sum( (mask_prp==1) * (structures_map==3) )
        total_prpc_lysosome  = np.sum( (mask_prp==1) * (structures_map==4) )

        total_prpc = total_prpc_nucleus + total_prpc_celledge + total_prpc_lysosome + total_prpc_cytoplasm
        #print("Total prpc inside entire cells:", sumPrPc, sumPrPc.dtype)
        #print(total_prpc_nucleus, total_prpc_celledge, total_prpc_lysosome, total_prpc_cytoplasm)
        #print("Sum of prpc  / Total prpc:", total_prpc/sumPrPc )
        assert(total_prpc == sumPrPc)

    else:
        total_prpc_nucleus = 0
        total_prpc_cytoplasm = 0
        total_prpc_celledge = 0
        total_prpc_lysosome = 0

    # number of nuclei and cell area
    if sumNuclei>0:
        UniqueNucLabels = np.unique(img_nuc)
        Nnuclei = len(UniqueNucLabels) - 1  # remove background label
        cellArea = np.sum(mask_nucleus_and_cells)
    else:
        Nnuclei = 0
        cellArea = 0

    return total_prpc_nucleus, total_prpc_cytoplasm, total_prpc_celledge, total_prpc_lysosome, cellArea, sumNuclei, Nnuclei


def check_path(path):
    if os.path.isfile(path):
        return path
    else:
        print("File does not exist!", path)
        sys.exit()
   

def get_path_lists(plate_name):
    basedir = "/media/user/SSD2/Mariana/PRPC/data_50_conditions/20x/%s_IJsplit" % plate_name
    files_nuc = sorted(glob.glob("%s/CH1_nuclei/nuclei_labels/*.tif" % basedir ))
    files_cel = []
    files_raw_cel = []
    files_lys = []
    files_prp = []

    for f in files_nuc:
        b0 = os.path.basename(f)
        b = b0.split("_labels.tif")[0]
        
        fcel = basedir + "/CH2_cells/probabilities_cells/" + b.split("CH1.tif")[0] + "CH2_Probabilities.tif"
        files_cel.append(check_path(fcel))

        fraw = basedir + "/CH2_cells/" + b.split("CH1.tif")[0] + "CH2.tif"
        files_raw_cel.append(check_path(fraw))

        flys = basedir + "/CH3_lysosome/probabilities/" + b.split("CH1.tif")[0] + "_Probabilities_.tif"
        files_lys.append(check_path(flys))

        fprp = basedir + "/CH4_prpc/probabilities/" + b.split("CH1.tif")[0] + "CH4_Probabilities.tif"
        files_prp.append(check_path(fprp))

    Nfiles = len(files_nuc)
    print("Found %d files" % Nfiles, len(files_lys))
    assert(len(files_lys) == Nfiles)
    assert(len(files_prp) == Nfiles)
    assert(len(files_cel) == Nfiles)
    assert(len(files_raw_cel) == Nfiles)

    return files_nuc, files_cel, files_raw_cel, files_lys, files_prp



if __name__ == "__main__":

    plate_name = "plate1"
    files_nuc, files_cel, files_raw_cel, files_lys, files_prp = get_path_lists(plate_name)


    for paths in zip(files_nuc, files_cel, files_raw_cel, files_lys, files_prp):
        print(paths)
        result = process_image(paths)

