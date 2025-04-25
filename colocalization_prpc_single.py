import os
import glob
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import erosion
import numba

from cells_propagation import segment_propagation_with_pmap



def distance_from_object_edge(mask):
    # replaces each foreground (non-zero) element,
    # with its shortest distance to the background (zeros)
    dist = distance_transform_edt(mask)
    return dist


def get_cell_edges(cell_labels, nuclei_labels):
    cell_surface_width = 10  # !!! in pixels
    Ncells = np.max(cell_labels)
    edges = np.zeros(cell_labels.shape, dtype=float)
    for label in range(1, Ncells+1):
        mask = (cell_labels==label) + (nuclei_labels==label)
        dist = distance_from_object_edge(mask)
#        plt.imshow(dist)
#        plt.show()
        edges[(mask>0) * (dist<cell_surface_width)] = label

#    plt.imshow(edges)
#    plt.title("surface width: %g pixels" % cell_surface_width)
#    plt.show()

    return edges



def process_image(paths):

    fnuc = paths[0]
    fcel = paths[1]
    rcel = paths[2]
    flys = paths[3]
    fprp = paths[4]
    print(paths)

    # load image files
    img_nuc = skimage.io.imread(fnuc, plugin='tifffile') # nuclei labels
    img_cel = skimage.io.imread(fcel, plugin='tifffile').astype(float)
    img_lys = skimage.io.imread(flys, plugin='tifffile').astype(float)
    img_prp = skimage.io.imread(fprp, plugin='tifffile').astype(float)
    raw_cel = skimage.io.imread(rcel, plugin='tifffile')

    # nuclei mask
    mask_nuc = np.zeros(img_nuc.shape, dtype=float)
    mask_nuc[img_nuc>0] = 1
    sumNuclei = np.sum(mask_nuc)

    # cells mask (excludes nucleus)
    mask_cel = np.zeros(img_cel.shape, dtype=float)
    mask_cel[img_cel>0.5] = 1
    mask_cel = skimage.morphology.remove_small_holes(mask_cel.astype(bool, copy=True), area_threshold=400)
    mask_cel[mask_nuc==1] = 0

    # lysosome mask (inside cell or nucleus)
    mask_lys = np.zeros(img_lys.shape, dtype=float)
    mask_lys[img_lys>0.5] = 1
    mask_lys[(mask_nuc+mask_cel)==0] = 0
    #sumLyso = np.sum(mask_lys)

    # prpc mask
    mask_prp = np.zeros(img_prp.shape, dtype=float)
    mask_prp[img_prp>0.5] = 1
    mask_prp[(mask_nuc+mask_cel)==0] = 0
    sumPrPc = np.sum(mask_prp)
    print("sum prpc", sumPrPc, sumPrPc.dtype)

    if sumNuclei>0 and sumPrPc>0:
        # PrPc in Nucleus (DAPI stain)
        colocalization_nuc = (mask_nuc==1) * (mask_prp==1)
        total_prpc_dapi = np.sum(colocalization_nuc)

        # PrPc in CellEdge (layer of XX pixels)
        cell_labels = segment_propagation_with_pmap(raw_cel, img_cel, img_nuc)
        edge = get_cell_edges(cell_labels, img_nuc)
        mask_edge = np.zeros(img_cel.shape, dtype=float)
        mask_edge[(edge>0)*(mask_nuc==0)] = 1
        colocalization_edge = (mask_edge==1) * (mask_prp==1)
        total_prpc_cellsurface = np.sum(colocalization_edge)

        # PrPc in Lysosome
        colocalization_lys = (mask_lys==1) * (mask_prp==1) * (mask_nuc==0)
        total_prpc_lysosome = np.sum(colocalization_lys)

        # PrPc in rest of the cell
        idx = (mask_cel==1) * (mask_edge==0) * (mask_nuc==0) * (mask_lys==0)
        mask_rest_of_cell = np.zeros(img_cel.shape, dtype=float)
        mask_rest_of_cell[idx] = 1
        colocalization_cell = (mask_rest_of_cell>0) * (mask_prp==1)
        total_prpc_cell = np.sum(colocalization_cell)

        plt.figure()
        plt.subplot(1,4,1)
        plt.imshow(mask_nuc + colocalization_nuc)
        plt.title("%.2f" % (total_prpc_dapi / sumPrPc * 100.))
        plt.subplot(1,4,2)
        plt.imshow(mask_edge + colocalization_edge)
        plt.title("%.2f" % (total_prpc_cellsurface / sumPrPc * 100.))
        plt.subplot(1,4,3)
        plt.imshow(mask_lys + colocalization_lys)
        plt.title("%.2f" % (total_prpc_lysosome / sumPrPc * 100.))
        plt.subplot(1,4,4)
        plt.imshow(mask_rest_of_cell + colocalization_cell)
        plt.title("%.2f" % (total_prpc_cell / sumPrPc * 100.))
        plt.show()

        total_prpc = total_prpc_dapi + total_prpc_cellsurface + total_prpc_lysosome + total_prpc_cell
        print(total_prpc_dapi, total_prpc_cellsurface, total_prpc_lysosome, total_prpc_cell)
        print( total_prpc/sumPrPc )
        assert(0)

    else:
        total_prpc_dapi = 0
        total_prpc_cellsurface = 0
        total_prpc_lysosome = 0
        total_prpc_cell = 0


def check_path(path):
    if os.path.isfile(path):
        return path
    else:
        print("File does not exist!", path)
        sys.exit()
   

if __name__ == "__main__":

    plate_name = "plate1"
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

    for paths in zip(files_nuc, files_cel, files_raw_cel, files_lys, files_prp):
        process_image(paths)
        assert(0)

