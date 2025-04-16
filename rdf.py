import os
import glob
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import numba

import plate as P
from cells_propagation import segment_propagation_with_pmap


@numba.jit(nopython=True, parallel=False, fastmath=True)  # race condition with parallel=True
def rdf_lyso(lyso, dist):
    nx,ny = lyso.shape
    L = 300  # max number of pixels away from nucleus edge
    dx = 1
    Nbins = int(L//dx)
    rdf = np.zeros(Nbins)
    for i in numba.prange(nx):
        for j in numba.prange(ny):
            q = int(dist[i,j] // dx)
            if q*dx < L:
                if lyso[i,j]>0:
                    rdf[q] += 1
    return rdf, Nbins, dx


def straight_distance_to_nucleus(mask):
    dist = distance_transform_edt(mask==0)
    return dist



@numba.jit(nopython=True, parallel=False, fastmath=True)
def binary_dilation(mask, kernel):
    kw, kh = kernel.shape
    ph, pw = kh//2, kw//2
    h,w = mask.shape

    padded = np.zeros((h+2*ph, w+2*pw), dtype=np.uint8)
    padded[ph:h+ph, pw:w+pw] = mask

    output = np.zeros_like(mask)
    for i in range(h):
        for j in range(w):
            for m in range(kh):
                for n in range(kw):
                    if kernel[m,n] and padded[i+m, j+n]:
                        output[i,j] = 1
                        break
                if output[i,j] == 1:
                    break
    return output



@numba.jit(nopython=True, parallel=True, fastmath=True)
def parallel_dilate_nuclei(labels, Nlabels, kernel, Ndilations):
    h,w = labels.shape
    stack = np.zeros((Nlabels, h, w), dtype=np.uint8)

    for label in numba.prange(Nlabels):
        mask = (labels == (label+1)).astype(np.uint8)
        for _ in range(Ndilations):
            mask = binary_dilation(mask, kernel)
        stack[label] = mask

    return stack


def distance_to_nucleus_of_specific_cell(nuclei_labels, cell_labels):
    Ncells = np.max(cell_labels)
    distance_map = np.zeros(nuclei_labels.shape, dtype=float)
    for label in range(1, Ncells+1):
        mask = nuclei_labels==label
        dist = straight_distance_to_nucleus(mask)
        dist[cell_labels!=label] = 0
        idx = cell_labels==label
        distance_map[idx] = dist[idx]
    return distance_map




if __name__ == "__main__":

    plate_name = "plate1"
    basedir = "/media/user/SSD2/Mariana/data_50_conditions/20x/%s_IJsplit" % plate_name
    files_lys = sorted(glob.glob("%s/%s_lysosome/probabilities/*_Probabilities_.tif" % (basedir, plate_name)))
    files_cel = sorted(glob.glob("%s/plate1_cells/masks/*_Probabilities.tif" % (basedir)))
    files_raw_cel = sorted(glob.glob("%s/plate1_cells/*CH2.tif" % (basedir)))
    files_nuc = sorted(glob.glob("%s/nuclei_labels/*_labels.tif" % (basedir)))

    Nfiles = len(files_nuc)

    output_file = "output_rdfLysosomes_%s.csv" % plate_name
    with open(output_file, 'w') as f:
        f.write("%s,%s,%s,%s,%s\n" % ("File", "Well", "Row", "Column", "lysosomes RDF"))

    for i in range(Nfiles):
        #if i%20==0:
        print(i, "/", Nfiles)
        print(files_cel[i])
        print(files_nuc[i])
        print(files_lys[i])
        print(files_raw_cel[i])
    
        fnuc = files_nuc[i]
        fcel = files_cel[i]
        flys = files_lys[i]
        rcel = files_raw_cel[i]
    
        # load image files
        img_nuc = skimage.io.imread(fnuc, plugin='tifffile')
        img_cel = skimage.io.imread(fcel, plugin='tifffile').astype(float)
        img_lys = skimage.io.imread(flys, plugin='tifffile')
        raw_cel = skimage.io.imread(rcel, plugin='tifffile')
    
        # lysosome mask
        mask_lys = np.zeros(img_lys.shape, dtype=float)
        mask_lys[img_lys>0.5] = 1
    
        # nuclei mask
        mask_nuc = np.zeros(img_nuc.shape, dtype=float)
        mask_nuc[img_nuc>0] = 1
    
        # cells mask
        mask_cel = np.zeros(img_cel.shape, dtype=float)
        mask_cel[img_cel>0.5] = 1
        mask_cel = skimage.morphology.remove_small_holes(mask_cel.astype(bool, copy=True), area_threshold=400)
    
        # distance field
        #dist_nuc = straight_distance_to_nucleus(mask_nuc)
        # 
        # distance to each cell's nucleus
        cell_labels = segment_propagation_with_pmap(raw_cel, img_cel, img_nuc)
        dist_nuc = distance_to_nucleus_of_specific_cell(img_nuc, cell_labels)

        # keep distance transform only inside cells
        dist_nuc[mask_cel==0] = 0

        # plots
        plt.subplot(1,4,1)
        plt.imshow(img_nuc)
        plt.subplot(1,4,2)
        plt.imshow(cell_labels)
        plt.subplot(1,4,3)
        plt.imshow(img_lys)
        plt.subplot(1,4,4)
        plt.imshow(dist_nuc)
        plt.show()

        # rdf of lysosomes in cells
        rdf, Nbins, dx = rdf_lyso(mask_lys, dist_nuc) 

        plt.plot(np.arange(0,Nbins), rdf)
        plt.xlabel("distance to nucleus (pixels)")
        plt.ylabel("lysosome area (pixels)")
        plt.show()
    
        # get row, column from file name
        well, field = flys.split("POM2")[1].split("_")[0].split("-")
        row = well[0]
        col = int(well[1::]) - 1
    
        # save rdf in file
        with open(output_file, 'a') as f:
            f.write("\"%s\",%s,%s,%g" % (os.path.basename(flys), well, row, col ))
            for i in range(len(rdf)):
                f.write(",%g" % (rdf[i]))
            f.write("\n")


