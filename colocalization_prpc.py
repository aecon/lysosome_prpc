from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

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
    cell_surface_width = 4  # !!! in pixels
    Ncells = np.max(cell_labels)
    edges = np.zeros(cell_labels.shape, dtype=float)
    for label in range(1, Ncells+1):
        mask = (cell_labels==label) + (nuclei_labels==label)
        plt.imshow(mask)
        plt.show()
        assert(0)
        dist = distance_from_object_edge(mask)
        edges[dist<cell_surface_width] = label
    return edges



def process_image(paths, shared_list, lock):

        edge_width = 3  # pixels

        fnuc = paths[0]
        fcel = paths[1]
        rcel = paths[2]
        flys = paths[3]
        fprp = paths[4]

        # load image files
        img_nuc = skimage.io.imread(fnuc, plugin='tifffile') # nuclei labels
        img_cel = skimage.io.imread(fcel, plugin='tifffile').astype(float)
        img_lys = skimage.io.imread(flys, plugin='tifffile').astype(float)
        img_prp = skimage.io.imread(fprp, plugin='tifffile').astype(float)
        raw_cel = skimage.io.imread(rcel, plugin='tifffile')

        # lysosome mask
        mask_lys = np.zeros(img_lys.shape, dtype=float)
        mask_lys[img_lys>0.5] = 1
        #sumLyso = np.sum(mask_lys)

        # prpc mask
        mask_prp = np.zeros(img_prp.shape, dtype=float)
        mask_prp[img_prp>0.5] = 1
        sumPrPc = np.sum(mask_prp)

        # nuclei mask
        mask_nuc = np.zeros(img_nuc.shape, dtype=float)
        mask_nuc[img_nuc>0] = 1
        sumNuclei = np.sum(mask_nuc)

        # cells mask
        mask_cel = np.zeros(img_cel.shape, dtype=float)
        mask_cel[img_cel>0.5] = 1
        mask_cel = skimage.morphology.remove_small_holes(mask_cel.astype(bool, copy=True), area_threshold=400)

        if sumNuclei>0 and sumPrPc>0:
            # PrPc in Nucleus (DAPI stain)
            total_prpc_dapi = np.sum((mask_nuc==1) * (mask_prp==1))

            # PrPc in CellSurface (layer of XX pixels)
            cell_labels = segment_propagation_with_pmap(raw_cel, img_cel, img_nuc)
            edge = get_cell_edges(cell_labels, img_nuc)
            total_prpc_cellsurface = np.sum((edge>0) * (mask_prp==1))

            # PrPc in Lysosome
            total_prpc_lysosome = np.sum((mask_lys==1) * (mask_prp==1))

            # PrPc in rest of the cell
            idx = (mask_cel>0) * (edge==0) * (mask_nuc==0) * (mask_lys==0)
            total_prpc_cell = np.sum(mask_prp[idx]==1)

            total_prpc = total_prpc_dapi + total_prpc_cellsurface + total_prpc_lysosome + total_prpc_cell
            print(total_prpc, sumPrPc)
            assert(0)

        else:
            total_prpc_dapi = 0
            total_prpc_cellsurface = 0
            total_prpc_lysosome = 0
            total_prpc_cell = 0
            

        # get row, column from file name
        well, field = flys.split("POM2")[1].split("_")[0].split("-")
        row = well[0]
        col = int(well[1::])

        # number of nuclei
        if sumNuclei>0:
            UniqueNucLabels = np.unique(img_nuc)
            Nnuclei = len(UniqueNucLabels) - 1  # remove background label
        else:
            Nnuclei = 0

        result = {
            "File": flys,
            "Well": well,
            "Row": row,
            "Column": col,
            "CellArea": cellArea,
            "NucleiArea": nucleiArea,
            "Ncells": Nnuclei,
            "PrPc_in_Lysosome": total_prpc_lysosome,
            "PrPc_in_CellSurface": total_prpc_cellsurface,
            "PrPc_in_DAPI": total_prpc_dapi,
            "PrPc_in_Cell": total_prpc_cell
        }

        with lock:
            shared_list.append(result)


if __name__ == "__main__":

    max_workers = 1

    plate_name = "plate1"
    basedir = "/media/user/SSD2/Mariana/data_50_conditions/20x/%s_IJsplit" % plate_name
    files_lys = sorted(glob.glob("%s/lysosome/probabilities/*_Probabilities_.tif" % basedir ))
    files_prp = sorted(glob.glob("%s/prpc/probabilities/*_Probabilities.tif" % basedir ))
    files_cel = sorted(glob.glob("%s/cells/masks/*_Probabilities.tif" % basedir ))
    files_raw_cel = sorted(glob.glob("%s/cells/*CH2.tif" % basedir ))
    files_nuc = sorted(glob.glob("%s/nuclei_labels/*_labels.tif" % (basedir) ))

    Nfiles = len(files_nuc)
    print("Found %d files" % Nfiles)

    output_file = "output_PrPc_localization_%s.csv" % plate_name

    with Manager() as manager:
        shared_list = manager.list()
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image, paths, shared_list, lock)
                for paths in zip(files_nuc, files_cel, files_raw_cel, files_lys, files_prp)
            ]
            for i,future in enumerate(as_completed(futures), 1):
                future.result()
                if i%10==0 or i==Nfiles:
                    print(f"Processed {i}/{Nfiles} images")

        header = ["File", "Well", "Row", "Column",
                    "CellArea", "NucleiArea",
                    "Ncells",
                    "PrPc_in_Lysosome", "PrPc_in_CellSurface", "PrPc_in_DAPI", "PrPc_in_Cell"]

        with open(output_file, 'w', newline='\n') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in shared_list:
                writer.writerow(row)

