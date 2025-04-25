from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

import os
import glob
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import numba

import plate as P
from cells_propagation import segment_propagation_with_pmap



def distance_from_object_edge(mask):
    dist = distance_transform_edt(mask==0)
    return dist


def distance_to_nucleus_of_specific_cell(nuclei_labels, cell_labels):
    Ncells = np.max(cell_labels)
    distance_map = np.zeros(nuclei_labels.shape, dtype=float)
    for label in range(1, Ncells+1):
        mask = nuclei_labels==label
        dist = distance_from_object_edge(mask)
        dist[cell_labels!=label] = 0
        idx = cell_labels==label
        distance_map[idx] = dist[idx]
    return distance_map


@numba.jit(nopython=True, parallel=False, fastmath=True)  # race condition with parallel=True
def get_rdf(lyso, dist):
    nx,ny = lyso.shape
    L = 300  # max number of pixels away from nucleus edge
    dx = 2
    Nbins = int(L//dx)
    rdf = np.zeros(Nbins)
    for i in numba.prange(nx):
        for j in numba.prange(ny):
            q = int(dist[i,j] // dx)
            if q*dx < L:
                if lyso[i,j]>0:
                    rdf[q] += 1
    return rdf, Nbins, dx



def process_image(paths, shared_list, lock):

        fnuc = paths[0]
        fcel = paths[1]
        rcel = paths[2]
        flys = paths[3]

        # load image files
        img_nuc = skimage.io.imread(fnuc, plugin='tifffile')
        img_cel = skimage.io.imread(fcel, plugin='tifffile').astype(float)
        img_lys = skimage.io.imread(flys, plugin='tifffile')
        raw_cel = skimage.io.imread(rcel, plugin='tifffile')

        # lysosome mask
        mask_lys = np.zeros(img_lys.shape, dtype=float)
        mask_lys[img_lys>0.5] = 1
        sumLyso = np.sum(mask_lys)

        # nuclei mask
        mask_nuc = np.zeros(img_nuc.shape, dtype=float)
        mask_nuc[img_nuc>0] = 1
        sumNuclei = np.sum(mask_nuc)

        # cells mask
        mask_cel = np.zeros(img_cel.shape, dtype=float)
        mask_cel[img_cel>0.5] = 1
        mask_cel = skimage.morphology.remove_small_holes(mask_cel.astype(bool, copy=True), area_threshold=400)

        if sumNuclei>0 and sumLyso>0:
            # distance to each cell's nucleus
            cell_labels = segment_propagation_with_pmap(raw_cel, img_cel, img_nuc)
            cellArea = np.sum(cell_labels>0)
            dist_nuc = distance_to_nucleus_of_specific_cell(img_nuc, cell_labels)

            # keep distance transform only inside cells
            dist_nuc[mask_cel==0] = 0

            # rdf of lysosomes in cells
            rdf, Nbins, dx = get_rdf(mask_lys, dist_nuc) 
        else:
            rdf = np.zeros(150)
            cellArea = np.sum(mask_cel)

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
            "Ncells": Nnuclei,
            "LysosomeRDF": ",".join(map(str,rdf))}

        with lock:
            shared_list.append(result)


def check_path(path):
    if os.path.isfile(path):
        return path
    else:
        print("File does not exist!", path)
        sys.exit()
   

if __name__ == "__main__":

    max_workers = 100

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
    assert(len(files_lys) == Nfiles)
    assert(len(files_prp) == Nfiles)
    assert(len(files_cel) == Nfiles)
    assert(len(files_raw_cel) == Nfiles)
    print("Found %d files" % Nfiles)

    output_file = "output_rdfLysosomes_%s.csv" % plate_name

    with Manager() as manager:
        shared_list = manager.list()
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image, paths, shared_list, lock)
                for paths in zip(files_nuc, files_cel, files_raw_cel, files_lys)
            ]
            for i,future in enumerate(as_completed(futures), 1):
                future.result()
                if i%10==0 or i==Nfiles:
                    print(f"Processed {i}/{Nfiles} images")

        header = ["File", "Well", "Row", "Column", "CellArea", "Ncells", "LysosomeRDF"]
        with open(output_file, 'w', newline='\n') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in shared_list:
                writer.writerow(row)

