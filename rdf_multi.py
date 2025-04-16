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

        # nuclei mask
        mask_nuc = np.zeros(img_nuc.shape, dtype=float)
        mask_nuc[img_nuc>0] = 1

        # cells mask
        mask_cel = np.zeros(img_cel.shape, dtype=float)
        mask_cel[img_cel>0.5] = 1
        mask_cel = skimage.morphology.remove_small_holes(mask_cel.astype(bool, copy=True), area_threshold=400)

        # distance field
        #dist_nuc = distance_from_object_edge(mask_nuc)
        # 
        # distance to each cell's nucleus
        cell_labels = segment_propagation_with_pmap(raw_cel, img_cel, img_nuc)
        dist_nuc = distance_to_nucleus_of_specific_cell(img_nuc, cell_labels)

        # keep distance transform only inside cells
        dist_nuc[mask_cel==0] = 0

#        # plots
#        plt.subplot(1,4,1)
#        plt.imshow(img_nuc)
#        plt.subplot(1,4,2)
#        plt.imshow(cell_labels)
#        plt.subplot(1,4,3)
#        plt.imshow(img_lys)
#        plt.subplot(1,4,4)
#        plt.imshow(dist_nuc)
#        plt.show()

        # rdf of lysosomes in cells
        rdf, Nbins, dx = get_rdf(mask_lys, dist_nuc) 

#        plt.plot(np.arange(0,Nbins), rdf)
#        plt.xlabel("distance to nucleus (pixels)")
#        plt.ylabel("lysosome area (pixels)")
#        plt.show()

        # get row, column from file name
        well, field = flys.split("POM2")[1].split("_")[0].split("-")
        row = well[0]
        col = int(well[1::])

#        # save rdf in file
#        with open(output_file, 'a') as f:
#            f.write("\"%s\",%s,%s,%g" % (os.path.basename(flys), well, row, col ))
#            for r in range(len(rdf)):
#                f.write(",%g" % (rdf[r]))
#            f.write("\n")

        # number of nuclei
        UniqueNucLabels = np.unique(img_nuc)
        Nnuclei = len(UniqueNucLabels) - 1  # remove background label

        CellArea = np.sum(cell_labels>0)

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


if __name__ == "__main__":

    max_workers = 60

    plate_name = "plate1"
    basedir = "/media/user/SSD2/Mariana/data_50_conditions/20x/%s_IJsplit" % plate_name
    files_lys = sorted(glob.glob("%s/%s_lysosome/probabilities/*_Probabilities_.tif" % (basedir, plate_name)))
    files_cel = sorted(glob.glob("%s/plate1_cells/masks/*_Probabilities.tif" % (basedir)))
    files_raw_cel = sorted(glob.glob("%s/plate1_cells/*CH2.tif" % (basedir)))
    files_nuc = sorted(glob.glob("%s/nuclei_labels/*_labels.tif" % (basedir)))

    Nfiles = len(files_nuc)
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

