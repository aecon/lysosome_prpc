import os
import glob
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import numba

import plate as P


plate_name = "plate1"
basedir = "/media/user/SSD2/Mariana/data_50_conditions/20x/%s_IJsplit" % plate_name
files_lys = sorted(glob.glob("%s/%s_lysosome/probabilities/*_Probabilities_.tif" % (basedir, plate_name)))
files_cel = sorted(glob.glob("%s/plate1_cells/masks/*_Probabilities.tif" % (basedir)))
files_nuc = sorted(glob.glob("%s/nuclei_labels/*_labels.tif" % (basedir)))

Nfiles = len(files_nuc)


output_file = "output_rdfLysosomes_%s.csv" % plate_name
with open(output_file, 'w') as f:
    f.write("%s,%s,%s,%s,%s\n" % ("File", "Well", "Row", "Column", "lysosomes RDF"))


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


for i in range(Nfiles):
    if i%20==0:
        print(i, "/", Nfiles)

    fnuc = files_nuc[i]
    fcel = files_cel[i]
    flys = files_lys[i]

    # load image files
    img_nuc = skimage.io.imread(fnuc, plugin='tifffile')
    img_cel = skimage.io.imread(fcel, plugin='tifffile').astype(float)
    img_lys = skimage.io.imread(flys, plugin='tifffile')

    # lysosome mask
    mask_lys = np.zeros(img_lys.shape, dtype=float)
    mask_lys[img_lys>0.5] = 1

    # nuclei mask
    mask_nuc = np.zeros(img_nuc.shape, dtype=float)
    mask_nuc[img_nuc>0] = 1

    # cells mask
    mask_cel = np.zeros(img_cel.shape, dtype=float)
    mask_cel[img_cel>0.5] = 1

    # nuclei distance transform
    dist_nuc = distance_transform_edt(mask_nuc==0)

    # keep distance transform only inside cells
    dist_nuc[mask_cel==0] = 0

#    # plots
#    plt.subplot(1,4,1)
#    plt.imshow(img_nuc)
#    plt.subplot(1,4,2)
#    plt.imshow(img_cel)
#    plt.subplot(1,4,3)
#    plt.imshow(img_lys)
#    plt.subplot(1,4,4)
#    #
#    dist_nuc2 = np.zeros(dist_nuc.shape, dtype=float)
#    dist_nuc2[:,:] = dist_nuc[:,:]
#    dist_nuc2[dist_nuc2==0] = np.nan
#    plt.imshow(dist_nuc2)
#    plt.show()

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


