import os
import glob
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import plate as P


plate_name = "plate1"
basedir = "/media/user/SSD2/Mariana/data_50_conditions/20x/%s_IJsplit" % plate_name
files_lys = sorted(glob.glob("%s/%s_lysosome/probabilities/*_Probabilities_.tif" % (basedir, plate_name)))
files_cel = sorted(glob.glob("%s/plate1_cells/masks/*_Probabilities.tif" % (basedir)))
files_nuc = sorted(glob.glob("%s/nuclei_labels/*_labels.tif" % (basedir)))

Nfiles = len(files_nuc)


output_file = "output_LysosomesInCells_%s.csv" % plate_name
with open(output_file, 'w') as f:
    f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % ("File", "Well", "Row", "Column", "LysoArea", "CellArea", "AreaFraction", "NumberNuclei"))

plate_map = np.zeros((P.Nrows,P.Ncolumns), dtype=float)
plate_map_Nnuclei = np.zeros((P.Nrows,P.Ncolumns), dtype=float)
counts = np.zeros(plate_map.shape, dtype=float)

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

    # area of lysosomes: number of pixels with probability greater than 0.5
    mask = np.zeros(img_lys.shape, dtype=float)
    mask[img_lys>0.5] = 1
    area_lys = np.sum(mask)

    # number of nuclei
    UniqueLabels = np.unique(img_nuc)
    Nnuclei = len(UniqueLabels) - 1  # remove background label

    # area of cells
    mask = np.zeros(img_cel.shape, dtype=float)
    mask[img_cel>0.5] = 1
    area_cel = np.sum(mask)

    # ratio of area lysosome over area cells
    ratio = area_lys / (area_cel + 1.e-14)
    if area_cel < (250.*250.):
        ratio = 0.

    # get well ID from file name
    well, field = flys.split("POM2")[1].split("_")[0].split("-")
    #print(well, field)
    row = well[0]
    col = int(well[1::]) - 1
    rowid = P.well2id(row)
    plate_map[rowid, col] += ratio
    counts[rowid, col] += 1
    plate_map_Nnuclei[rowid, col] += Nnuclei

    # save area in file
    with open(output_file, 'a') as f:
        f.write("\"%s\",%s,%s,%g,%g,%g,%g,%g\n" % (os.path.basename(flys), well, row, col, area_lys, area_cel, ratio, Nnuclei ))


assert(np.min(counts)>=0)
assert(np.max(counts)<=25)

fig = plt.figure(figsize=(16,5))

plate_map = plate_map / (counts + 1.e-14)
plt.subplot(1,2,1)
plt.imshow(plate_map)
plt.colorbar()
ax = plt.gca()
ax.set_xticks(np.linspace(0, P.Ncolumns-1, P.Ncolumns))
ax.set_yticks(np.linspace(0, P.Nrows-1, P.Nrows))
ax.set_xticklabels(np.linspace(1, P.Ncolumns, P.Ncolumns, dtype=int))
ax.set_yticklabels(P.row_letters)
ax.xaxis.tick_top()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Area ratio")
plt.tight_layout()

plt.subplot(1,2,2)
plt.imshow(plate_map_Nnuclei)
plt.colorbar()
ax = plt.gca()
ax.set_xticks(np.linspace(0, P.Ncolumns-1, P.Ncolumns))
ax.set_yticks(np.linspace(0, P.Nrows-1, P.Nrows))
ax.set_xticklabels(np.linspace(1, P.Ncolumns, P.Ncolumns, dtype=int))
ax.set_yticklabels(P.row_letters)
ax.xaxis.tick_top()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Number of nuclei")
plt.tight_layout()

#plt.show()
plt.savefig("fig2_lysosomeAreaOverCells_%s.png" % plate_name)
plt.close()
