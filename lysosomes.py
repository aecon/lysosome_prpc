"""
Run with all images in a plate.
Takes as inputs probability files from ilastik.
Computes area of detected lysosomes per image.
"""

import os
import argparse
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import plate as P

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, nargs='+', help="all .tif 2D lysosome probability maps")
parser.add_argument('-p', type=str, required=True, help="plate name")
args = parser.parse_args()


output_file = "output_lysosomes_%s.csv" % args.p
with open(output_file, 'w') as f:
    f.write("%s,%s,%s,%s,%s,%s\n" % ("File", "Well", "Row", "Column", "Area", "AreaFraction"))

areas = np.zeros(len(args.i), dtype=float)
plate_map = np.zeros((P.Nrows,P.Ncolumns), dtype=float)  # deafult for 96-well plate
counts = np.zeros(plate_map.shape, dtype=float)
for i,file in enumerate(args.i):

    # read probability map
    img = skimage.io.imread(file, plugin='tifffile')
    N = img.shape

    # area of lysosomes: number of pixels with probability greater than 0.5
    mask = np.zeros(img.shape, dtype=float)
    mask[img>0.5] = 1
    area = np.sum(mask)

    # save area in array
    areas[i] = area

    # TODO: ADD AREAS OF THE REST OF THE STRUCTURES.. BEST ADD IN A IPYNB FILE...
    

    # get well ID from file name
    well, field = file.split("POM2")[1].split("_")[0].split("-")
    row = well[0]
    col = int(well[1::]) - 1
    rowid = P.well2id(row)
    plate_map[rowid, col] += area
    counts[rowid, col] += 1

    # save area in file
    with open(output_file, 'a') as f:
        f.write("%s,%s,%s,%g,%g,%g\n" % (os.path.basename(file), well, row, col, area, (area/float((N[0]*N[1])))  ))


assert(np.min(counts)>=0)
assert(np.max(counts)<=25)

plate_map = plate_map / (counts + 1.e-7)
percentage_image = plate_map / float((N[0]*N[1]))
plt.imshow(percentage_image)
plt.colorbar()
ax = plt.gca()
ax.set_xticks(np.linspace(0, P.Ncolumns-1, P.Ncolumns))
ax.set_yticks(np.linspace(0, P.Nrows-1, P.Nrows))
ax.set_xticklabels(np.linspace(1, P.Ncolumns, P.Ncolumns, dtype=int))
ax.set_yticklabels(P.row_letters)
ax.xaxis.tick_top()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig("fig_lysosomeArea_%s.png" % args.p)
plt.close()
