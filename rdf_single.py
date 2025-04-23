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




#@numba.jit(nopython=True, parallel=False, fastmath=True)
#def binary_dilation(mask, kernel):
#    kw, kh = kernel.shape
#    ph, pw = kh//2, kw//2
#    h,w = mask.shape
#
#    padded = np.zeros((h+2*ph, w+2*pw), dtype=np.uint8)
#    padded[ph:h+ph, pw:w+pw] = mask
#
#    output = np.zeros_like(mask)
#    for i in range(h):
#        for j in range(w):
#            for m in range(kh):
#                for n in range(kw):
#                    if kernel[m,n] and padded[i+m, j+n]:
#                        output[i,j] = 1
#                        break
#                if output[i,j] == 1:
#                    break
#    return output
#
#
#
#@numba.jit(nopython=True, parallel=True, fastmath=True)
#def parallel_dilate_nuclei(labels, Nlabels, kernel, Ndilations):
#    h,w = labels.shape
#    stack = np.zeros((Nlabels, h, w), dtype=np.uint8)
#
#    for label in numba.prange(Nlabels):
#        mask = (labels == (label+1)).astype(np.uint8)
#        for _ in range(Ndilations):
#            mask = binary_dilation(mask, kernel)
#        stack[label] = mask
#
#    return stack


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




if __name__ == "__main__":

    plate_name = "plate1"
    basedir = "/media/user/SSD2/Mariana/data_50_conditions/20x/%s_IJsplit" % plate_name
    files_lys = sorted(glob.glob("%s/lysosome/probabilities/*_Probabilities_.tif" % (basedir)))
    files_cel = sorted(glob.glob("%s/plate1_cells/masks/*_Probabilities.tif" % (basedir)))
    files_raw_cel = sorted(glob.glob("%s/plate1_cells/*CH2.tif" % (basedir)))
    files_nuc = sorted(glob.glob("%s/nuclei_labels/*_labels.tif" % (basedir)))

    Nfiles = len(files_nuc)

#    output_file = "output_rdfLysosomes_%s.csv" % plate_name
#    with open(output_file, 'w') as f:
#        f.write("%s,%s,%s,%s,%s\n" % ("File", "Well", "Row", "Column", "lysosomes RDF"))

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


        # Visualization
        # All together
        fig = plt.figure(figsize=(26,6))

        plt.subplot(1,5,1)
        plt.imshow(np.ma.masked_where(img_nuc==0, img_nuc), vmin=0, vmax=np.max(img_nuc))
        plt.axis('off')

        plt.subplot(1,5,2)
        plt.imshow(raw_cel, vmin=np.percentile(raw_cel,1), vmax=np.percentile(raw_cel,99.9))
        plt.axis('off')

        plt.subplot(1,5,3)
        plt.imshow(np.ma.masked_where(cell_labels==0, cell_labels), vmin=0, vmax=np.max(cell_labels))
        plt.axis('off')

        plt.subplot(1,5,4)
        out = mask_lys*dist_nuc
        import skimage.morphology
        edges = skimage.morphology.dilation(cell_labels)
        edges = skimage.morphology.dilation(edges)
        edges[cell_labels>0] = 0
        edges[edges>0]=1
        out[out==0] = np.nan
        plt.imshow(out, cmap="jet", vmin=0, vmax=200)
        plt.axis('off')
        #plt.imshow(np.ma.masked_where(edges==0, edges), cmap='Greys_r')
        #plt.contour(np.ma.masked_where(cell_labels==0, dist_nuc), 20, colors='k')

        plt.subplot(1,5,5)
        plt.plot(np.arange(0,Nbins), rdf)
        plt.xlabel("distance to nucleus (pixels)")
        plt.ylabel("lysosome area (pixels)")
        plt.grid()

        plt.tight_layout()
        plt.show()



        # Visualization
        # Save separately
        #fig = plt.figure(figsize=(26,6))

        skimage.io.imsave("fig_nuclei_labels_%s_%s.tif" % (plate_name, os.path.basename(fnuc)), img_nuc, plugin='tifffile', check_contrast=False)
        #plt.imshow(np.ma.masked_where(img_nuc==0, img_nuc), vmin=0, vmax=np.max(img_nuc))
        #plt.axis('off')
        #plt.savefig("fig_nuclei_labels_%s_%s.pdf" % (plate_name, os.path.basename(fnuc)))
        #plt.close()

        skimage.io.imsave("fig_cells_raw_%s_%s.tif" % (plate_name, os.path.basename(fnuc)), raw_cel, plugin='tifffile', check_contrast=False)
        #plt.imshow(raw_cel, vmin=np.percentile(raw_cel,1), vmax=np.percentile(raw_cel,99.9))
        #plt.axis('off')
        #plt.savefig("fig_cells_raw_%s_%s.pdf" % (plate_name, os.path.basename(fnuc)))
        #plt.close()

        skimage.io.imsave("fig_cells_labels_%s_%s.tif" % (plate_name, os.path.basename(fnuc)), cell_labels, plugin='tifffile', check_contrast=False)
        #plt.imshow(np.ma.masked_where(cell_labels==0, cell_labels), vmin=0, vmax=np.max(cell_labels))
        #plt.axis('off')
        #plt.savefig("fig_cells_labels_%s_%s.pdf" % (plate_name, os.path.basename(fnuc)))
        #plt.close()

        out = mask_lys*dist_nuc
        skimage.io.imsave("fig_lysosome_distance_%s_%s.tif" % (plate_name, os.path.basename(fnuc)), out, plugin='tifffile', check_contrast=False)
        #import skimage.morphology
        #edges = skimage.morphology.dilation(cell_labels)
        #edges = skimage.morphology.dilation(edges)
        #edges = skimage.morphology.dilation(edges)
        #edges[cell_labels>0] = 0
        #edges[edges>0]=1
        #plt.imshow(np.ma.masked_where(out==0, out), cmap="jet", vmin=0, vmax=200)
        #plt.imshow(np.ma.masked_where(edges==0, edges), cmap='Greys_r')
        #plt.axis('off')
        #plt.savefig("fig_lysosome_distance_%s_%s.pdf" % (plate_name, os.path.basename(fnuc)))
        #plt.close()

        plt.plot(np.arange(0,Nbins), rdf)
        plt.xlabel("distance to nucleus (pixels)")
        plt.ylabel("lysosome area (pixels)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("fig_lysosome_rdf_%s_%s.pdf" % (plate_name, os.path.basename(fnuc)))
        plt.close()



        # save to file
#        # get row, column from file name
#        well, field = flys.split("POM2")[1].split("_")[0].split("-")
#        row = well[0]
#        col = int(well[1::]) - 1
#
#        # save rdf in file
#        with open(output_file, 'a') as f:
#            f.write("\"%s\",%s,%s,%g" % (os.path.basename(flys), well, row, col ))
#            for i in range(len(rdf)):
#                f.write(",%g" % (rdf[i]))
#            f.write("\n")


