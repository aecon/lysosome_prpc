# Lysosomal degradation mechanism for PrPC regulation


## Research goal
We selected 50 candidate hits from the PrPC modulator screen. Here we preformed 4 color imaging (DAPI, CellMask, LAMP1-lysosome, POM2-PrPC) to answer two major questions:
1) Do some of these hits regulate lysosomal number?  
2) Do some of these hits regulate the transportation of PrPC into the lysosome?  

We performed 20x single plane imaging (3 replicates per hit gene, 25 fov per replicate). There are three analysis we can do from these images:
1) quantify the area and intensity of lysosomal signal (LAMP1), to analyze if any hit regulate the amount of lysosome in the cell.  
2) quantify PrPC signal (POM2), as a imaging based validation for our screen, that these hit genes can regulate PrPC level.  
3) analyze colocalization of PrPC-cell surface/PrPC-lysosome. With this, we know the percentage of PrPC on the membrane/lysosome/cytosol. And from that, we know if any of the hit genes help with the endocytosis of PrPC / transportation of PrPC into the lysosome.   

For those hits that show difference in c. we can then perform 100x, 3D imaging to verify the effects. 


## Image analysis

The image analysis workflow is split into two parts: segmentation and colocalization.

### Image segmentation
Segmentation is performed to detect cell nuclei, cell cytoplasm, lysosomes and PrPc signal.  
* nuclei: segmented using the pre-trained neural netowrk StarDist, and subsequent filters to exclude false positive/negative candidates. 
* cytoplasm: trained a Random Forest Classifier (RFC) in ilastik to perform pixel classification. The pre-trained model Cellpose failed to detect all cells in these images, since the cells have very irregular shape.
* lysosomes: trained a RFC model in ilastik for pixel classification.
* prpc: trained a RFC model in ilastik for pixel classification.

In the cases where a RFC model was trained for pixel classification, the predicted probability maps were thresholded at `p=0.5` to obtain the foreground.

Cell edges were identified as follows:  
TODO.

### Colocalization
TODO.


# Contributors

The research was done at the lab of Prof. Adriano Aguzzi, University of Zurich, 2025.  
* Experiment design: Hao Wang
* Training of RFC models: Mariana Moore, Athena Economides
* Image analysis: Athena Economides
