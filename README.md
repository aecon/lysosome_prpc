# Image analysis code for the publication "A genetic network coordinating endolysosomal remodeling drives prion catabolism"

## STEP 1. Data preparation

Images are acquired in the .vsi file format, as a multi-channel image with 4 channels per field of view (FOV).  
It is convenient (for the subsequent image processing steps) to split into 4 separate images and save in .tif format.  
This is done with the ImageJ script `split_channels.ijm`.  
Usage: Drag and drop into ImageJ, press Run and follow the directions.  
This will create a new folder at the same level as the .vsi folder, called `<ORIGINAL_FOLDER_NAME>_IJsplit`.


## STEP 2. Image analysis

The image analysis workflow is split into two parts: Segmentation and Colocalization.

Use the existing `HCS-AggreQuant` conda environment.

For the 20x, the nuclei are segmented with the file `nuclei.py`. All other channels are segmented with `ilastik`.  
For the 60x, all channels are segmented with `ilastik`.  
ilastik project files are here: `/media/user/SSD2/Hao`

### 2.1. Image segmentation
Segmentation is performed to detect cell nuclei, cell cytoplasm, lysosomes and PrPc signal.  

#### 20x resolution
* nuclei: segmented using the pre-trained neural netowrk StarDist, and subsequent filters to exclude false positive/negative candidates. 
* cytoplasm: trained a Random Forest Classifier (RFC) in ilastik to perform pixel classification. The pre-trained model Cellpose failed to detect all cells in these images, since the cells have very irregular shape.
* lysosomes: trained a RFC model in ilastik for pixel classification.
* prpc: trained a RFC model in ilastik for pixel classification.

In the cases where a RFC model was trained for pixel classification, the predicted probability maps were thresholded at `p=0.5` to obtain the foreground.


#### 60x resolution
Use file: `main_60x.ipynb`.
* nuclei: trained a RFC model in ilastik for pixel classification. The pre-trained model StarDist gave inconsistent results (applied on plate 1). 
* cytoplasm: trained a RFC model in ilastik for pixel classification.
* lysosomes: trained a RFC model in ilastik for pixel classification.
* prpc: trained a RFC model in ilastik for pixel classification.


### 2.2. Colocalization analysis
Drivers:  
1. `lysosomes.py`  
2. `rdf_multi.py`  
3. `colocalization_prpc_multi.py`  

For 60x, all processing is done inside `main_60x.ipynb`.  


# Contributors

The research was done at the lab of Prof. Adriano Aguzzi, University of Zurich, 2025.  
* Experiment design: Hao Wang
* Training of RFC models: Mariana Moore, Athena Economides
* Image analysis: Athena Economides
