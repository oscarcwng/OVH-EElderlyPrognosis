# Convolutional neural network algorithms and hematoxylin-and-eosin–stained images predict clinical outcomes in high-grade serous ovarian cancer patients with advanced age and stage
Chun Wai Ng1, Kwong-Kwok Wong1, Berrett C. Lawson1, Sammy Ferri-Borgogno1, Samuel C. Mok1*

1 Department of Gynecologic Oncology and Reproductive Medicine, The University of Texas MD Anderson Cancer Center, Houston, TX 77030, USA


1. Download TCGA-OV tissue images
2. Download TCGA-OV clinical data from GDC and TCGA-OV Pan-Cancer survival data from cBioportal
3. Pre-process the images
4. Model training with 5-fold cross-validation and evaluation


# 1. Download TCGA-OV tissue images

Download TCGA-OV tissue images using gdc data transfer tool (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) with the manifest file gdc_manifest_tissue_images.txt.

# 2. Download TCGA-OV clinical data from GDC and TCGA-OV Pan-Cancer survival data from cBioportal

cBioporal: https://www.cbioportal.org/study/summary?id=ov_tcga_pan_can_atlas_2018

GDC data portal: download the file with the maifest file gdc_manifest_clinical.txt using gdc data transfer tool

# 3. Pre-process the images

The code is in the Jupyter notebook 3.Image pre-processing.ipynb

# 4. Model training with 5-fold cross-validation with TCGA training images and evaluation with TCGA testing images

The code used to generate the results for TCGA data is in the Jupyter notebook 4.Training.ipynb
