# 3D-SEM-Segmentation

## Overview
This project provides the image datasets and codes used in the paper “Deep
learning for three-dimensional segmentation of electron microscopy images of
complex ceramic materials” (npj Computational Materials 2024, available here &lt;
https://doi.org/10.1038/s41524-024-01226-5 &gt;).

## Description

### Datasets
1000 paired images (secondary electron images of iron-based high temperature superconductor ceramics obtained by scanning electron microscopy & manually segmented images).

Filename:“3D-SEM-Segmentation\dataset\train_dataset\train_dataset.npy.gz”

### Codes
pre-trained U-Net based semantic segmentation model.

Filename: “3D-SEM-Segmentation\demos\unet.h5”


## Demo

## Checking the Used Dataset
Run "3D-SEM-Segmentation/dataset/img_check.ipynb".

## Segmentation Using the Trained Model
Run "3D-SEM-Segmentation/demos/segmentation_demo.ipynb".

## Licence

[MIT](https://github.com/YamamotoLaboratory/3D-SEM-Segmentation/blob/main/LICENSE)

## References

[1] [Hirabayashi, Y., Iga, H., Ogawa, H., Tokuta, S., Shimada, Y. &amp; Yamamoto, A.
Deep learning for three-dimensional segmentation of electron microscopy
images of complex ceramic materials. npj Comp. Mat. (2024)](
https://doi.org/10.1038/s41524-024-01226-5)

[2] [Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image
Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015.
MICCAI 2015. Lecture Notes in Computer Science (Navab, N., Hornegger, J., Wells, W., Frangi, A.
(eds); Springer, Cham) 9351, 234–241 (2015).](https://doi.org/10.1007/978-3-319-24574-4_28).