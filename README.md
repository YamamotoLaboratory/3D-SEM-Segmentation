## 3D-SEM-Segmentation

## Overview
This project is aimed at fully automated and accurate segmentation and binarization of SEM images of samples in two phases,
This project aims to perform fully automatic and accurate segmentation and binarization of images of samples in two phases taken by SEM.

U-net[1], one of the models of neural networks, is employed for the binarization method.

This study binarizes 3D slice SEM images of polycrystalline ceramic materials [2],
3D mapping, which can be viewed in detail at npj Computational Materials [3].

This method showed a higher IoU compared to traditional thresholding binarization [4].

The dataset and program used in this study are available in this repository.


## Description

This program is intended for people with some programming experience. Please see the requierment for the system requirements.

For those who have no experience, please see Demo, as there is a guide for those who have a PC to see how it works.

The program also requires you to prepare a pair of arbitrary grayscale images and their binarized teacher images,
You can also binarize your own image by preparing a pair of any grayscale image and its binarized teacher image. Please see Usage for more information on how to do this.

## Requirement
These studies were conducted under the following environment

- GPU: Nvidia Quadro RTX5000 16 GB GPU

- Python environment
```
Python environment
python==3.8.8
tensorflow==2.4.1
numpy==1.19.5
pandas==1.2.3
matplotlib==3.3.4
rich==10.15.0
opencv-python==4.5.1.48
scikit-learn=0.24.2
scikit-image=0.19.3
PyYAML==5.4.1
jupyterlab==4.0.10
```

## Demo
This program runs in python.
You need to set up a python environment for jupyter in your environment.
This guide only supports Windows users. Please follow the instructions below to set up your environment.

### Obtain a repository

- Download, unzip, and run the zip from Gitbub.

Download and unzip the repository from the github page by clicking Code→download ZIP.

- Clone the repository from Gitbub

If you have Git, execute the following command.
```
git clone https://github.com/YamamotoLaboratory/3D-SEM-Segmentation.git
```

- Preparing the Python environment in the local environment

Go to the repository directory and install the library.
```
pip install -r requiremets.txt
pip install jupyterlab==4.0.10
```
Next, run the following command to start jupyter lab.
```
jupyter lab
```

### Check the dataset you used. 1.

1. access [jupyter lab](http://localhost:8888/lab?) after startup.
1. extract the file 3D-SEM-Segmentation/dataset/img_check.ipynb from FileBrowzer in the left part of the jupyterlab browser. 2.
Execute the file from the ▶▶ symbol at the top of jupyter.

If the SEM image is displayed in the left part of the bottom row and the binarized image is displayed in the right part of the bottom row, it is working properly.

### Segmentation using learned models

1. extract the file 3D-SEM-Segmentation/demos/segmentation_demo.ipynb from FileBrowzer in the left part of the jupyterlab browser. 2.
2. run the file from the ▶▶ symbol at the top of jupyter.

## Licence

[MIT](https://github.com/YamamotoLaboratory/3D-SEM-Segmentation/blob/main/LICENSE)

## Reference
[1] [Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image
Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015.
MICCAI 2015. Lecture Notes in Computer Science (Navab, N., Hornegger, J., Wells, W., Frangi, A.
(eds); Springer, Cham) 9351, 234-241 (2015).] (https://doi.org/10.1007/978-3-319-24574-4_28).

[2]Kamihara, Y., Watanabe, T., Hirano, M. & Hosono, H., Iron-Based Layered Superconductor La[O1-
xFx]FeAs (x=0.05-0.12) with Tc=26 K. J. Am. Chem. Soc. 130, 3296-3297 (2008). 

[3] Our paper.

[4] [Schneider, C., Rasband, W. & Eliceiri, K. NIH Image to ImageJ: 25 years of image analysis. Nat.
Methods 9, 671-675 (2012).] (https://doi.org/10.1038/nmeth.2089).

## Contact us
[YamamotoLaboratory](https://web.tuat.ac.jp/~yamamoto/)

[Github](https://github.com/YamamotoLaboratory)