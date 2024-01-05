# 3D-SEM-Segmentation

## Overview
This project uses SEM images of a sample with two phases.
The purpose is to perform fully automatic and highly accurate segmentation and binarization.

The binarization method uses U-net [1], which is a neural network model.

In this research, we binarized 3D slice SEM images of polycrystalline ceramic materials [2],
We are conducting 3D mapping, and you can check the details of this research from npj Computational Materials (once decided) [3].

This method showed higher IoU compared to traditional thresholding binarization [4].

The datasets and programs used in this research are publicly available in this repository.




## Description

This program is open to those with some programming experience. Please see requirements for operating environment.

For those with no experience, if you have a PC, please take a look at the Demo to see how it works.

In addition, this program can be used by preparing a pair of arbitrary grayscale images and a teacher image that has been binarized.
It supports binarization of your images. Please see Usage for this method.

## Requirement

## Demo
This program runs on python.
You need to prepare a python operating environment for jupyter in your environment.
This guide only supports operations for Windows users. Please follow the instructions below to prepare the operating environment.

Preparation in local environment
1. Download [anaconda](https://www.anaconda.com/download). install.
2. Start anaconda.nabigator and execute create from environments to create a new environment and switch.
3. Search for the library specified in requirement from serach packages, check it, and install it with apply.
4. Check that all applications on your choose name is selected from home and start jupyterlab.

Download, unzip, and execute data
1. Download the data from the page on github using Code → download ZIP.
2. Extract the ZIP data and use FileBrowzer on the left side of the jupyterlab browser.
Expand 3D-SEM-Segmentation/dataset/img_check.ipynb.
3. Run from the ▶▶ mark at the top of jupyter.

In this state, if the SEM image is displayed on the bottom left and the binarized image is displayed on the right, it is operating normally.
## Usage

## License

[MIT](https://github.com/YamamotoLaboratory/3D-SEM-Segmentation/blob/main/LICENSE)

## Reference
[1] [Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image
Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015.
MICCAI 2015. Lecture Notes in Computer Science (Navab, N., Hornegger, J., Wells, W., Frangi, A.
(eds); Springer, Cham) 9351, 234–241 (2015).](https://doi.org/10.1007/978-3-319-24574-4_28).

[2]Kamihara, Y., Watanabe, T., Hirano, M. & Hosono, H., Iron-Based Layered Superconductor La[O1-
xFx]FeAs (x=0.05−0.12) with Tc=26 K. J. Am. Chem. Soc. 130, 3296–3297 (2008).

[4] [Schneider, C., Rasband, W. & Eliceiri, K. NIH Image to ImageJ: 25 years of image analysis. Nat.
Methods 9, 671–675 (2012).](https://doi.org/10.1038/nmeth.2089).

## Contact us
[YamamotoLaboratory](https://web.tuat.ac.jp/~yamamoto/)

[Github](https://github.com/YamamotoLaboratory)