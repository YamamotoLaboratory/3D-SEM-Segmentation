# 3D-SEM-Segmentation

## Overview
This project aims to perform high-precision, fully automated segmentation and binarization of sample images with two phases captured by SEM (Scanning Electron Microscopy).

The binarization technique adopts the U-net[1] model, one of the neural network models.

This research binarizes 3D slice SEM images of polycrystalline ceramic materials[2] and conducts three-dimensional mapping. The details of this research can be checked from npj Computational Materials (to be determined)[3].

This method has shown higher IoU compared to traditional thresholding binarization[4].

The dataset and program used in this study are publicly available in this repository.

## Description
This program is published for those with some programming experience. Please refer to the requirements for the operating environment.

For those without experience, guidance is available to check the operation on a PC, so please refer to the Demo.

This program also supports binarization of your grayscale images by preparing a pair of grayscale images and their binarized teacher images. For this method, please see Usage.

## Requirement
The research was conducted under the following environment:

- GPU: Nvidia Quadro RTX5000 16 GB GPU
- Python environment

```
python==3.8.8
tensorflow==2.4.1
numpy==1.19.5
pandas==1.2.3
matplotlib==3.3.4
rich==10.15.0
opencv-python==4.5.1.48
scikit-learn==0.24.2
scikit-image==0.19.3
PyYAML==5.4.1
jupyterlab==4.0.10
```

## Demo
This program operates with Python. You will need to set up a Python operating environment in Jupyter on your device.
The guide supports operations for Windows users only. Please follow the instructions below to set up your environment.

### Repository Acquisition

- Download and unzip the ZIP from GitHub, then execute.
Download and unzip the repository from the GitHub page via Code→Download ZIP.

- Clone the repository from GitHub
If Git is installed, execute the following command.

```
git clone https://github.com/YamamotoLaboratory/3D-SEM-Segmentation.git
```

### Preparation of Python Environment
- Preparation of Python environment with Docker
Execute the following command.

```
docker compose up -d
```

- Preparation of Python environment locally
Move to the repository directory and install the libraries.

```
pip install -r requirements.txt
pip install jupyterlab==4.0.10
```

Then execute the following command to start Jupyter Lab.

```
jupyter lab
```

## Checking the Used Dataset
1. After starting, access [jupyter lab](http://localhost:8888/lab?).
2. From the File Browser on the left side of the jupyter lab, expand 3D-SEM-Segmentation/dataset/img_check.ipynb.
3. Execute from the ▶▶ mark "Restart the kernel and run all cells" at the top of jupyter.
If the SEM image is displayed on the left and the binarized image on the right at the bottom, it is operating correctly.

## Segmentation Using the Trained Model
1. From the File Browser on the left side of jupyter lab, expand 3D-SEM-Segmentation/demos/segmentation_demo.ipynb.
2. Execute from the ▶▶ mark "Restart the kernel and run all cells" at the top of jupyter.
3. If the SEM image is displayed on the left and the binarized image on the right at the bottom, it is operating correctly.


## Licence

[MIT](https://github.com/YamamotoLaboratory/3D-SEM-Segmentation/blob/main/LICENSE)

## Reference
[1] [Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image
Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015.
MICCAI 2015. Lecture Notes in Computer Science (Navab, N., Hornegger, J., Wells, W., Frangi, A.
(eds); Springer, Cham) 9351, 234–241 (2015).](https://doi.org/10.1007/978-3-319-24574-4_28).

[2] Kamihara, Y., Watanabe, T., Hirano, M. & Hosono, H., Iron-Based Layered Superconductor La[O1-
xFx]FeAs (x=0.05−0.12) with Tc=26 K. J. Am. Chem. Soc. 130, 3296–3297 (2008). 

[3] Our paper.

[4] [Schneider, C., Rasband, W. & Eliceiri, K. NIH Image to ImageJ: 25 years of image analysis. Nat.
Methods 9, 671–675 (2012).](https://doi.org/10.1038/nmeth.2089).

## Contact us
[YamamotoLaboratory](https://web.tuat.ac.jp/~yamamoto/)

[Github](https://github.com/YamamotoLaboratory)