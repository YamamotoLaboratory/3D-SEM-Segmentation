# 3D-SEM-Segmentation

## Overview
このプロジェクトは以下の論文で使用される画像データセットとコードについての解説です。“Deep
learning for three-dimensional segmentation of electron microscopy images of
complex ceramic materials” (npj Computational Materials 2024, available here &lt;
https://doi.org/10.1038/s41524-024-01226-5 &gt;).



## Description

### データセット
1000 個のペア画像 (鉄系超伝導体の走査電子顕微鏡による二次電子像と手動でセグメント化された画像)
Filename: “3D-SEM-Segmentation\dataset\train_dataset\train_dataset.npy.gz”

### プログラムコード
事前にトレーニングされたU-Netのセマンティックセグメンテーションモデル
Filename: “3D-SEM-Segmentation\demos\unet.h5”
## Demo

### 使用したデータセットの確認

3D-SEM-Segmentation/dataset/img_check.ipynbを実行してください。


### 学習済みモデルを使用したセグメンテーション

3D-SEM-Segmentation/demos/segmentation_demo.ipynbを実行してください。


## Licence

[MIT](https://github.com/YamamotoLaboratory/3D-SEM-Segmentation/blob/main/LICENSE)

## References
[1] Hirabayashi, Y., Iga, H., Ogawa, H., Tokuta, S., Shimada, Y. &amp; Yamamoto, A.
Deep learning for three-dimensional segmentation of electron microscopy
images of complex ceramic materials. npj Comp. Mat. (2024)
https://doi.org/10.1038/s41524-024-01226-5

[2] [Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image
Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015.
MICCAI 2015. Lecture Notes in Computer Science (Navab, N., Hornegger, J., Wells, W., Frangi, A.
(eds); Springer, Cham) 9351, 234–241 (2015).](https://doi.org/10.1007/978-3-319-24574-4_28).
