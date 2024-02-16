# 3D-SEM-Segmentation

## Overview
このプロジェクトはＳＥＭで撮影された二つの相を持つ試料の画像を、
完全に自動で高精度なセグメンテーションを行い二値化することを目的としています。

二値化の手法にはニューラルネットワークのモデルの一つであるU-net[1]が採用されています。

本研究は多結晶セラミック材料[2]の３ＤスライスＳＥＭ画像を二値化し、
3次元マッピングを行っており、この研究はnpj Computational Materials(決まったら)[3]から詳細を確認することができます。

この手法は従来の閾値処理による二値化[4]と比べてより高いIoUを示しました。

本研究に用いられたデータセット及びプログラムはこのrepositoryで公開されています。


## Description

このプログラムはある程度のプログラム経験者向けに公開されています。動作環境についてはrequiermentをご覧ください。

経験がない方に向けてPCをお持ちの方であれば動作を確認できるよう案内があるため、Demoをご覧ください。

またこのプログラムは任意のグレースケールの画像とその二値化を行った教師画像のペアを用意していただくことで、
お手持ちの画像の二値化に対応しています。こちらの方法に関してはUsageをご覧ください。

## Requirement
これらの研究は以下の環境のもと実施いたしました。

- GPU：Nvidia Quadro RTX5000 16 GB GPU

- Python環境
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
このプログラムはPythonで動作するプログラムです。
お使いの環境でjupyterのPython動作環境を整備する必要があります。
この案内ではWindowsユーザーの動作のサポートのみを行います。以下の案内に従い動作環境を整えてください。

### リポジトリの取得

- Githubからのzipダウンロード、解凍、実行

Github上のページからCode→download ZIPでリポジトリをダウンロードし解凍する。

- Githubからリポジトリをクローン

Gitがある場合、下記コマンドを実行する。
```
git clone https://github.com/YamamotoLaboratory/3D-SEM-Segmentation.git
```

### Python環境の準備

- DockerでのPython環境の準備
  
以下コマンド実行する。
```
docker compose up -d
```

- ローカル環境でのPython環境の準備

リポジトリのディレクトリに移動し、ライブラリをインストールする。
```
pip install -r requiremets.txt
pip install jupyterlab==4.0.10
```
次に以下コマンドを実行しjupyter labを起動する。
```
jupyter lab
```

### 使用したデータセットの確認

1. 起動後に[jupyter lab](http://localhost:8888/lab?)アクセスする。
1. jupyterlabのブラウザ左部にあるFileBrowzerから3D-SEM-Segmentation/dataset/img_check.ipynbを展開する。
2. jupyter上部にある▶▶マーク「Restart the kernel and run all cells」から実行する。

この状態で最下段左部にSEM画像、右部に二値化画像が表示されていれば正常に動作しています。

### 学習済みモデルを使用したセグメンテーション

1. jupyterlabのブラウザ左部にあるFileBrowzerから3D-SEM-Segmentation/demos/segmentation_demo.ipynbを展開する。
2. jupyter上部にある▶▶マーク「Restart the kernel and run all cells」から実行する。
3. 最下段左部にSEM画像、右部に二値化画像が表示されていれば正常に動作しています。

## Licence

[MIT](https://github.com/YamamotoLaboratory/3D-SEM-Segmentation/blob/main/LICENSE)

## Reference
[1] [Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image
Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015.
MICCAI 2015. Lecture Notes in Computer Science (Navab, N., Hornegger, J., Wells, W., Frangi, A.
(eds); Springer, Cham) 9351, 234–241 (2015).](https://doi.org/10.1007/978-3-319-24574-4_28).

[2]Kamihara, Y., Watanabe, T., Hirano, M. & Hosono, H., Iron-Based Layered Superconductor La[O1-
xFx]FeAs (x=0.05−0.12) with Tc=26 K. J. Am. Chem. Soc. 130, 3296–3297 (2008). 

[3]私たちの論文です。

[4] [Schneider, C., Rasband, W. & Eliceiri, K. NIH Image to ImageJ: 25 years of image analysis. Nat.
Methods 9, 671–675 (2012).](https://doi.org/10.1038/nmeth.2089).

## Contact us
[YamamotoLaboratory](https://web.tuat.ac.jp/~yamamoto/)

[Github](https://github.com/YamamotoLaboratory)