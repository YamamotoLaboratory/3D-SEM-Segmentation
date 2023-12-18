# 3D-SEM-Segmentation

## Overview
このプロジェクトはＳＥＭで撮影された二つの相を持つ試料の画像を、
完全に自動で高精度なセグメンテーションを行い二値化することを目的としています。

二値化の手法にはニューラルネットワークのモデルの一つであるU-net[1]が採用されています。

本研究は多結晶セラミック材料[2]の３ＤスライスＳＥＭ画像を二値化し、
3次元マッピングを行っており、この研究はnpj Computational Materials(決まったら)[3]から詳細を確認することができます。

この手法は従来の閾値処理による二値化[4]と比べてより高いIoUを示しました。

本研究に用いられたデータセット及びプログラムはこのrepositoriで公開されています。




## Description

このプログラムはある程度のプログラム経験者向けに公開されています。動作環境についてはrequiermentをご覧ください。

経験がない方に向けてPCをお持ちの方であれば動作を確認できるよう案内があるため、Demoをご覧ください。

またこのプログラムは任意のグレースケールの画像とその二値化を行った教師画像のペアを用意していただくことで、
お手持ちの画像の二値化に対応しています。こちらの方法に関してはUsageをご覧ください。

## Requirement

## Demo
このプログラムはpythonで動作するプログラムです。
お使いの環境でjupyterのpython動作環境を整備する必要があります。
この案内ではWindowsユーザーの動作のサポートのみを行います。以下の案内に従い動作環境を整えてください。

ローカル環境での準備
1. [anaconda](https://www.anaconda.com/download)をダウンロード。インストールする。
2. anaconda.nabigator を起動しenvironmentsからcreateを実行し新規環境を作成し、切り替える。
3. serach packagesからrequirementに指定されたライブラリを検索し、チェックを入れてapplyから導入する。
4. homeからall applications on your choose name になっていることを確認し、jupyterlabを起動する。

データのダウンロード、解凍、実行
1. github上のページからCode→download ZIPでデータをダウンロードする。
2. ZIPデータを展開し、jupyterlabのブラウザ左部にあるFileBrowzerから
3D-SEM-Segmentation/dataset/img_check.ipynbを展開する。
3. jupyter上部にある▶▶マークから実行する。

この状態で最下段左部にSEM画像、右部に二値化画像が表示されていれば正常に動作しています。
## Usage

## Licence

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
[YamamotoLaboratory]
(https://github.com/YamamotoLaboratory)
(https://web.tuat.ac.jp/~yamamoto/)
