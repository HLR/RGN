# RGN
The source code of IJCAI 2021 paper: Relational Gating for ``What If'' Reasoning

<p align="center"><img width="95%" src="images/rgn_framework.pdf" /></p>


## Download the parameters to the RGN_model folder
link:
https://drive.google.com/file/d/1xT_h5Fe2Uf0KXOgGGZce558RV6Emy9eO/view?usp=sharing


## Conda environment
>- conda env create -f rgn.yaml
>- source activate rgn

## How to test RGN model:
>- cd RGN_model
>- tar zxvf RGN_ckpt.tar.gz
>- mv RGN_ckpt saved_model
>- sh run_test.sh

## How to train RGN model:
>- cd RGN_model
>- sh run_train.sh
