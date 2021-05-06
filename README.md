# RGN
The source code of IJCAI 2021 paper


## download the parameters
link:
https://drive.google.com/file/d/1xT_h5Fe2Uf0KXOgGGZce558RV6Emy9eO/view?usp=sharing


## conda environment
>- conda env create -f rgn.yaml
>- source activate rgn

## Test model:
>- cd RGN_model
>- tar zxvf RGN_ckpt.tar.gz
>- mv RGN_ckpt saved_model
>- sh run_test.sh

## Train model:
>- cd RGN_model
>- sh run_train.sh
