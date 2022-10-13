# MRSRGAN
Notice there's no pre-trained weight available for this model, but the weight could be provided per request via email.

## Pre-processing

* Standardize your images by mean=0, std=1
* Crop your 3D MRI images into 64\*64\*64 cubes by running:

  `python mains/utils/crop_nifti.py /data/path/to/your/images "your subjectes prefix string(for wildcard search)"` 
  
## Run the training:

__WGAN-GP__

  `python mains/MRSRGAN_WGAN_GP.py --path /your/data/crop/path --val_path /your/validation_data/crop/path`

__Res10-GAN__

This script works the same way as is in __WGAN-GP__

## After inferencing

After the training you will get weights for your patches MRI cubes, then run the following after you've inferenced LR pachtes to assemble them up:

  `python mains/utils/assemble_crop_v3.py --path /path/to/the/inferred_patches/ --subj "wildcard_name" --scale int(upscale_factor)`

## Citation
```
@inproceedings{
wang2022superresolution,
title={Super-Resolution for Ultra High-Field {MR} Images},
author={Qi Wang and Julius Steiglechner and Tobias Lindig and Benjamin Bender and Klaus Scheffler and Gabriele Lohmann},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=EFiFV2MSNEB}
}
```
