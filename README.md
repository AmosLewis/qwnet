# Reduced-Precision Network for Image Reconstruction Sample (RP-Network)

Copyright 2020 Intel Corporation

Licensed under the Apache License, Version 2.0 (the License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

https://creativecoding.soe.ucsc.edu/QW-Net/

## Requirements
Python  - 3.10
Pytorch with cuda
OpenEXR 
see env.txt

## Single GPU Training

python train.py --val_dir VAL_DIR --config configs/train.json --session_name MY_SESSION TRAIN_DIR

The default checkpoint directory is ../checkpoints/session_name

python train.py --val_dir ./val_dir/ --config configs/train.json --session_name MY_SESSION --image_dir ./image_dir/


## Single GPU Inference (Windows Only)

python predict.py --config config/train.json --session_name MY_SESSION LPMSKPAA_Dataset/data/Infiltrator/180_320

python predict.py  --config configs/train.json --session_name MY_SESSION --image_dir ./image_dir/     


