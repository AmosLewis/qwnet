'''
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the 'License');
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an 'AS IS' BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and limitations under the License.
'''
import sys
sys.path.insert(1, sys.path[0] + '\\..\\')

import argparse
import torch as to
import numpy as np
from pathlib import Path

import utils.exr as exr
from utils.input import *
from model.warp  import *

def perceptual_tonemap(image):
  L = 10000.0
  n = 0.1593
  m = 78.8438
  c1 = 0.8359
  c2 = 18.8516
  c3 = 18.6815
  scake1KL = 1.33

  c = np.power(image / L, n)
  out = np.power((c2 * c + c1) / (c3 * c + 1), m) * scake1KL
  return out


noaa_file0 = Path('D:/Data/LPMSKPAA_Dataset/data/Infiltrator/180_320/ssaa/VIS_ArtDemo_P_PreTonemapHDRColor_0180.exr')
noaa_file1 = Path('D:/Data/LPMSKPAA_Dataset/data/Infiltrator/180_320/ssaa/VIS_ArtDemo_P_PreTonemapHDRColor_0181.exr')
vel_file   = Path('D:/Data/LPMSKPAA_Dataset/data/Infiltrator/180_320/velocity/VIS_ArtDemo_P_PreTonemapHDRColor_0181.exr')

noaa_image0 = exr.open(str(noaa_file0)).get() 
noaa_image1 = exr.open(str(noaa_file1)).get() 
vel_image   = exr.open(str(vel_file)).get()

dims = noaa_image0.shape
image_shape = (3, dims[1], dims[2])
warp = WarpBicubic(image_shape)

noaa_image0 = decode_image(noaa_image0)
noaa_image1 = decode_image(noaa_image1)

#noaa_image0 = perceptual_tonemap(noaa_image0)
#noaa_image1 = perceptual_tonemap(noaa_image1)

vel_image = decode_vel(vel_image, dims[1:])

warped_image = warp(noaa_image0, vel_image)

exr.write('fr2.exr', data = warped_image.squeeze(0).numpy(), precision = exr.HALF)
exr.write('fr0.exr', data = noaa_image0.squeeze(0).numpy() , precision = exr.HALF)
exr.write('fr1.exr', data = noaa_image1.squeeze(0).numpy() , precision = exr.HALF)


