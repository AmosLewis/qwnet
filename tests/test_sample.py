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

smp_file = Path('/home/chi/src/ubuntu20/shark/RP_Network/image_dir/noaa/VIS_ArtDemo_P_PreTonemapHDRColor_0200.exr')

smp_image = exr.open(str(smp_file)).get() 
dims = smp_image.shape
print("dim: ", dims)  # (4, 1080, 1920)

smp_image = decode_sample(smp_image,dims[1:])
print("smp_image: ", smp_image)  # (array(0.17260742, dtype=float32), array(0.19995117, dtype=float32))

