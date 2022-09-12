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
import argparse

import utils.exr as exr
from utils.input import *
from utils.warp  import *

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


ap = argparse.ArgumentParser()
ap.add_argument('image_dir', default='./', help="Path to directory with noaa, ssaa, vel sub directories")
args = vars(ap.parse_args())

batch_size = 1
image_dir  = Path(args['image_dir'])

# setup training and validation image paths
_, train_paths = make_path_lists(image_dir,num_gpus=1,batch_size=batch_size)

# load one image to get dimensions
sample_image = exr.open(str(train_paths[0][0]))
height = sample_image.height

time_steps = sample_image.height // sample_image.width
image_shape = (3, sample_image.width, sample_image.width)

block_dims = (time_steps,) +  image_shape

# make dataset
train_dataset = StreamDataset(train_paths,block_dims,(1080,1920))
train_sampler = DistributedSampler(train_dataset,num_replicas=1,rank=0)
train_loader  = DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler,shuffle=False,pin_memory=True,num_workers=0)

warp = Warp(image_shape)

for batch in train_loader:
  vel_images = batch[:,0  :2    ,...]
  inp_images = batch[:,2  :2+3  ,...]
  ref_images = batch[:,2+3:2+3*2,...]

  exr.write('fr0.exr', data = ref_images[:,:,0,...].cpu().squeeze(0).numpy(), precision = exr.HALF)
  exr.write('fr1.exr', data = ref_images[:,:,1,...].cpu().squeeze(0).numpy(), precision = exr.HALF)

  test = warp(ref_images[:,:,0,...], vel_images[:,:,1,...])

  exr.write('fr2.exr', data = test.cpu().squeeze(0).numpy() , precision = exr.HALF)
  exit(0)
