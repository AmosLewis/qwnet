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
import os
import argparse
import json

from tqdm import tqdm
from pathlib import Path

import numpy as np
import multiprocessing as mp
import random

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from utils.input        import *
from utils.log          import *
from model.recurrent    import *

CHANNELS = 3

if __name__ == '__main__': 
  ap = argparse.ArgumentParser()
  ap.add_argument('--image_dir'         , default='./image_dir/'                , help="Path to directory with noaa, ssaa, vel image directories")
  ap.add_argument('--output_dir'      , default='./output'         , required=False, help="output directory for the predicted images")
  ap.add_argument('--session_name'    , default=None                , required=False, help="name for the training session")
  ap.add_argument('--model'           , default='unet'              , required=False, help="frnn_staged or frnn_direct")
  ap.add_argument('--config'          , default='configs/train.json', required=False, help="config file")
  ap.add_argument('--checkpoint_dir'  , default='./checkpoints'    , required=False, help="checkpoint directory")
  ap.add_argument('--jit'             , default=False               , required=False, help="jit the model", action='store_true')

  args = vars(ap.parse_args())

  # load json config
  with open(args['config']) as f:
    config = json.load(f)
  args.update(config)

  model_config = args['model_config']

  # setup paths
  image_dir = Path(args['image_dir'])
  out_dir   = Path(args['output_dir'])
  bin_dir = Path('./bin')
  tmp_dir = Path('../tmp')

  noaa_dir = image_dir / Path('noaa_mip-1')
  vel_dir  = image_dir / Path('velocity')

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

  file_names = [x.name for x in noaa_dir.iterdir() if x.suffix == '.exr' ]
  image_count = len(file_names)

  # load one image to get dimensions
  file_name = file_names[0]
  noaa_path = str(noaa_dir/file_name)    
    
  sample_image = exr.open(noaa_path)
  height = sample_image.height
  width  = sample_image.width

  padded_height = ((height + 15) // 16) * 16   
  image_shape = (3, padded_height, width)

  pad_size = height % 16
  pad_size_top = pad_size // 2
  pad_size_btm = pad_size - pad_size_top
  paddings = (0,0,pad_size_top, pad_size_btm)

  # setup model
  model = RecurrentPredict(args,image_shape)
  print('-------------------------------------------Network----------------------------------------------')
  print(model)
  print('------------------------------------------------------------------------------------------------')

  #model.half()
  model.cuda()

  # jit model
  if args['jit']:
    print('Jitting Model')
    net_channels = CHANNELS * 2 + 2 # current, prev and velocity
    ip_size = (1,) +  image_shape
    vel_size = (1,2) +  image_shape[1:]
    dummy_image = torch.ones(ip_size,dtype=torch.float32 ).cuda()
    dummy_state = torch.ones(ip_size,dtype=torch.float32 ).cuda()
    dummy_vel   = torch.ones(vel_size,dtype=torch.float32).cuda()
    with torch.no_grad():
      model = torch.jit.trace(model,(dummy_image,dummy_vel,dummy_state))

  # setup checkpoint dirs
  checkpoint_file = Path('cp')

  if args['session_name'] is None:
    checkpoint_path = Path(args['checkpoint_dir'])/checkpoint_file
  else:
    checkpoint_path = Path(args['checkpoint_dir'])/Path(args['session_name'])/checkpoint_file

  # All processes load checkpoint
  if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    model_cp = checkpoint['model_state_dict']
    model_cp = conv_dist_checkpoint(model_cp)

    # load params
    model.load_state_dict(model_cp)
    print('loading checkpoint for epoch ' + str(epoch))
  else:
    print('no checkpoints found')
    exit()


  print('Processing %d images' % image_count)
  print('Tonemapping')
  # TODO: tonemap
  print()

  file_name = file_names[0]
  tmp_path  = str(tmp_dir/file_name)
  prev_image_exr = exr.open(tmp_path)
  header = prev_image_exr.header

  prev_image = prev_image_exr.get()
  prev_image = decode_image(prev_image)
  prev_image = torch.nn.functional.pad(prev_image,paddings,mode='reflect')
  #prev_image = prev_image.half()
  prev_image = prev_image.cuda()

  # training loop
  bar = tqdm(total=image_count,ascii=True)

  # Evaluate model
  model.eval()

  with torch.no_grad():
    for image_idx in range(0, image_count):
      file_name = file_names[image_idx]
      vel_path  = str(vel_dir/file_name)
      tmp_path  = str(tmp_dir/file_name)
      out_path  = str(out_dir/file_name)

      image_noaa = exr.open(tmp_path).get()  
      image_vel  = exr.open(vel_path).get()  

      image_noaa = decode_image(image_noaa)
      image_vel  = decode_vel(image_vel,(height,width))

      image_noaa = torch.nn.functional.pad(image_noaa,paddings,mode='reflect')
      image_vel  = torch.nn.functional.pad(image_vel , paddings,mode='reflect')

      #image_noaa = image_noaa.half()
      #image_vel  = image_vel.half()

      image_noaa = image_noaa.cuda()
      image_vel  = image_vel.cuda()

      prev_image = model(image_noaa,image_vel,prev_image)
      out_image = prev_image.cpu().numpy()
      out_image = np.maximum(out_image,0.0).squeeze(0)
      out_image = out_image[:,pad_size_top:pad_size_top+height,:]
      exr.write(out_path, data=out_image, precision=exr.HALF, header=header)

      bar.update(1)

  print()
  # TODO: inverse tonemap
