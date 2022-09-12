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
import torch
import numpy as np
import utils.exr as exr

from collections import deque
from pathlib import Path

# compute sample offset
def decode_sample(image, image_size, dtype=torch.float32):
  code = image[2,0,0].astype(int)
  smp_image = image[0:2,...]
  smp_x = np.where(code & 1, -image[0,0,0], image[0,0,0])
  smp_y = np.where(code & 2, -image[1,0,0], image[1,0,0])
  return (smp_x, smp_y)

def decode_image(image, dtype=torch.float32):
  image = torch.tensor(image[0:3,...], dtype=dtype) 
  image = image.unsqueeze(0)
  return image

# compute velocity in pixels
def decode_vel(image, image_size, dtype=torch.float32):
  code = image[2,...].astype(int)
  vel_image = image[0:2,...]
  vel_image[0,...] = np.where(code & 1, -image[0,...], image[0,...])
  vel_image[1,...] = np.where(code & 2, -image[1,...], image[1,...])
  vel_image[0,...] =  vel_image[0,...] * image_size[1] / 2
  vel_image[1,...] = -vel_image[1,...] * image_size[0] / 2
  vel_image = torch.tensor(vel_image, dtype=dtype) 
  vel_image = vel_image.unsqueeze(0)
  return vel_image

def make_path_lists(image_dir, batch_size, num_gpus=1, max_files=0):
  ssaa_dir = image_dir / Path('ssaa')
  noaa_dir = image_dir / Path('noaa')
  vel_dir  = image_dir / Path('vel')

  file_names = [x.name for x in ssaa_dir.glob('*.exr')]

  if max_files:
    file_names = file_names[0:max_files]

  num_files = len(file_names)

  parallel_batch_size = batch_size*num_gpus
  rem_files = num_files % parallel_batch_size
  if rem_files:
    pad = parallel_batch_size - rem_files
    file_names = file_names + file_names[0:pad]

  num_batches = len(file_names) // batch_size

  file_paths = [(str(noaa_dir/x), str(vel_dir/x), str(ssaa_dir/x)) for x in file_names]
  return (num_batches, file_paths)

class StreamDataset(torch.utils.data.Dataset):
  def __init__(self, input_paths, block_dims, resolution):
    super(StreamDataset, self).__init__()
    self.block_dims = block_dims
    self.input_paths = input_paths
    self.resolution  = resolution
    return

  def block_loader(self, idx):
    input_paths = self.input_paths
    time_steps  = self.block_dims[0]
    channels    = self.block_dims[1]
    image_shape = self.block_dims[2:]

    noaa_shape  = (channels,time_steps) + image_shape
    vel_shape   = (2       ,time_steps) + image_shape
    block_shape = (2*channels+2,time_steps) + image_shape

    block = torch.empty(block_shape, dtype=torch.float32)
    input_path = input_paths[idx]
    noaa_image = exr.open(input_path[0]).get()  
    vel_image  = exr.open(input_path[1]).get()  
    ssaa_image = exr.open(input_path[2]).get() 

    noaa_image = decode_image(noaa_image)
    ssaa_image = decode_image(ssaa_image)
    vel_image  = decode_vel(vel_image, self.resolution)

    vel_image  = np.reshape(vel_image ,vel_shape)
    # cannot reshape array of size 4147200 into shape (2,0,1920,1920)
    noaa_image = np.reshape(noaa_image,noaa_shape)
    ssaa_image = np.reshape(ssaa_image,noaa_shape)

    block[0         :2           ,...] = vel_image
    block[2         :2+channels  ,...] = noaa_image
    block[2+channels:2+2*channels,...] = ssaa_image    

    return block

  def __getitem__(self,idx):
    block = self.block_loader(idx)
    return block

  def __len__(self):
    return len(self.input_paths)

#def batch_loader(idx, input_paths, batch_dims):
#  input_paths = input_paths
#  batch_size  = batch_dims[0]
#  time_steps  = batch_dims[1]
#  channels    = batch_dims[2]
#  image_shape = batch_dims[3:]

#  noaa_shape  = (channels,time_steps) + image_shape
#  vel_shape   = (2       ,time_steps) + image_shape
#  batch_shape = (batch_size,2*channels+2,time_steps) + image_shape

#  start_image = idx * batch_size
#  batch_paths = input_paths[start_image:start_image+batch_size]
  
#  batch = torch.empty(batch_shape, dtype=torch.float32)

#  for sample_idx in range(0,batch_size):
#    input_path = batch_paths[sample_idx]
#    noaa_image = exr.open(input_path[0]).get()  
#    vel_image  = exr.open(input_path[1]).get()  
#    ssaa_image = exr.open(input_path[2]).get() 

#    noaa_image = decode_image(noaa_image)
#    ssaa_image = decode_image(ssaa_image)
#    vel_image  = decode_vel(vel_image)

#    vel_image  = np.reshape(vel_image ,vel_shape)
#    noaa_image = np.reshape(noaa_image,noaa_shape)
#    ssaa_image = np.reshape(ssaa_image,noaa_shape)

#    batch[sample_idx,0         :2           ,...] = vel_image
#    batch[sample_idx,2         :2+channels  ,...] = noaa_image
#    batch[sample_idx,2+channels:2+2*channels,...] = ssaa_image    

#  return batch

#class BatchStreamer:
#  def __init__(self, pool, input_paths, batch_dims, queue_size=32, num_threads=8):
#    self.process_queue = deque()
#    batch_size         = batch_dims[0]
#    self.num_batches   = len(input_paths) // batch_size
#    self.chunk_size    = min(queue_size, self.num_batches)
#    self.pool          = pool
#    self.input_paths   = input_paths
#    self.batch_dims    = batch_dims
#    return

#  def __iter__(self):
#    # load up processes
#    for x in range(0,self.chunk_size):
#      args=(x,self.input_paths,self.batch_dims)
#      self.process_queue.append(self.pool.apply_async(batch_loader,args=args))

#    x = self.chunk_size
#    while len(self.process_queue):
#      batch_data = self.process_queue.popleft().get()
#      yield batch_data

#      if x < self.num_batches:
#        args=(x,self.input_paths,self.batch_dims)
#        self.process_queue.append(self.pool.apply_async(batch_loader, args=args))
#        x = x + 1

