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
import torch as torch
import torch.nn as nn

import torch.nn.functional as fun
from model.unet       import *
from model.unet_skip  import *
from model.wnet       import *
# from model.dlaa       import *
from model.common     import *
from model.warp       import *

class RecurrentBase(nn.Module):

  def half(self,*args,**kwargs):
    super(RecurrentBase, self).half(*args,**kwargs)
    self.warp.half(*args,**kwargs)

  def cuda(self,*args,**kwargs):
    super(RecurrentBase, self).cuda(*args,**kwargs)
    self.warp.cuda(*args,**kwargs)
    return

  def to(self,*args,**kwargs):
    super(RecurrentBase, self).to(*args,**kwargs)
    self.warp.to(*args,**kwargs)

  def __init__(self, args, image_shape):
    super(RecurrentBase, self).__init__()

    self.warp = Warp(image_shape)

    (channels, height, width) = image_shape
    self.channels = channels
    self.height   = height
    self.width    = width

    model_config = args['model_config']
    model_name   = args['model']

    if model_name == 'unet':
      self.model = UNet(model_config, 2 * channels)
    elif model_name == 'unet_skip':
      self.model = UNetSkip(model_config, 2 * channels)
    elif model_name == 'wnet':
      self.model = WNet(model_config, 2 * channels)
    # elif model_name == 'dlaa':
      # self.model = DLAA(model_config, 2 * channels, height, width)
    else:
      self.model = UNet(model_config, 2 * channels)

    # not registering buffers since image dimensions change for inference
    # causing checkpoint mismatches
    #self.register_buffer('grid_x',grid_x)
    #self.register_buffer('grid_y',grid_y)
    return


class RecurrentTrain(RecurrentBase):

  # Have to explicitly suppot these methods 
  # since we do not register these buffers in the checkpoint
  def half(self,*args,**kwargs):
    super(RecurrentTrain, self).half(*args,**kwargs)
    self.step_weights   = self.step_weights.half(*args,**kwargs)
    self.spatial_factor = self.spatial_factor.half(*args,**kwargs)
    
  def cuda(self,*args,**kwargs):
    super(RecurrentTrain, self).cuda(*args,**kwargs)
    self.step_weights   = self.step_weights.cuda(*args,**kwargs)
    self.spatial_factor = self.spatial_factor.cuda(*args,**kwargs)
    
  def to(self,*args,**kwargs):
    super(RecurrentTrain, self).to(*args,**kwargs)
    self.step_weights   = self.step_weights.to(*args,**kwargs)
    self.spatial_factor = self.spatial_factor.to(*args,**kwargs)
    
  def __init__(self, args, image_shape, timesteps):
    super(RecurrentTrain, self).__init__(args, image_shape)

    self.timesteps = timesteps
    self.loss_fn = nn.L1Loss(reduction='mean')

    if 'loss_config' in args:
      loss_config         = args['loss_config']
      self.spatial_factor = torch.tensor(loss_config['spatial_factor'], dtype=torch.float32)
      self.step_weights   = torch.tensor(loss_config['step_weights'], dtype=torch.float32)
      self.step_weights   = timesteps * (self.step_weights / torch.sum(self.step_weights))
    else:
      self.spatial_factor = torch.tensor(0.3, dtype=torch.float32)
      self.step_weights   = torch.ones((timesteps),dtype=torch.float32)

    self.temporal_factor = 1 - self.spatial_factor
    return

  def recurrent_step(self, input, ref, prev, ref_prev):
    input_cat = torch.cat([input,prev],dim=1)
    output = self.model(input_cat)
    spatial_loss = self.loss_fn(output, ref)
        
    dt_ref = ref - ref_prev
    dt_out = output - prev
    temporal_loss = self.loss_fn(dt_out, dt_ref)

    return output, spatial_loss, temporal_loss

  def forward(self, batch):
    vel_images = batch[:,0              :2                ,...]
    inp_images = batch[:,2              :2+self.channels  ,...]
    ref_images = batch[:,2+self.channels:2+self.channels*2,...]

    # first time step, previous and current frames are the same
    out_prev = inp_images[:,:,0,...]
    ref_prev = ref_images[:,:,0,...]
    inp      = inp_images[:,:,0,...]
    ref      = ref_images[:,:,0,...]

    out, spatial_loss, temporal_loss = self.recurrent_step(inp,ref,out_prev,ref_prev)
    spatial_loss_tot  = [spatial_loss]
    temporal_loss_tot = [temporal_loss]

    for idx in range(1,self.timesteps):
      vel      = vel_images[:,:,idx,...]
      out_prev = self.warp(out, vel)
      ref_prev = self.warp(ref, vel)

      inp = inp_images[:,:,idx,...]
      ref = ref_images[:,:,idx,...]

      out, spatial_loss, temporal_loss = self.recurrent_step(inp,ref,out_prev,ref_prev)

      spatial_loss_tot.append(spatial_loss)
      temporal_loss_tot.append(temporal_loss)

    spatial_loss_avg  = torch.mean(torch.stack(spatial_loss_tot))
    temporal_loss_avg = torch.mean(torch.stack(temporal_loss_tot) * self.step_weights)

    temporal_factor = 1 - self.spatial_factor
    loss_avg = self.spatial_factor * spatial_loss_avg + temporal_factor * temporal_loss_avg

    return out, loss_avg, spatial_loss_avg

class RecurrentPredict(RecurrentBase):
  def __init__(self, args, image_shape):
    super(RecurrentPredict, self).__init__(args, image_shape)
    return

  def forward(self, input, vel, prev):
    prev_warp = self.warp(prev, vel)
    input_cat = torch.cat([input,prev_warp],dim=1)
    output = self.model(input_cat)
    return output