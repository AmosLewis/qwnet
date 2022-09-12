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
from model.unet import *
# from model.dlaa import *

class SingleFrame(nn.Module):
  def __init__(self, model_name, model_config, image_shape, timesteps):
    super(SingleFrame, self).__init__()

    (channels, height, width) = image_shape

    if model_name == 'unet':
      self.model = UNet(model_config, channels)
    # elif model_name == 'dlaa':
    #   self.model = DLAA(model_config, channels, height, width)
    else:
      return

    self.channels = channels
    self.timesteps = timesteps
    self.loss_fn = nn.L1Loss(reduction='mean')
    return

  def step(self, input, ref):
    output = self.model(input)
    spatial_loss = self.loss_fn(output, ref)
    return output, spatial_loss

  def forward(self, batch):
    inp_images = batch[:,2              :2+self.channels  ,...]
    ref_images = batch[:,2+self.channels:2+self.channels*2,...]

    # first time step
    _, spatial_loss_tot = self.step(inp_images[:,:,0,...],ref_images[:,:,0,...])

    for idx in range(1,self.timesteps):    
      _, spatial_loss = self.step(inp_images[:,:,idx,...],ref_images[:,:,idx,...])
      spatial_loss_tot += spatial_loss

    loss = spatial_loss_tot / self.timesteps

    return loss