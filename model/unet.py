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
import sys

from model.common import *


class EncoderDecoder(nn.Module):
  def __init__(self, model_config, size_in, depth=0):
    super(EncoderDecoder, self).__init__()

    self.pooled_skip = False
    if 'pooled_skip' in model_config:
      self.pooled_skip = model_config['pooled_skip']

    enc_config = model_config['encoder_stages']
    dec_config = model_config['decoder_stages']

    enc_size = enc_config[depth][0]
    num_conv = enc_config[depth][1]

    self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv)
    depth_n = depth + 1

    if depth_n == len(enc_config):
      self.encoder_decoder = None
      return

    enc_size_next = enc_config[depth_n][0]
    dec_size = dec_config[depth][0]
    num_conv = dec_config[depth][1]

    self.encoder_decoder = EncoderDecoder(model_config,enc_size,depth_n)

    if self.pooled_skip:
      dec_size_in = size_in + enc_size_next
    else:
      dec_size_in = enc_size + enc_size_next

    self.decoder_stage = DecoderStage(model_config,dec_size_in,dec_size,num_conv=num_conv)
    return

  def forward(self, x_in):
    x = self.encoder_stage(x_in)

    if self.encoder_decoder:
      y = fun.avg_pool2d(x, kernel_size=2)
      y = self.encoder_decoder(y)
    else:
      return x
    
    y = fun.interpolate(y, scale_factor=2, mode='bilinear',align_corners=False)

    if self.pooled_skip:
      xy = torch.cat([x_in,y],dim=1)
    else:
      xy = torch.cat([x,y],dim=1)

    y = self.decoder_stage(xy)

    return y
   
class UNet(nn.Module):
  def __init__(self, model_config, size_in):
    super(UNet, self).__init__()
    
    dec_config = model_config['decoder_stages']
    out_channels = dec_config[0][0]

    if "pix_shuffle_size" in model_config:
      self.pix_shuffle_size = model_config["pix_shuffle_size"]
      shuffle_size = self.pix_shuffle_size ** 2

      if out_channels % shuffle_size != 0:
        sys.exit('Decoder output not a multiple of pixel shuffle size')

      size_in = size_in * shuffle_size
      out_channels = out_channels // shuffle_size
    else:
      self.pix_shuffle_size = 1

    self.encoder_decoder = EncoderDecoder(model_config,size_in)

    # convolution layer to project to RGB
    self.conv = nn.Conv2d(out_channels + 6,3,kernel_size=1,bias=True,padding=0)
    self.conv.bias.data.fill_(0)
    self.act = get_act("relu",model_config,self.conv)

  def forward(self, x):
    if self.pix_shuffle_size > 1:
      x1 = space_to_depth(x, block_size=self.pix_shuffle_size)
      y1 = self.encoder_decoder(x1)
      y  = depth_to_space(y1, block_size=self.pix_shuffle_size)
    else:
      y = self.encoder_decoder(x)

    xy = torch.cat([x,y],dim=1)
    xy = self.conv(xy)
    xy = self.act(xy)
    return xy
