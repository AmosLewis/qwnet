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

from model.common import *
from model.quantize import *

class FilterStage3x3(nn.Module):
  def __init__(self, model_config, num_features, num_channels, num_filters=1, skip=False, linear=False):
    super(FilterStage3x3, self).__init__()
    quantize, _ = get_quant_settings(model_config)

    if skip:
      self.conv = QConv2D(num_features,num_filters*9+1,quantize=quantize,num_bits=8,kernel_size=1,bias=True,padding=0)
    else:
      self.conv = QConv2D(num_features,num_filters*9  ,quantize=quantize,num_bits=8,kernel_size=1,bias=True,padding=0)

    self.conv.bias.data.fill_(0)
    self.linear = linear
    if linear:
      self.act = nn.ReLU()
    else:
      self.act = nn.Softmax(dim=1)

    self.filter = Filter(num_channels,num_filters,skip)

  def forward(self, f, x, skip=None):
    f = self.conv(f)
    if self.linear:
      y = self.filter(f,x,skip)
      y = self.act(y)
    else:
      k = self.act(f)
      y = self.filter(k,x,skip)

    return y

class FilterNetwork(nn.Module):
  def __init__(self, model_config, size_in, shuffle_size=1,depth=0):
    super(FilterNetwork, self).__init__()

    self.shuffle_size = shuffle_size

    linear_kernel = False
    if 'linear_kernel' in model_config:
      linear_kernel = model_config['linear_kernel']

    enc_config = model_config['encoder_stages']
    enc_size = enc_config[depth][0]
    red_size = size_in // 2
    
    depth_n = depth + 1

    if depth_n < len(enc_config):
      dec_config = model_config['decoder_stages']
      dec_size = dec_config[depth][0]

      if depth == 0:
        self.filter_stage_dwn = FilterStage3x3(model_config,dec_size,red_size,num_filters=2,linear=linear_kernel)
      else:
        self.filter_stage_dwn = FilterStage3x3(model_config,dec_size,red_size,num_filters=1,linear=linear_kernel)

      self.filter_nw = FilterNetwork(model_config,size_in,shuffle_size,depth_n)
      self.filter_stage_ups = FilterStage3x3(model_config,dec_size,red_size,num_filters=1,skip=True,linear=linear_kernel)

    else:    
      self.filter_stage_dwn = FilterStage3x3(model_config,enc_size,red_size,num_filters=1,linear=linear_kernel)
      self.filter_nw = None
      
    return

  def forward(self, f, x):
    features = f.pop()
    
    if self.shuffle_size > 1:
      features = fun.interpolate(features, scale_factor=self.shuffle_size, mode='nearest',align_corners=None)

    x = self.filter_stage_dwn(features,x)

    if self.filter_nw:
      yr = fun.avg_pool2d(x, kernel_size=2)
      yr = self.filter_nw(f,yr)
    else:
      return x
    
    y = fun.interpolate(yr, scale_factor=2, mode='bilinear',align_corners=False)
    y = self.filter_stage_ups(features,y,x)
    return y

class EncoderDecoder(nn.Module):
  def __init__(self, model_config, size_in, dec_quant, depth=0):
    super(EncoderDecoder, self).__init__()

    enc_config = model_config['encoder_stages']
    dec_config = model_config['decoder_stages']

    enc_size = enc_config[depth][0]
    num_conv = enc_config[depth][1]

    quantize, num_bits_act = get_quant_settings(model_config)
    enc_quant = TrainedQuantUnsigned(quantize, num_bits_act)

    depth_n = depth + 1

    if depth_n == len(enc_config):
      self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv,quantizer=dec_quant,depth=depth)
      self.encoder_decoder = None
      return
    else:
      self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv,quantizer=enc_quant,depth=depth)

    enc_size_next = enc_config[depth_n][0]
    dec_size = dec_config[depth][0]
    num_conv = dec_config[depth][1]

    self.encoder_decoder = EncoderDecoder(model_config,enc_size, enc_quant, depth_n)

    dec_size_in = enc_size + enc_size_next
    self.decoder_stage = DecoderStage(model_config,dec_size_in,dec_size,num_conv=num_conv,quantizer=dec_quant)
    return

  def forward(self, x):
    x = self.encoder_stage(x)

    if self.encoder_decoder:
      yr = fun.max_pool2d(x, kernel_size=2)
      yr = self.encoder_decoder(yr)
    else:
      return [x]
    
    y = fun.interpolate(yr[-1], scale_factor=2, mode='nearest',align_corners=None)
    xy = torch.cat([x,y],dim=1)
    y = self.decoder_stage(xy)

    yr.append(y)
    return yr
   

class WNet(nn.Module):
  def __init__(self, model_config, size_in):
    super(WNet, self).__init__()

    self.shuffle_size = 1

    dec_config   = model_config['decoder_stages']
    out_channels = dec_config[0][0]

    self.quantize, num_bits_act = get_quant_settings(model_config)
    self.input_quant = TrainedQuantUnsigned(self.quantize, num_bits=8, max_val=0.5)
    dec_quant = TrainedQuantUnsigned(self.quantize, num_bits_act)

    if "pix_shuffle_size" in model_config:
      self.shuffle_size = model_config["pix_shuffle_size"]
      shuffle_size = self.shuffle_size ** 2

      if out_channels % shuffle_size != 0:
        sys.exit('Decoder output not a multiple of pixel shuffle size')

      enc_size_in = size_in * shuffle_size
    else:
      enc_size_in  = size_in

    self.encoder_decoder = EncoderDecoder(model_config,enc_size_in,dec_quant)
    self.filter_nw = FilterNetwork(model_config,size_in,shuffle_size=self.shuffle_size)

  def forward(self, x):

    if self.shuffle_size > 1:
      x1 = space_to_depth(x, block_size=self.shuffle_size)
      fs = self.encoder_decoder(x1)
    else:
      x1 = x
      if self.quantize:
        x1 = self.input_quant(x)
      fs = self.encoder_decoder(x1)

    y = self.filter_nw(fs,x)
    return y
