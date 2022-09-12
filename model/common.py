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
from model.quantize import *

# RGB to Y and CbCr
def rgb2y_cbcr(rgb):
    # rgb shape = (b,c,h,w)
    # rgb in [0, 1]
    # ycbcr in [0, 1]
    r,g,b = torch.split(rgb, rgb.size(1)//3, dim=1)
    y  = r *  0.299  + g * 0.587  + b * 0.114
    cb = r * -0.1687 - g * 0.3313 + b * 0.5
    cr = r *  0.5    - g * 0.4187 - b * 0.0813
    cb += 0.5
    cr += 0.5
    cbcr = torch.cat((cb, cr), dim=1)
    return y, cbcr

# Y and CbCr to RGB
def y_cbcr2rgb(y, cbcr):
    # ycbcr shape = (b,c,h,w)
    # ycbcr in [0, 1]
    # rgb in [0, 1]
    cb, cr = torch.split(cbcr, cbcr.size(1)//2, dim=1)
    cb -= 0.5
    cr -= 0.5
    r = y * 1. + cb * 0.      + cr * 1.402
    g = y * 1. - cb * 0.34414 - cr * 0.71414
    b = y * 1. + cb * 1.772   + cr * 0.
    return torch.cat((r, g, b), dim=1)

# Pad data to be multiple of alignment_size
def pad(x, alignment_size):
    b,c,h,w = x.size()
    mod_h = np.mod(h, alignment_size)
    mod_w = np.mod(w, alignment_size)
    pad_h = alignment_size - mod_h if mod_h != 0 else 0
    pad_w = alignment_size - mod_w if mod_w != 0 else 0
    return fun.pad(x, pad=(0,pad_w, 0,pad_h), mode="reflect") if (pad_w > 0 or pad_h > 0) else x

# Unpad data to output size
def unpad(x, out_size):
    # x shape = b,c,h,w
    return x[:, :, :out_size[0], :out_size[1]]

def space_to_depth(x, block_size):
  b, c, h, w = x.size()
  unfolded_x = fun.unfold(x, block_size, stride=block_size)
  return unfolded_x.view(b, c * block_size ** 2, h // block_size, w // block_size)

def depth_to_space(x, block_size):
  return fun.pixel_shuffle(x, block_size)

def l1_norm(x):
  x_sum = sum(x)
  size = len(x)

  for idx in range(size):
    x[idx] *= (size / x_sum)

  return x

def get_act(act_type, model_config, conv):
  init_type = model_config['init']
  slope = 0.0

  if act_type == "leaky_relu":
    slope = model_config['leaky_slope']
    act = nn.LeakyReLU(negative_slope=slope)
  elif act_type == "prelu":
    slope = model_config['leaky_slope']
    act = nn.PReLU(init=slope)
  elif act_type == "relu":
    act = nn.ReLU()
  elif act_type == "elu":
    act = nn.ELU()
  else:
    act = nn.ReLU()

  init_fn = None
  if init_type == "he_unif":
    init_fn = torch.nn.init.kaiming_uniform_
  elif init_type == "he_norm":
    init_fn = torch.nn.init.kaiming_normal_

  if init_fn:
    if act_type == "leaky_relu" or act_type == "prelu":
      init_fn(conv.weight, a=slope, nonlinearity='leaky_relu')
    else:
      init_fn(conv.weight, nonlinearity='relu')

  return act

def get_quant_settings(model_config):
  if 'quantize' in model_config:
    quantize = model_config['quantize']
  else:
    quantize = False
  if 'num_bits' in model_config:
    num_bits = model_config['num_bits']
  else:
    num_bits = 32
  return quantize, num_bits

class ConvGroup(nn.Module):
  def __init__(self, model_config, size_in, size_out, kernel_size=3, quantizer=None, depth=None):
    super(ConvGroup, self).__init__()

    quantize, num_bits_act = get_quant_settings(model_config)
    num_bits_wts = num_bits_act
    if quantize:
      if depth==0:
        num_bits_wts = 8

    use_bn = model_config['use_bn']
    self.conv = QConv2D(size_in, size_out, quantize=quantize, num_bits=num_bits_wts, kernel_size=kernel_size, bias=not use_bn, padding=kernel_size//2)

    self.bn = None
    if use_bn:
      self.bn = nn.BatchNorm2d(size_out, affine=True, track_running_stats=True)
    else:
      self.conv.bias.data.fill_(0)

    self.act = get_act(model_config['activation'],model_config,self.conv)

    if quantizer:
      self.quantizer = quantizer
    else:
      self.quantizer = TrainedQuantUnsigned(quantize, num_bits_act)

  def forward(self, x):
    x = self.conv(x)
    if self.bn:
      x = self.bn(x)
    x = self.act(x)
    x = self.quantizer(x)
    return x

class DecoderStage(nn.Module):
  def __init__(self, model_config, size_in, size_out, quantizer=None, num_conv=1):
    super(DecoderStage, self).__init__()
    self.conv_groups = nn.ModuleList()

    group = ConvGroup(model_config,size_in,size_out,kernel_size=1,quantizer=None)
    self.conv_groups.append(group)

    for conv_id in range(num_conv):
      if conv_id == num_conv-1:
        group = ConvGroup(model_config,size_out,size_out,kernel_size=3,quantizer=quantizer)
      else:
        group = ConvGroup(model_config,size_out,size_out,kernel_size=3,quantizer=None)
      self.conv_groups.append(group)

  def forward(self, x):
    for group in self.conv_groups:
      x = group(x)
    return x

class EncoderStage(nn.Module):
  def __init__(self, model_config, size_in, size_out, quantizer=None, num_conv=1, depth=0):
    super(EncoderStage, self).__init__()
    self.conv_groups = nn.ModuleList()

    if num_conv == 1:
      group = ConvGroup(model_config, size_in, size_out, kernel_size=3, quantizer=quantizer, depth=depth)
    else:
      group = ConvGroup(model_config, size_in, size_out, kernel_size=3, quantizer=None, depth=depth)
    self.conv_groups.append(group)

    for conv_id in range(num_conv - 1):
      if conv_id == num_conv-2:
        group = ConvGroup(model_config, size_out, size_out, kernel_size=3, quantizer=quantizer)
      else:
        group = ConvGroup(model_config, size_out, size_out, kernel_size=3, quantizer=None)
      self.conv_groups.append(group)

  def forward(self, x):
    for group in self.conv_groups:
      x = group(x)
    return x

class Filter(nn.Module):
  def __init__(self, num_channels, num_filters=1,skip=False):
    super(Filter, self).__init__()
    self.num_channels = num_channels
    self.num_filters  = num_filters
    self.skip         = skip

  def forward(self, k, x, skip=None):
    input_dims = x.size()
    patch_dim = (input_dims[0], self.num_channels, 9) + input_dims[2:4]
    patches_list = []

    for idx in range(self.num_filters):
      x_slice = x[:,idx*self.num_channels : idx*self.num_channels+self.num_channels,...]
      patches = fun.unfold(x_slice,kernel_size=(3,3),padding=1)
      patches = torch.reshape(patches,patch_dim)
      patches_list.append(patches)

    if self.skip:
      skip = skip.unsqueeze(dim=2)
      patches_list.append(skip)

    patches_cat = torch.cat(patches_list,dim=2)
    kernel = k.unsqueeze(dim=1)

    y = patches_cat * kernel
    y = torch.sum(y, dim=2)
    return y
