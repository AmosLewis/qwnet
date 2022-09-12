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

def sample_nn(image, uv):
  uv = 2 * uv - 1
  warped_image = fun.grid_sample(image, uv, mode='nearest', align_corners=False)
  return warped_image

def sample_bicubic_16(image, uv, size):
  xy = uv * size
  tc = torch.floor( xy - 0.5 ) + 0.5
  f  = xy - tc

  f2 = f * f;
  f3 = f2 * f;

  w0 =  f2 - 0.5 * (f3 + f)
  w1 =  1.5 * f3 - 2.5 * f2 + 1.0
  w2 = -1.5 * f3 + 2 * f2 + 0.5 * f
  w3 =  0.5 * (f3 - f2)

  tc0 = tc - 1
  tc1 = tc
  tc2 = tc + 1
  tc3 = tc + 2

  invSize = torch.reciprocal(size)
  tc0 *= invSize;
  tc1 *= invSize;
  tc2 *= invSize;
  tc3 *= invSize;

  res  = sample_nn(image, torch.stack((tc0[...,0],tc0[...,1]),dim=-1)) * w0[...,0] * w0[...,1]
  res += sample_nn(image, torch.stack((tc0[...,0],tc1[...,1]),dim=-1)) * w0[...,0] * w1[...,1]
  res += sample_nn(image, torch.stack((tc0[...,0],tc2[...,1]),dim=-1)) * w0[...,0] * w2[...,1]
  res += sample_nn(image, torch.stack((tc0[...,0],tc3[...,1]),dim=-1)) * w0[...,0] * w3[...,1]
  res += sample_nn(image, torch.stack((tc1[...,0],tc0[...,1]),dim=-1)) * w1[...,0] * w0[...,1]
  res += sample_nn(image, torch.stack((tc1[...,0],tc1[...,1]),dim=-1)) * w1[...,0] * w1[...,1]
  res += sample_nn(image, torch.stack((tc1[...,0],tc2[...,1]),dim=-1)) * w1[...,0] * w2[...,1]
  res += sample_nn(image, torch.stack((tc1[...,0],tc3[...,1]),dim=-1)) * w1[...,0] * w3[...,1]
  res += sample_nn(image, torch.stack((tc2[...,0],tc0[...,1]),dim=-1)) * w2[...,0] * w0[...,1]
  res += sample_nn(image, torch.stack((tc2[...,0],tc1[...,1]),dim=-1)) * w2[...,0] * w1[...,1]
  res += sample_nn(image, torch.stack((tc2[...,0],tc2[...,1]),dim=-1)) * w2[...,0] * w2[...,1]
  res += sample_nn(image, torch.stack((tc2[...,0],tc3[...,1]),dim=-1)) * w2[...,0] * w3[...,1]
  res += sample_nn(image, torch.stack((tc3[...,0],tc0[...,1]),dim=-1)) * w3[...,0] * w0[...,1]
  res += sample_nn(image, torch.stack((tc3[...,0],tc1[...,1]),dim=-1)) * w3[...,0] * w1[...,1]
  res += sample_nn(image, torch.stack((tc3[...,0],tc2[...,1]),dim=-1)) * w3[...,0] * w2[...,1]
  res += sample_nn(image, torch.stack((tc3[...,0],tc3[...,1]),dim=-1)) * w3[...,0] * w3[...,1]
  return res


def catmull_rom_params(uv, size, invSize):
  xy = uv * size

  tc = torch.floor(xy - 0.5) + 0.5
  f = xy - tc
  f2 = f * f
  f3 = f2 * f

  w0 = f2 - 0.5 * (f3 + f)
  w1 = 1.5 * f3 - 2.5 * f2 + 1
  w3 = 0.5 * (f3 - f2)
  w2 = 1 - w0 - w1 - w3

  ww0 = w0
  ww1 = w1 + w2
  ww2 = w3

  s0 = tc - 1
  s1 = tc + w2 / ww1
  s2 = tc + 2

  s0 *= invSize;
  s1 *= invSize;
  s2 *= invSize;
  return (s0,s1,s2,ww0,ww1,ww2)

def sample_bilinear(image, uv):
  uv = 2 * uv - 1
  warped_image = fun.grid_sample(image, uv, align_corners=False)
  return warped_image

def sample_bicubic(image, uv, size):
  invSize = torch.reciprocal(size)
  s0,s1,s2,w0,w1,w2 = catmull_rom_params(uv, size, invSize)

  sample_uv0 = torch.stack([s1[...,0], s0[...,1]],dim=3)
  sample_uv1 = torch.stack([s0[...,0], s1[...,1]],dim=3)
  sample_uv2 = torch.stack([s1[...,0], s1[...,1]],dim=3)
  sample_uv3 = torch.stack([s2[...,0], s1[...,1]],dim=3)
  sample_uv4 = torch.stack([s1[...,0], s2[...,1]],dim=3)

  sample_w0 = (w1[...,0] * w0[...,1]).unsqueeze(dim=1)
  sample_w1 = (w0[...,0] * w1[...,1]).unsqueeze(dim=1)
  sample_w2 = (w1[...,0] * w1[...,1]).unsqueeze(dim=1)
  sample_w3 = (w2[...,0] * w1[...,1]).unsqueeze(dim=1)
  sample_w4 = (w1[...,0] * w2[...,1]).unsqueeze(dim=1)

  cornerWeights = sample_w0 + sample_w1 + sample_w2 + sample_w3 + sample_w4;
  finalMultiplier = torch.reciprocal(cornerWeights);

  outColor0 = sample_bilinear(image, sample_uv0)
  outColor1 = sample_bilinear(image, sample_uv1)
  outColor2 = sample_bilinear(image, sample_uv2)
  outColor3 = sample_bilinear(image, sample_uv3)
  outColor4 = sample_bilinear(image, sample_uv4)

  outColor = (outColor0 * sample_w0 + 
              outColor1 * sample_w1 + 
              outColor2 * sample_w2 + 
              outColor3 * sample_w3 + 
              outColor4 * sample_w4 ) * finalMultiplier
  outColor  = outColor.clamp(min=0)

  return outColor


class Warp(nn.Module):
  # Have to explicitly suppot these methods 
  # since we do not register these buffers in the checkpoint
  def half(self,*args,**kwargs):
    self.grid_x = self.grid_x.half(*args,**kwargs)
    self.grid_y = self.grid_y.half(*args,**kwargs)
    self.size   = self.size.half(*args,**kwargs)

  def cuda(self,*args,**kwargs):
    self.grid_x = self.grid_x.cuda(*args,**kwargs)
    self.grid_y = self.grid_y.cuda(*args,**kwargs)
    self.size   = self.size.cuda(*args,**kwargs)

  def to(self,*args,**kwargs):
    self.grid_x = self.grid_x.to(*args,**kwargs)
    self.grid_y = self.grid_y.to(*args,**kwargs)
    self.size   = self.size.to(*args,**kwargs)

  def __init__(self, image_shape):
    super(Warp, self).__init__()

    (channels, height, width) = image_shape

    self.size = torch.tensor([width,height], dtype=torch.float32)

    x = torch.arange(width , dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(y , x)
    self.grid_x = (grid_x + 0.5) / width
    self.grid_y = (grid_y + 0.5) / height

  def __call__(self, image_prev, vel_image):
    grid_x = self.grid_x - vel_image[:,0,...] / self.size[0]
    grid_y = self.grid_y - vel_image[:,1,...] / self.size[1]
    grid = torch.stack([grid_x, grid_y],dim=3)

    warped_image = sample_bilinear(image_prev, grid)
    return warped_image


class WarpBicubic(Warp):

  def __init__(self, image_shape):
    super(WarpBicubic, self).__init__(image_shape)

  def __call__(self, image_prev, vel_image):
    grid_x = self.grid_x - vel_image[:,0,...] / self.size[0]
    grid_y = self.grid_y - vel_image[:,1,...] / self.size[1]
    grid = torch.stack([grid_x, grid_y],dim=3)

    warped_image = sample_bicubic(image_prev, grid, self.size)
    return warped_image
