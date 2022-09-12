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
import torch.distributed as dist

class AvgMetric:
  def __init__(self, rank=0, num_gpus=1):
    self.value = torch.tensor(0.0).cuda(rank)
    self.num_acc = 0.0
    self.num_gpus = num_gpus
    self.avg = 0.0

  def clear():
    self.value = 0
    self.num_acc = 0

  def __call__(self, x):
    self.value += x
    self.num_acc += 1

  def get_value(self):
    return self.avg 

  def eval(self):
    avg_val = self.value / self.num_acc
    if self.num_gpus > 1:
      dist.all_reduce(avg_val, op=dist.ReduceOp.SUM)
      avg_val = avg_val / self.num_gpus
    
    self.avg = avg_val.item()
    return self.avg 

