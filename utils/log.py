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

from tqdm import tqdm
from collections import OrderedDict
from utils.metrics import *
import torchvision
from torch.utils.tensorboard import SummaryWriter

def rank_enabled(rank,enable_ranks):
  for enable_rank in enable_ranks:
    if enable_rank == rank:
      return True
  return False

def conv_dist_checkpoint(checkpoint):
  checkpoint_conv = OrderedDict()
  for k, v in checkpoint.items():
    if not k[0:6] == "module":
      return checkpoint
    name = k[7:] # remove `module.`
    checkpoint_conv[name] = v
  return checkpoint_conv

class Progress():
  def __init__(self,num_batches,epoch,rank=0,enable_ranks=[0]):
    self.enable = rank_enabled(rank,enable_ranks)

    if (self.enable):
      self.bar = tqdm(desc="Epoch:"+str(epoch),total=num_batches,ascii=True,unit=' steps',bar_format='{l_bar}{bar:40}{r_bar}')

  def step(self,training_loss,increment=1):  
    if (self.enable):
      self.bar.set_postfix(loss=training_loss)
      self.bar.update(increment)

  def step_val(self,training_loss,val_loss,increment=1):
    if (self.enable):
      self.bar.set_postfix(loss=float(training_loss),val_loss=float(val_loss))
      self.bar.update(increment)

  def close(self):
    if (self.enable):
      self.bar.close()

class Printer():
  def __init__(self,rank=0,enable_ranks=[0]):
    self.rank = rank
    self.enable_print = rank_enabled(rank,enable_ranks)
    self.print_rank = False

    if len(enable_ranks) > 1:
      self.print_rank = True

  def __call__(self, *args):
    if self.enable_print:
      if self.print_rank:
        print("rank:",rank,*args)
      else:
        print(*args)

class TBWriter():
  def __init__(self,log_dir,rank=0,enable_ranks=[0]):
    self.enabled = rank_enabled(rank,enable_ranks)

    if self.enabled:
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      self.writer = SummaryWriter(str(log_dir))  

  def add_scalar(self,*args,**kwargs):
    if self.enabled:
      self.writer.add_scalar(*args,**kwargs)

  def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
    if self.enabled:
      self.writer.add_image(tag, torchvision.utils.make_grid(img_tensor), global_step, walltime, dataformats)

  def add_graph(self, model, input_to_model=None, verbose=False):
    if self.enabled:
      self.writer.add_graph(model, input_to_model, verbose)

class Checkpoint():
  def __init__(self,checkpoint_path,num_gpus=1,rank=0,enable_ranks=[0]):
    self.save_enabled = rank_enabled(rank,enable_ranks)
    self.checkpoint_path = checkpoint_path
    self.rank = rank
    self.dist = num_gpus > 1

    if self.save_enabled:
      if not os.path.exists(checkpoint_path.parent):
        os.makedirs(checkpoint_path.parent)

  def load(self,model,optimizer):
    start_epoch = 0
    loss = float("inf")

    # All processes load checkpoint
    if self.checkpoint_path.exists():
      loc = 'cuda:{}'.format(self.rank)
      checkpoint = torch.load(self.checkpoint_path, map_location=loc)

      model_cp = checkpoint['model_state_dict']

      if not self.dist:
        model_cp = conv_dist_checkpoint(model_cp)

      model.load_state_dict(model_cp)
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

      start_epoch = checkpoint['epoch']
      loss        = checkpoint['loss']

      if self.save_enabled:
        print('loading checkpoint for epoch ' + str(start_epoch))
    else:
      if self.save_enabled:
        print('no previous checkpoints found')

    return (start_epoch, loss)

  def save(self, epoch, loss, model, optimizer):
    if self.save_enabled:
      torch.save({'epoch'               : epoch,
                  'loss'                : loss,
                  'model_state_dict'    : model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, str(self.checkpoint_path))


