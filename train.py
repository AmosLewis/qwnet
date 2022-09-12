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

import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.multiprocessing as mpt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import training.optimizer as opt
import training.schedule as sch

from utils.metrics      import *
from utils.input        import *
from model.recurrent    import *
from model.single_frame import *

import utils.log as log

RESOLUTION = (1080,1920)
CHANNELS = 3
UPDATE_INTERVAL = 8

def train(rank, args, train_paths, val_paths, num_gpus):
  torch.cuda.set_device(rank)
  dist_print = log.Printer(rank)
  print("train_paths: ", train_paths)

  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  if num_gpus > 1:
    dist.init_process_group("nccl", rank=rank, world_size=num_gpus, init_method='env://')

  batch_size = args['batch_size']
  num_train_batches = len(train_paths) // batch_size
  num_val_batches = len(val_paths) // batch_size
  num_batches = num_train_batches + num_val_batches

  # load one image to get dimensions
  sample_image = exr.open(str(train_paths[0][0]))
  height = sample_image.height

  time_steps = sample_image.height // sample_image.width
  image_shape = (CHANNELS, sample_image.width, sample_image.width)

  block_dims = (time_steps,) +  image_shape

  # make dataset
  train_dataset = StreamDataset(train_paths,block_dims, RESOLUTION)
  val_dataset   = StreamDataset(val_paths  ,block_dims, RESOLUTION)

  train_sampler = DistributedSampler(train_dataset,num_replicas=num_gpus,rank=rank)
  val_sampler   = DistributedSampler(val_dataset  ,num_replicas=num_gpus,rank=rank,shuffle=False)

  num_workers   = int(args['num_workers'])
  train_loader  = DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler,shuffle=False,pin_memory=True,num_workers=num_workers)
  val_loader    = DataLoader(val_dataset  ,batch_size=batch_size,sampler=val_sampler  ,shuffle=False,pin_memory=True,num_workers=num_workers)

  # setup model
  
  model = RecurrentTrain(args, image_shape, time_steps)

  dist_print('-------------------------------------------Network----------------------------------------------')
  dist_print(model)
  dist_print('------------------------------------------------------------------------------------------------')

  model.cuda(rank)

  dist_print("Training batches:",num_train_batches,"| Validation batches:",num_val_batches)

  # jit model
  if args['jit']:
    dist_print('Jitting Model')
    net_channels = CHANNELS * 2 + 2 # current, prev and velocity
    ip_size = (1, net_channels, time_steps) +  image_shape[1:]
    dummy_batch = torch.ones(ip_size).cuda()
    model = torch.jit.trace(model,dummy_batch)

  if num_gpus > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  
  # optimizer
  train_config = args['train_config']
  optimizer = opt.get_optimizer(train_config, model.parameters())
  scheduler = sch.get_scheduler(train_config, optimizer)
  per_batch_schedule = train_config['per_batch_schedule']

  # Checkpoint
  if args['session_name'] is None:
    base_path = Path(args['checkpoint_dir'])
  else:
    base_path = Path(args['checkpoint_dir'])/Path(args['session_name'])

  checkpoint_file      = Path('cp')
  checkpoint_file_min  = Path('cp_min')

  checkpoint_path     = base_path/checkpoint_file
  checkpoint_path_min = base_path/checkpoint_file_min
  
  checkpoint     = log.Checkpoint(checkpoint_path     ,num_gpus,rank=rank)
  checkpoint_min = log.Checkpoint(checkpoint_path_min ,num_gpus,rank=rank)

  if args['load'] == 'min':
    (start_epoch,epoch_loss) = checkpoint_min.load(model,optimizer)
  else:
    (start_epoch,epoch_loss) = checkpoint.load(model,optimizer)

  # Barrier before any further writes to this checkpoint
  if num_gpus > 1:
    dist.barrier()

  # Tensorboard
  if args['session_name'] is None:
    log_dir = Path(args['log_dir'])
  else:
    log_dir = Path(args['log_dir'])/Path(args['session_name'])

  tb_writer = log.TBWriter(log_dir,rank=rank)

  # Train Loop
  num_epochs = int(args['epochs'])
  batch_count = 0
  log_image = False

  if 'log_image' in args:
    log_image = args['log_image']

  for epoch in range(start_epoch,num_epochs):
    pbar = log.Progress(num_batches,epoch,rank=rank)

    train_sampler.set_epoch(epoch)
    model.train()

    spatial_loss_avg = AvgMetric(rank,num_gpus)
    train_loss_avg   = AvgMetric(rank,num_gpus)
    val_loss_avg     = AvgMetric(rank,num_gpus)

    for batch in train_loader:
      optimizer.zero_grad()
      batch = batch.cuda(rank, non_blocking=True)
      (out, loss, spatial_loss) = model(batch)
      loss.backward()
      optimizer.step()

      # optionally reduce and log loss every step in tensorboard
      # only valid for LR testing and not for regular training
      if per_batch_schedule:
        if num_gpus > 1:
          dist.all_reduce(loss, op=dist.ReduceOp.SUM)
          loss_avg = loss / num_gpus
        tb_writer.add_scalar('training loss',loss_avg,batch_count)
        scheduler.step()

      # per GPU average loss
      train_loss_avg(loss)
      spatial_loss_avg(spatial_loss)

      # all_reduce average loss every N steps for console log
      if not per_batch_schedule and (batch_count % UPDATE_INTERVAL)== 0:
        train_loss_avg.eval()

      pbar.step(train_loss_avg.get_value(),increment=num_gpus)
      batch_count += 1

    # reduce and log average training loss at epoch end for tensorboard
    if not per_batch_schedule:
      tb_writer.add_scalar('training loss',train_loss_avg.eval(),epoch)
      tb_writer.add_scalar('training loss spatial',spatial_loss_avg.eval(),epoch)
      if log_image: 
        tb_writer.add_image('Output/image/color', out, epoch)
      scheduler.step()

    model.eval()
    with torch.no_grad():
      for batch in val_loader:
        batch = batch.cuda(rank, non_blocking=True)
        (_, val_loss, spatial_loss) = model(batch)
        val_loss_avg(val_loss)

        if (batch_count % UPDATE_INTERVAL) == 0:
          val_loss_avg.eval()

        pbar.step_val(train_loss_avg.get_value(),val_loss_avg.get_value(),increment=num_gpus)
        batch_count += 1

    pbar.close()
    # reduce and log average val loss at epoch end for tensorboard
    tb_writer.add_scalar('validation loss',val_loss_avg.eval(),epoch)

    checkpoint.save(epoch+1, train_loss_avg.get_value(), model, optimizer)

    if train_loss_avg.get_value() < epoch_loss:
      epoch_loss = train_loss_avg.get_value()
      checkpoint_min.save(epoch+1, train_loss_avg.get_value(), model, optimizer)

  if num_gpus > 1:
    dist.destroy_process_group()

if __name__ == '__main__': 
  ap = argparse.ArgumentParser()
  ap.add_argument('--image_dir'         , default='./image_dir/'                                , help="Path to directory with noaa, ssaa, vel sub directories")
  ap.add_argument('--val_dir'         , default='./val_dir/'                  , required=False, help="Path to validation directory with noaa, ssaa, vel sub directories")
  ap.add_argument('--epochs'          , default=2000                  , required=False, help="Number of epochs")
  ap.add_argument('--log_dir'         , default='./logs'           , required=False, help="log directory")
  ap.add_argument('--session_name'    , default=None                , required=False, help="name for this training session")
  ap.add_argument('--model'           , default='unet'              , required=False, help="frnn_staged or frnn_direct")
  ap.add_argument('--config'          , default='configs/train.json', required=False, help="config file")
  ap.add_argument('--load'            , default='last'              , required=False, help="load checkpoints from epoch: last, min (min loss) or init (previous initialization)")
  ap.add_argument('--checkpoint_dir'  , default='./checkpoints'    , required=False, help="checkpoint directory")
  ap.add_argument('--num_workers'     , default=4                   , required=False, help="number of CPU workers for each GPU")
  ap.add_argument('--jit'             , default=False               , required=False, help="jit the model", action='store_true')
  ap.add_argument('--num_gpus'        , default=0                   ,required=False , help="Force number of GPUs (N > 0)")

  mpt.set_start_method('spawn')
  torch.manual_seed(524287) # seeding for approximate repeatibility 

  args = vars(ap.parse_args())

  num_gpus = 0
  if int(args['num_gpus']) > 0:
    num_gpus = int(args['num_gpus'])
  else:
    num_gpus = torch.cuda.device_count()
  print("Num GPUs:",num_gpus)
  print(torch.cuda.get_device_name(0))

  # load json config
  with open(args['config']) as f:
    config = json.load(f)
  args.update(config)

  batch_size = args['batch_size']
  image_dir  = Path(args['image_dir'])
  print("image_dir: ", image_dir)

  # setup training and validation image paths
  _, train_paths = make_path_lists(image_dir,num_gpus=num_gpus,batch_size=batch_size)

  if args['val_dir']:
    _, val_paths = make_path_lists(args['val_dir'],num_gpus=num_gpus,batch_size=batch_size)
  else:
    val_paths = []

  #train(0,args,train_paths,val_paths,num_gpus)

  mpt.spawn(train,args=(args,train_paths,val_paths,num_gpus),nprocs=num_gpus,join=True)
  
