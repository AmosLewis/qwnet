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
import math as math
from torch.optim.lr_scheduler import LambdaLR


# step can be per epoch or a batch. The user controls the update by calling scheduler.step()
def flat():
  def lr_schedule(step):
    return 1.0
  return  lr_schedule

def flat_linear(lr, final_lr, start_step, falloff_steps):
  final_ratio = final_lr / lr
  delta = (1.0 - final_ratio) / falloff_steps

  def lr_schedule(step):
    if step > start_step:
      return max(final_ratio, 1.0 - (step - start_step) * delta)
    else:
      return 1.0
  return  lr_schedule

def step(lr, final_lr, steps):
  final_ratio = final_lr / lr
  def lr_schedule(step):
    if step > steps:
      return final_ratio
    else:
      return 1.0
  return  lr_schedule

def clamped_exp(lr, final_lr, steps):
  final_ratio = final_lr / lr
  falloff = math.exp(math.log(final_ratio) / steps)

  def lr_schedule(step):
    return max(final_ratio, falloff ** step)
  return  lr_schedule

def ramp(steps):
  step_size = 1/steps
  def lr_schedule(step):
    return min(1.0, step * step_size)
  return  lr_schedule

def get_scheduler(train_config, optimizer):
  lr = float(train_config['lr'])

  if 'schedule' in train_config:
    schedule = train_config['schedule']

    if schedule == 'clamped_exp':
      final_lr    = train_config['final_lr']
      steps       = train_config['steps']
      lr_schedule = clamped_exp(lr, final_lr, steps)

    elif schedule == 'ramp':
      steps       = train_config['steps']
      lr_schedule = ramp(steps)

    elif schedule == 'step':
      final_lr    = train_config['final_lr']
      start_step  = train_config['start_step']
      lr_schedule = step(lr, final_lr, start_step)

    elif schedule == 'flat_linear':
      final_lr    = train_config['final_lr']
      start_step  = train_config['start_step']
      steps       = train_config['steps']
      lr_schedule = flat_linear(lr, final_lr, start_step, steps)

    else:
      lr_schedule = flat()
  else:
    lr_schedule = flat()

  return LambdaLR(optimizer, lr_lambda=[lr_schedule])
