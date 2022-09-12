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
import torch.optim as optim

from training.ranger import Ranger
#from training.radam import RAdam


def get_optimizer(train_config, parameters):
  lr = float(train_config['lr'])

  if 'optimizer' in train_config:
    optimizer = train_config['optimizer']

    if optimizer == 'ranger':
      optimizer = Ranger(parameters, lr=lr)
      #optimizer = Ranger(parameters, lr=lr, betas=(.9,0.999), eps=1e-7)
    # elif optimizer == 'radam':
    #   optimizer = RAdam(parameters, lr=lr, betas=(.95,0.999), eps=1e-5)
      #optimizer = RAdam(parameters, lr=lr, betas=(.9,0.999), eps=1e-7)
    else:
      optimizer = optim.Adam(parameters, lr=lr)
  else:
    optimizer = optim.Adam(parameters, lr=lr)

  return optimizer