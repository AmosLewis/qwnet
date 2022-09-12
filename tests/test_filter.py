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
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as fun
from common import *


input = torch.randn(1, 3, 4, 4)
print(input)

filter = torch.ones(1, 9, 4, 4)
output = convole(input, filter)
print(output)