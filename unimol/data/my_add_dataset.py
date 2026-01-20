# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache

from unicore.data import BaseWrapperDataset


class PrependTokenDataset2D(BaseWrapperDataset):

    def __init__(self, dataset, token=None, special_token=None):
        super().__init__(dataset)
        self.token = token
        self.special_token = special_token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([torch.full_like(item[0], self.token).unsqueeze(0), item], dim=0)
            item = torch.cat([torch.full_like(item[:,0], self.token).unsqueeze(1), item], dim=1)
        if self.special_token is not None:
            item[0,0] = self.special_token
        return item
    

class AppendTokenDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, token=None, special_token=None):
        super().__init__(dataset)
        self.token = token
        self.special_token = special_token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([item, torch.full_like(item[0], self.token).unsqueeze(0)], dim=0)
            item = torch.cat([item, torch.full_like(item[:,0], self.token).unsqueeze(1)], dim=1)
        if self.special_token is not None:
            item[-1,-1] = self.special_token
        return item
