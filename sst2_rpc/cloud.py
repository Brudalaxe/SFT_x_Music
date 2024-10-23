# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/10/25
-----------------------------------
"""
import os
import torch.distributed.rpc as rpc

from model import DistBertDecomposition, DistBertOri
from config import args

print("Cloud process initialised")

rpc.init_rpc("cloud", rank=1, world_size=2)

print("Waiting for requests from the edge process...")

rpc.shutdown()

print("Training is complete")