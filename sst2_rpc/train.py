# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/6/15
-----------------------------------
"""
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.distributed.optim")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.c10d_logger")

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc

from config import args, logger
from model import DistBertDecomposition, DistBertOri, DistBertFFN
from data import TextClassificationDataset

devices = {
    "edge": "cpu",
    "cloud": "cuda"
}

rpc.init_rpc("edge", rank=0, world_size=2)

train_data = TextClassificationDataset(args.pretrain_dir)
train_data.build_label_index()
train_data.read_file(args.train_path, max_length=args.max_length)
train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=train_data.collate, shuffle=True)

dev_data = TextClassificationDataset(args.pretrain_dir)
dev_data.build_label_index()
dev_data.read_file(args.dev_path, max_length=args.max_length)
dev_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=dev_data.collate)

logger.info(f" Split layer: {args.split}")
logger.info(f" Rank of decomposition: {args.rank}")

# classifier = DistBertDecomposition(args.cloud_pretrain_dir,
#                                    args.edge_pretrain_dir,
#                                    args.split,
#                                    args.rank,
#                                    train_data.label_nums,
#                                    devices)

# classifier = DistBertFFN(args.cloud_pretrain_dir,
#                          args.edge_pretrain_dir,
#                          args.split,
#                          args.rank,
#                          train_data.label_nums,
#                          devices)

# classifier = DistBertOri(args.cloud_pretrain_dir,
#                          args.edge_pretrain_dir,
#                          args.split,
#                          args.rank,
#                          train_data.label_nums,
#                          devices)

model_choice = input("Choose the model to use (1 for Split Fine-Tuning with Decomposition, 2 for Split Fine-Tuning without Decomposition, 3 for Split Learning): ")

if model_choice == "1":
    classifier = DistBertDecomposition(args.cloud_pretrain_dir,
                                      args.edge_pretrain_dir,
                                      args.split,
                                      args.rank,
                                      train_data.label_nums,
                                      devices)
elif model_choice == "2":
    classifier = DistBertFFN(args.cloud_pretrain_dir,
                            args.edge_pretrain_dir,
                            args.split,
                            args.rank,
                            train_data.label_nums,
                            devices)
elif model_choice == "3":
    classifier = DistBertOri(args.cloud_pretrain_dir,
                            args.edge_pretrain_dir,
                            args.split,
                            args.rank,
                            train_data.label_nums,
                            devices)
else:
    print("Invalid choice. Using default model (Split Fine-Tuning with Decomposition).")
    classifier = DistBertDecomposition(args.cloud_pretrain_dir,
                                      args.edge_pretrain_dir,
                                      args.split,
                                      args.rank,
                                      train_data.label_nums,
                                      devices)

criterion = nn.CrossEntropyLoss()

opt = DistributedOptimizer(AdamW, classifier.parameter_rrefs(), lr=args.learning_rate)

for e in range(args.epoch):
    classifier.train()
    logger.info(" Start of epoch")

    with tqdm(train_loader, desc=f"Epoch {e+1}/{args.epoch} - Training", leave=False) as train_tqdm:
        for i, batch in enumerate(train_tqdm):
            batch_input_ids = batch["batch_input_ids"]
            batch_token_type_ids = batch["batch_token_type_ids"]
            batch_attention_mask = batch["batch_attention_mask"]
            y = batch["batch_labels"].to(devices["edge"])

            with dist_autograd.context() as context_id:
                y_hat = classifier(input_ids=batch_input_ids,
                                   token_type_ids=batch_token_type_ids,
                                   attention_mask=batch_attention_mask).to(devices["edge"])
                loss = criterion(y_hat, y)
                dist_autograd.backward(context_id, [loss])
                opt.step(context_id)
            
            train_tqdm.set_postfix({"Loss": loss.item()})

    logger.info(" End of epoch")
    classifier.eval()

    dev_loss = 0.0
    right_num = 0

    with tqdm(dev_loader, desc=f"Epoch {e+1}/{args.epoch} - Evaluating", leave=False) as dev_tqdm:
        for batch in dev_tqdm:
            batch_input_ids = batch["batch_input_ids"]
            batch_token_type_ids = batch["batch_token_type_ids"]
            batch_attention_mask = batch["batch_attention_mask"]
            y = batch["batch_labels"].to(devices["edge"])

            with torch.no_grad():
                y_hat = classifier(input_ids=batch_input_ids,
                                   token_type_ids=batch_token_type_ids,
                                   attention_mask=batch_attention_mask).to(devices["edge"])
                loss = criterion(y_hat, y)
                pred = torch.argmax(y_hat, dim=-1)
                loss = loss.detach().cpu().item()
                right = torch.eq(y, pred).sum().detach().item()
            dev_loss += loss * len(batch)
            right_num += right

            dev_tqdm.set_postfix({"Loss": loss, "Acc": right / len(batch)})

    all_num = len(dev_data)
    logger.info(f" Epoch: {e}, dev set loss: {dev_loss / all_num}, acc: {right_num / all_num}")

    # torch.save(classifier.state_dict(), args.save_path)

rpc.shutdown()