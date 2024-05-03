# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/9/5
-----------------------------------
"""
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import args, logger
from model import Classification, SvdBlock
from data import TextClassificationDataset


device = torch.device(f"cuda:{args.cuda}")

train_data = TextClassificationDataset(args.pretrain_dir)
train_data.build_label_index()
train_data.read_file(args.train_path, max_length=args.max_length)
train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=train_data.collate, shuffle=True)

dev_data = TextClassificationDataset(args.pretrain_dir)
dev_data.build_label_index()
dev_data.read_file(args.dev_path, max_length=args.max_length)
dev_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=dev_data.collate)

train_steps = len(train_loader) * args.epoch
warmup_steps = train_steps * args.warm_up_pct

classifier = Classification(args.pretrain_dir, train_data.label_nums)
classifier.to(device)
if args.split != 0:
    layer = classifier.bert.encoder.layer[args.split-1]
    classifier.bert.encoder.layer[args.split-1] = SvdBlock(layer, rank=args.rank)

criterion = nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}]

opt = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, train_steps)

for e in range(args.epoch):
    classifier.train()
    for i, batch in tqdm(enumerate(train_loader), desc="training", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        y = batch["batch_labels"].to(device)

        y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                           attention_mask=batch_attention_mask)

        loss = criterion(y_hat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    classifier.eval()

    dev_loss = 0.0
    right_num = 0
    for batch in tqdm(train_loader, desc="eval train data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        y = batch["batch_labels"].to(device)

        with torch.no_grad():
            y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                               attention_mask=batch_attention_mask)
            loss = criterion(y_hat, y)
            pred = torch.argmax(y_hat, dim=-1)
            loss = loss.detach().cpu().item()
            right = torch.eq(y, pred).sum().detach().item()
        dev_loss += loss * len(batch)
        right_num += right

    all_num = len(train_data)
    logger.info(f"epoch: {e} train set loss: {dev_loss / all_num} acc: {right_num / all_num}")

    dev_loss = 0.0
    right_num = 0
    for batch in tqdm(dev_loader, desc="eval dev data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        y = batch["batch_labels"].to(device)

        with torch.no_grad():
            y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                               attention_mask=batch_attention_mask)
            loss = criterion(y_hat, y)
            pred = torch.argmax(y_hat, dim=-1)
            loss = loss.detach().cpu().item()
            right = torch.eq(y, pred).sum().detach().item()
        dev_loss += loss * len(batch)
        right_num += right

    all_num = len(dev_data)
    logger.info(f"epoch: {e} dev set loss: {dev_loss / all_num} acc: {right_num / all_num}")

    # torch.save(classifier.state_dict(), args.save_path)
