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
import scipy.stats

from config import args, logger
from model import Classification, SvdBlock
from data import TextClassificationDataset

device = torch.device(f"cuda:{args.cuda}")

train_data = TextClassificationDataset(args.pretrain_dir)
train_data.read_file(args.train_path, max_length=args.max_length)
train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=train_data.collate, shuffle=True)

dev_data = TextClassificationDataset(args.pretrain_dir)
dev_data.read_file(args.dev_path, max_length=args.max_length)
dev_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=dev_data.collate)

train_steps = len(train_loader) * args.epoch
warmup_steps = train_steps * args.warm_up_pct

classifier = Classification(args.pretrain_dir, 1)
classifier.to(device)
if args.split != 0:
    layer = classifier.bert.encoder.layer[args.split - 1]
    classifier.bert.encoder.layer[args.split - 1] = SvdBlock(layer, rank=args.rank)

criterion = nn.MSELoss()

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
        y_hat = y_hat.squeeze()

        loss = criterion(y_hat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    classifier.eval()

    dev_loss = 0.0
    y_true = []
    y_pred = []
    for batch in tqdm(train_loader, desc="eval train data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        y = batch["batch_labels"].to(device)

        with torch.no_grad():
            y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                               attention_mask=batch_attention_mask)
            y_hat = y_hat.squeeze()
            loss = criterion(y_hat, y).detach().cpu().item()
            pred = y_hat.detach().cpu().tolist()

        dev_loss += loss * len(batch)
        y_true.extend(batch["batch_labels"].tolist())
        y_pred.extend(pred)

    all_num = len(train_data)
    score = scipy.stats.spearmanr(y_true, y_pred)[0]
    logger.info(f"epoch: {e} train set loss: {dev_loss / all_num} spearmanr: {score}")

    dev_loss = 0.0
    y_true = []
    y_pred = []
    for batch in tqdm(dev_loader, desc="eval dev data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        y = batch["batch_labels"].to(device)

        with torch.no_grad():
            y_hat = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                               attention_mask=batch_attention_mask)
            y_hat = y_hat.squeeze()
            loss = criterion(y_hat, y).detach().cpu().item()
            pred = y_hat.detach().cpu().tolist()

        dev_loss += loss * len(batch)
        y_true.extend(batch["batch_labels"].tolist())
        y_pred.extend(pred)

    all_num = len(dev_data)
    score = scipy.stats.spearmanr(y_true, y_pred)[0]
    logger.info(f"epoch: {e} dev set loss: {dev_loss / all_num} spearmanr: {score}")

    # torch.save(classifier.state_dict(), args.save_path)
