# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/9/5
-----------------------------------
"""
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np

from config import args, logger
from model import QA, SvdBlock
from data import QADataset
from compute_score import compute_score

device = torch.device(f"cuda:{args.cuda}")

train_data = QADataset(args.pretrain_dir)
train_data.read_file(args.train_path, max_length=args.max_length)
train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=train_data.collate, shuffle=True)
logger.info(f"total nums of train set : {len(train_data)}")

dev_data = QADataset(args.pretrain_dir)
dev_data.read_file(args.dev_path, max_length=args.max_length)
dev_loader = DataLoader(dev_data, batch_size=args.dev_batch_size, collate_fn=dev_data.collate)
logger.info(f"total nums of dev set : {len(dev_data)}")

train_steps = len(train_loader) * args.epoch
warmup_steps = train_steps * args.warm_up_pct

classifier = QA(args.pretrain_dir)
classifier.to(device)
if args.split != 0:
    layer = classifier.bert.encoder.layer[args.split - 1]
    classifier.bert.encoder.layer[args.split - 1] = SvdBlock(layer, rank=args.rank)

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
        start = batch["batch_start_labels"].to(device)
        end = batch["batch_end_labels"].to(device)

        start_logits, end_logits = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                                              attention_mask=batch_attention_mask)
        start_logits = start_logits.squeeze()
        end_logits = end_logits.squeeze()

        loss = criterion(start_logits, start) + criterion(end_logits, end)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    classifier.eval()

    dev_loss = 0.0
    predictions = {}
    for batch in tqdm(train_loader, desc="eval train data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        start = batch["batch_start_labels"].to(device)
        end = batch["batch_end_labels"].to(device)

        with torch.no_grad():
            start_logits, end_logits = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                                                  attention_mask=batch_attention_mask)
            start_logits = start_logits.squeeze()
            end_logits = end_logits.squeeze()

            loss = criterion(start_logits, start) + criterion(end_logits, end)

            start_logits = start_logits.detach().cpu().numpy()
            end_logits = end_logits.detach().cpu().numpy()
            for i in range(len(start_logits)):
                token_type_ids = batch["batch_token_type_ids"][i].tolist()
                attention_mask = batch["batch_attention_mask"][i].tolist()
                offset = batch["batch_offsets"][i].tolist()
                context = batch["batch_sample"][i]["context"]
                start_indexes = np.argsort(start_logits[i])[-1: -20 - 1: -1].tolist()
                end_indexes = np.argsort(end_logits[i])[-1: -20 - 1: -1].tolist()
                valid_answer = ""

                max_score = 0
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index > end_index:
                            continue
                        if token_type_ids[start_index] == 0:
                            continue
                        if token_type_ids[end_index] == 0:
                            continue
                        if end_index - start_index > 30:
                            continue
                        if start_logits[i][start_index] + end_logits[i][end_index] < max_score:
                            continue
                        valid_answer = context[offset[start_index][0]: offset[end_index][1]]
                        max_score = start_logits[i][start_index] + end_logits[i][end_index]

                predictions[batch["batch_sample"][i]["id"]] = valid_answer

        dev_loss += loss * len(batch)

    all_num = len(train_data)
    references = train_data.convert_for_eval()
    res = compute_score(references, predictions)
    logger.info(f"epoch: {e} train set loss: {dev_loss / all_num} metric: {res}")

    dev_loss = 0.0
    predictions = {}
    for batch in tqdm(dev_loader, desc="eval dev data", leave=False):
        batch_input_ids = batch["batch_input_ids"].to(device)
        batch_token_type_ids = batch["batch_token_type_ids"].to(device)
        batch_attention_mask = batch["batch_attention_mask"].to(device)
        start = batch["batch_start_labels"].to(device)
        end = batch["batch_end_labels"].to(device)

        with torch.no_grad():
            start_logits, end_logits = classifier(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids,
                                                  attention_mask=batch_attention_mask)
            start_logits = start_logits.squeeze()
            end_logits = end_logits.squeeze()

            loss = criterion(start_logits, start) + criterion(end_logits, end)

            start_logits = start_logits.detach().cpu().numpy()
            end_logits = end_logits.detach().cpu().numpy()
            for i in range(len(start_logits)):
                token_type_ids = batch["batch_token_type_ids"][i].tolist()
                attention_mask = batch["batch_attention_mask"][i].tolist()
                offset = batch["batch_offsets"][i].tolist()
                context = batch["batch_sample"][i]["context"]
                start_indexes = np.argsort(start_logits[i])[-1: -20 - 1: -1].tolist()
                end_indexes = np.argsort(end_logits[i])[-1: -20 - 1: -1].tolist()
                valid_answer = ""

                max_score = 0
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index > end_index:
                            continue
                        if token_type_ids[start_index] == 0:
                            continue
                        if token_type_ids[end_index] == 0:
                            continue
                        if end_index - start_index > 30:
                            continue
                        if start_logits[i][start_index] + end_logits[i][end_index] < max_score:
                            continue
                        valid_answer = context[offset[start_index][0]: offset[end_index][1]]
                        max_score = start_logits[i][start_index] + end_logits[i][end_index]

                predictions[batch["batch_sample"][i]["id"]] = valid_answer

        dev_loss += loss * len(batch)

    all_num = len(dev_data)
    references = dev_data.convert_for_eval()
    res = compute_score(references, predictions)
    logger.info(f"epoch: {e} dev set loss: {dev_loss / all_num} metric: {res}")

    # torch.save(classifier.state_dict(), args.save_path)
