# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/9/5
-----------------------------------
"""
from tqdm import tqdm
import json

import torch
from transformers import BertTokenizerFast
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from config import args, logger


class TextClassificationDataset(Dataset):
    def __init__(self, pretrain_dir):
        super().__init__()
        self.sample = []
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.labels = []

        self.tokenizer = BertTokenizerFast.from_pretrained(pretrain_dir)
        self.label2id = {}
        self.id2label = {}

    def build_label_index(self):
        self.label2id['0'] = 0
        self.label2id['1'] = 1
        self.id2label[0] = "0"
        self.id2label[1] = "1"

    def read_file(self, data_path, max_length, need_label=True):
        data = []
        with open(data_path, encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                data.append(line)

        error_line_nums = 0
        for line in tqdm(data, desc="processing data", leave=False):
            _, label, __, s = line.split('\t')
            if label not in self.label2id:
                error_line_nums += 1
                continue

            encode_text = self.tokenizer(s)
            if len(encode_text["input_ids"]) > max_length:
                continue

            encode_text = self.tokenizer.pad(encode_text,
                                             padding="max_length",
                                             max_length=max_length,
                                             return_tensors="pt")

            self.input_ids.append(encode_text['input_ids'])
            self.token_type_ids.append(encode_text['token_type_ids'])
            self.attention_mask.append(encode_text['attention_mask'])
            self.sample.append(line)
            if need_label:
                self.labels.append(torch.tensor([self.label2id[label]], dtype=torch.int64))
            else:
                self.labels.append(torch.tensor([-1], dtype=torch.int64))

        logger.info(f"共{error_line_nums}条数据标签错误")

    def __getitem__(self, idx):
        return {
            "sample": self.sample[idx],
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.token_type_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

    @property
    def label_nums(self):
        return len(self.label2id)

    def __len__(self):
        return len(self.sample)

    @staticmethod
    def collate(batch_data):
        batch_input_ids = torch.stack([data["input_ids"] for data in batch_data], dim=0)
        batch_token_type_ids = torch.stack([data["token_type_ids"] for data in batch_data], dim=0)
        batch_attention_mask = torch.stack([data["attention_mask"] for data in batch_data], dim=0)
        batch_labels = torch.concat([data["labels"] for data in batch_data], dim=0)
        batch_sample = [data["sample"] for data in batch_data]

        return {
            "batch_input_ids": batch_input_ids,
            "batch_token_type_ids": batch_token_type_ids,
            "batch_attention_mask": batch_attention_mask,
            "batch_labels": batch_labels,
            "batch_sample": batch_sample,
        }


if __name__ == '__main__':
    dev_data = TextClassificationDataset(args.pretrain_dir)
    dev_data.build_label_index()
    dev_data.read_file(args.dev_path, max_length=30)
    dev_loader = DataLoader(dev_data, batch_size=3, collate_fn=dev_data.collate)

    # print(len(dev_data))
    # print(len(train_loader))

    for batch in dev_loader:
        print(batch)
        break
