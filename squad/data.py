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


class QADataset(Dataset):
    def __init__(self, pretrain_dir):
        super().__init__()
        self.sample = []
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.offsets = []
        self.start_labels = []
        self.end_labels = []

        self.tokenizer = BertTokenizerFast.from_pretrained(pretrain_dir)

    def read_file(self, data_path, max_length, need_label=True):
        data = []

        with open(data_path, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]
                        data.append(
                            {
                                "title": title,
                                "context": context,
                                "question": qa["question"],
                                "id": qa["id"],
                                "answers": {"answer_start": answer_starts, "text": answers, },
                            }
                        )

        for line in tqdm(data, desc="processing data", leave=False):

            question = line["question"]
            context = line["context"]
            answers = line["answers"]
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0]) - 1

            encode_text = self.tokenizer(question,
                                         text_pair=context,
                                         padding="max_length",
                                         max_length=max_length,
                                         truncation="only_second",
                                         return_offsets_mapping=True,
                                         return_tensors="pt")

            if need_label:
                start_token = encode_text.char_to_token(start_char, sequence_index=1)
                end_token = encode_text.char_to_token(end_char, sequence_index=1)
                if not start_token or not end_token:
                    continue
            else:
                start_token = -1
                end_token = -1

            self.input_ids.append(encode_text['input_ids'])
            self.token_type_ids.append(encode_text['token_type_ids'])
            self.attention_mask.append(encode_text['attention_mask'])
            self.offsets.append(encode_text["offset_mapping"])
            self.sample.append(line)

            self.start_labels.append(torch.tensor(start_token))
            self.end_labels.append(torch.tensor(end_token))

    def __getitem__(self, idx):
        return {
            "sample": self.sample[idx],
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.token_type_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "offsets": self.offsets[idx],
            "start_labels": self.start_labels[idx],
            "end_labels": self.end_labels[idx],
        }

    def __len__(self):
        return len(self.sample)

    def convert_for_eval(self):
        references = [{
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [{"text": answer_text} for answer_text in line["answers"]["text"]],
                            "id": line["id"],
                        }
                        for line in self.sample
                    ]
                }
            ]
        }]
        return references

    @staticmethod
    def collate(batch_data):
        batch_input_ids = torch.concat([data["input_ids"] for data in batch_data], dim=0)
        batch_token_type_ids = torch.concat([data["token_type_ids"] for data in batch_data], dim=0)
        batch_attention_mask = torch.concat([data["attention_mask"] for data in batch_data], dim=0)
        batch_offsets = torch.concat([data["offsets"] for data in batch_data], dim=0)
        batch_start_labels = torch.stack([data["start_labels"] for data in batch_data], dim=0)
        batch_end_labels = torch.stack([data["end_labels"] for data in batch_data], dim=0)
        batch_sample = [data["sample"] for data in batch_data]

        return {
            "batch_input_ids": batch_input_ids,
            "batch_token_type_ids": batch_token_type_ids,
            "batch_attention_mask": batch_attention_mask,
            "batch_offsets": batch_offsets,
            "batch_start_labels": batch_start_labels,
            "batch_end_labels": batch_end_labels,
            "batch_sample": batch_sample,
        }


if __name__ == '__main__':
    # squad_data = load_dataset("squad")
    dev_data = QADataset(args.pretrain_dir)
    dev_data.read_file(args.dev_path, max_length=100)
    dev_loader = DataLoader(dev_data, batch_size=2, collate_fn=dev_data.collate, shuffle=True)

    print(len(dev_data))
    # print(len(train_loader))

    for batch in dev_loader:
        input_ids = batch["batch_input_ids"][0]
        sample = batch["batch_sample"][0]
        offset = batch["batch_offsets"][0]
        start = batch["batch_start_labels"][0]
        end = batch["batch_end_labels"][0]

        print(sample["answers"])
        print(sample["context"][offset[start][0]: offset[end][1] + 1])
        break
