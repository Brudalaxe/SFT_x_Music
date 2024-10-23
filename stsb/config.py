# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/9/5
-----------------------------------
"""
import os
import argparse
import logging
import sys

# system settings
root = os.path.dirname(__file__)
sys.path.append(root)

parser = argparse.ArgumentParser()
# path settings
parser.add_argument("--pretrain_dir", type=str, default="/import/c4dm-05/bja01/models/splitfinetuning/pretrain/bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/")
parser.add_argument("--data_dir", type=str, default="/import/c4dm-05/bja01/models/splitfinetuning/data/STS-B")

parser.add_argument("--save_path", type=str, default="stsb.pt")
parser.add_argument("--log_path", type=str, default="stsb.log")

# hyperparameter settings
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=96)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--warm_up_pct", type=float, default=0.1)
parser.add_argument("--cuda", type=int, default=0, help="the index of gpus")
parser.add_argument("--split", type=int, default=6, help="indicate which layer of the bert to be split, from 1 to 12")
parser.add_argument("--rank", type=int, default=8, help="the rank of svd decomposition of the split layer")

args = parser.parse_args()

args.vocab_path = os.path.join(args.pretrain_dir, "vocab.txt")
args.weight_path = os.path.join(args.pretrain_dir, "pytorch_model.bin")
args.config_path = os.path.join(args.pretrain_dir, "config.json")

args.train_path = os.path.join(args.data_dir, "train.tsv")
args.dev_path = os.path.join(args.data_dir, "dev.tsv")
args.test_path = os.path.join(args.data_dir, "test.tsv")
args.label_path = os.path.join(args.data_dir, "labels.tsv")

# log settings
with open(args.log_path, 'a', encoding="utf-8") as f:
    f.write('\n')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(args.log_path, 'a', encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)
