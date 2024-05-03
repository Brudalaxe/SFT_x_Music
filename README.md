# SFT : Split Fine-Tuning
This repository contains the code for our paper An Efficient Split Fine-tuning Framework for Edge and Cloud Collaborative Learning. 

## dataset and pretrained model
We use Bert-base as our experiment neural network and finetune the model on 9 datasets from GLUE and SQuAD, the download link is shown below.
```
CoLA = "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
SST = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
QQP = "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
STS = "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
MNLI = "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
QNLI = "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
RTE = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
WNLI = "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip"
MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"
SQuAD_TRAIN = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQuAD_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
bert-base-uncased = "https://huggingface.co/bert-base-uncased/resolve/main/"
```

## requirement
pytorch == 1.10.2
transformers == 4.20.1
scikit-learn == 1.1.1
scipy == 1.8.1
numpy == 1.22.3

## usage

We have implemented the evaluation script and split finetune framework for our paper. Evaluation script is used to evaluate the impact on convergence results of our decomposition policies. The codes can be found in directories without "rpc" suffix. For these codes, There are 3 steps to reproduce our results.

1. download datasets and pretrained model
2. modify the arguments `--pretrain_dir` and `--data_dir` in `config.py` .
3. modify `--split` and `--rank`
4. run `python train.py`.

The directory with "rpc" suffix contains our implementation of split finetune framework. You should run the script on 2 interconnected machines. For test, you could also run it on 1 machine with 2 processes. The instructions is shown below.

1. download datasets and pretrained model
2. modify the arguments `--edge_pretrain_dir`, `--cloud_pretrain_dir` and `--data_dir` in `config.py` .
3. modify the `MASTER_ADDR` and `MASTER_PORT` in `config.py`
4. modify `--split` and `--rank`
3. run `python edge.py` on edge machine and `python cloud.py` on cloud machine.



descriptions for some key arguments:

`--split`: Int type. Indicates which layer to be decomposed and splited. The value range is determined according to the model structure. For example, value ranges from 1 to 12 for bert-base and 1 to 24 for bert-large.
`--rank`: Int type. Indicates the rank of svd decomposition of the split layer. The value range is determined according to the model structure. For example, value ranges from 1 to 768 for bert-base and 1 to 1024 for bert-large.