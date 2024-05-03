# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/10/31
-----------------------------------
"""
import os

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from transformers import BertModel, BertTokenizerFast

from config import args


class BertFrontDecomposition(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention

        weight = split_layer.intermediate.dense.weight
        u, s, v = torch.linalg.svd(weight)

        self.dense_s = nn.Linear(in_features=1, out_features=1, bias=False)
        self.dense_v = nn.Linear(in_features=1, out_features=1, bias=False)

        self.dense_s.weight = nn.Parameter(torch.diag(s[:rank]))
        self.dense_v.weight = nn.Parameter(v[:rank].clone())

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.attention(hidden_states, extended_attention_mask)[0]
        hidden_states = self.dense_v(hidden_states)
        hidden_states = self.dense_s(hidden_states)

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class BertBackDecomposition(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = device

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        weight = split_layer.intermediate.dense.weight
        bias = split_layer.intermediate.dense.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=1, out_features=1, bias=True)

        self.dense_u.weight = nn.Parameter(u[:, :rank].clone())
        self.dense_u.bias = bias

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.dense_u(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class DistBertDecomposition(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, label_nums, devices):
        super().__init__()
        self.front_ref = rpc.remote("cloud", BertFrontDecomposition,
                                    args=(cloud_pretrain_dir, split_num, rank, devices["cloud"]))
        self.back_ref = rpc.remote("edge", BertBackDecomposition,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["edge"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)

        return prediction

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())

        return remote_params


class BertFrontOri(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num)])

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class BertBackOri(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = device

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class DistBertOri(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, label_nums, devices):
        super().__init__()
        self.front_ref = rpc.remote("cloud", BertFrontOri, args=(cloud_pretrain_dir, split_num, rank, devices["cloud"]))
        self.back_ref = rpc.remote("edge", BertBackOri,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["edge"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)

        return prediction

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())

        return remote_params


class BertFrontFFN(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, device):
        super().__init__()
        self.device = device
        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.embeddings = bert.embeddings
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num - 1)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.attention = split_layer.attention
        self.dense = split_layer.intermediate.dense

        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.attention(hidden_states, extended_attention_mask)[0]
        hidden_states = self.dense(hidden_states)

        return hidden_states.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class BertBackFFN(nn.Module):
    def __init__(self, pretrain_dir, split_num, rank, label_nums, device):
        super().__init__()
        self.device = device

        bert = BertModel.from_pretrained(pretrain_dir)
        bert.to(device)

        self.pooler = bert.pooler
        self.encoder = nn.ModuleList([bert.encoder.layer[i] for i in range(split_num, 12)])

        split_layer = bert.encoder.layer[split_num - 1]
        self.intermediate_act_fn = split_layer.intermediate.intermediate_act_fn
        self.output = split_layer.output

        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)
        self.to(device)

    def forward(self, hidden_states=None, attention_mask=None):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        for layer in self.encoder:
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]

        hidden_states = self.pooler(hidden_states)
        cls_embedding = self.dropout(hidden_states)
        prediction = self.fc_prediction(cls_embedding)

        return prediction.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class DistBertFFN(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, label_nums, devices):
        super().__init__()
        self.front_ref = rpc.remote("cloud", BertFrontFFN,
                                    args=(cloud_pretrain_dir, split_num, rank, devices["cloud"]))
        self.back_ref = rpc.remote("edge", BertBackFFN,
                                   args=(edge_pretrain_dir, split_num, rank, label_nums, devices["edge"]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hs = self.front_ref.rpc_sync().forward(input_ids, token_type_ids, attention_mask)
        prediction = self.back_ref.rpc_sync().forward(hs, attention_mask)

        return prediction

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.front_ref.rpc_sync().parameter_rrefs())
        remote_params.extend(self.back_ref.rpc_sync().parameter_rrefs())

        return remote_params
