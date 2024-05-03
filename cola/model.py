# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Encore
@Date : 2022/8/18
-----------------------------------
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from transformers.pytorch_utils import apply_chunking_to_forward

from config import args

torch.set_printoptions(precision=8)


class Classification(nn.Module):
    def __init__(self, pretrain_dir, label_nums):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_dir)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_prediction = nn.Linear(in_features=768, out_features=label_nums)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        cls_embedding = self.bert(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask).pooler_output
        cls_embedding = self.dropout(cls_embedding)
        prediction = self.fc_prediction(cls_embedding)

        return prediction


class SvdLayer(nn.Module):
    def __init__(self, layer, rank):
        super().__init__()

        self.LayerNorm = layer.LayerNorm
        self.dropout = layer.dropout
        weight = layer.dense.weight
        bias = layer.dense.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=1, out_features=1, bias=True)
        self.dense_s = nn.Linear(in_features=1, out_features=1, bias=False)
        self.dense_v = nn.Linear(in_features=1, out_features=1, bias=False)

        self.dense_u.weight = nn.Parameter(u[:, :rank].clone())
        self.dense_u.bias = bias
        self.dense_s.weight = nn.Parameter(torch.diag(s[:rank]))
        self.dense_v.weight = nn.Parameter(v[:rank].clone())

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_v(hidden_states)
        hidden_states = self.dense_s(hidden_states)
        hidden_states = self.dense_u(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class SvdLinear(nn.Module):
    def __init__(self, layer, rank):
        super().__init__()
        weight = layer.weight
        bias = layer.bias
        u, s, v = torch.linalg.svd(weight)

        self.dense_u = nn.Linear(in_features=1, out_features=1, bias=True)
        self.dense_s = nn.Linear(in_features=1, out_features=1, bias=False)
        self.dense_v = nn.Linear(in_features=1, out_features=1, bias=False)

        self.dense_u.weight = nn.Parameter(u[:, :rank].clone())
        self.dense_u.bias = bias
        self.dense_s.weight = nn.Parameter(torch.diag(s[:rank]))
        self.dense_v.weight = nn.Parameter(v[:rank].clone())

    def forward(self, inputs):
        out = self.dense_v(inputs)
        out = self.dense_s(out)
        out = self.dense_u(out)

        return out


class SvdBlock(nn.Module):
    def __init__(self, layer, rank):
        super().__init__()

        self.chunk_size_feed_forward = layer.chunk_size_feed_forward
        self.seq_len_dim = layer.seq_len_dim
        self.attention = layer.attention
        self.is_decoder = layer.is_decoder
        self.add_cross_attention = layer.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = layer.crossattention
        self.intermediate = layer.intermediate
        self.output = layer.output

        self.intermediate.dense = SvdLinear(self.intermediate.dense, rank=rank)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        hidden_states = self.intermediate.dense.dense_v(attention_output)
        hidden_states = self.intermediate.dense.dense_s(hidden_states)
        hidden_states = self.intermediate.dense.dense_u(hidden_states)
        hidden_states = self.intermediate.intermediate_act_fn(hidden_states)
        hidden_states = self.output.dense(hidden_states)
        hidden_states = self.output.dropout(hidden_states)
        hidden_states = self.output.LayerNorm(hidden_states)

        return hidden_states

# if __name__ == '__main__':
    # t = torch.randn(5, 10)
    # layer = nn.Linear(10, 3)
    # print(layer(t))
    #
    # svd = SvdLinear(layer, 3)
    # print(svd(t))

    # text = "今天在下雨"
    # tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_dir)
    # encode_text = tokenizer(text, return_tensors="pt")
    #
    # model = Classification(args.pretrain_dir, 2)
    # model.eval()
    # print(model.bert.encoder.layer[5].output)
    # print(model(**encode_text))
    #
    # layer = model.bert.encoder.layer[5].output.dense
    # model.bert.encoder.layer[5].output.dense = SvdLinear(layer, rank=500)
    # print(model.bert.encoder.layer[5].output)
    # print(model(**encode_text))
