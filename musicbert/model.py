import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.nn.functional as F
from transformers import BertModel
from config import args
import torch.linalg
from torch.utils.data import DataLoader, Dataset
from fairseq.models.roberta import TransformerSentenceEncoder, RobertaEncoder, RobertaModel


logger = logging.getLogger(__name__)
disable_cp = 'disable_cp' in os.environ
print('disable_cp =', disable_cp)
mask_strategy = os.environ['mask_strategy'].split(
    '+') if 'mask_strategy' in os.environ else ['bar']
print('mask_strategy =', mask_strategy)
assert all(item in ['element', 'compound', 'bar'] for item in mask_strategy)
convert_encoding = os.environ['convert_encoding'] if 'convert_encoding' in os.environ else 'OCTMIDI'
print('convert_encoding =', convert_encoding)
crop_length = int(os.environ['crop_length']
                  ) if 'crop_length' in os.environ else None
print('crop_length =', crop_length)  # of compound tokens
max_bars = 256
max_instruments = 256


class OctupleEncoder(TransformerSentenceEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tpu = False
        embedding_dim = kwargs['embedding_dim']
        if not disable_cp:
            self.downsampling = nn.Sequential(
                nn.Linear(embedding_dim * 8, embedding_dim))
            self.upsampling = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 8))

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ratio = 1 if disable_cp else 8
        if not disable_cp:
            assert tokens.shape[1] % ratio == 0, 'token sequences length should be multiple of ' + str(
                ratio) + ' for compound mode'
            assert last_state_only, 'hidden states not available for compound mode'
            assert positions is None, 'custom positions is not supported for compound mode'
            assert token_embeddings is None, 'custom token embeddings is not supported for compound mode'
            assert segment_labels is None, 'segment embedding not supported for compound mode'
        padding_mask = tokens[:, ::ratio].eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)
        if not disable_cp:
            x = self.downsampling(x.view(x.shape[0], x.shape[1] // ratio, -1))
        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.embed_positions is not None:
            x = x + \
                self.embed_positions(tokens[:, ::ratio], positions=positions)
        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)
        if not disable_cp:
            x = x.transpose(0, 1)
            x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
            x = x.transpose(0, 1)
        sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
        

class MusicBERTEncoder(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.sentence_encoder = OctupleEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )


class MusicBERTFront(nn.Module):
    def __init__(self, args, dictionary, split_num, rank, device):
        super(MusicBERTFront, self).__init__()
        self.device = device
        self.encoder = MusicBERTEncoder(args, dictionary)
        self.encoder.layers = nn.ModuleList([self.encoder.layers[i] for i in range(split_num)])

        # Split the FFN layer using SVD (FFN-1)
        ffn_layer = self.encoder.layers[split_num - 1].ffn
        weight = ffn_layer.intermediate.dense.weight
        u, s, v = torch.linalg.svd(weight)
        self.ffn_1 = nn.Linear(in_features=1, out_features=1, bias=False)
        self.ffn_1.weight = nn.Parameter(u[:, :rank].clone())
        
        self.to(device)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        hidden_states = self.encoder(input_ids, attention_mask)
        hidden_states = self.ffn_1(hidden_states)
        return hidden_states.cpu()


class MusicBERTBack(nn.Module):
    def __init__(self, args, dictionary, split_num, rank, device):
        super(MusicBERTBack, self).__init__()
        self.device = device
        self.encoder = MusicBERTEncoder(args, dictionary)
        self.encoder.layers = nn.ModuleList([self.encoder.layers[i] for i in range(split_num, len(self.encoder.layers))])

        # Continue splitting FFN into FFN-2 and FFN-3
        ffn_layer = self.encoder.layers[split_num - 1].ffn
        s, v = torch.linalg.svd(ffn_layer.intermediate.dense.weight)[1:3]
        self.ffn_2 = nn.Linear(in_features=1, out_features=1, bias=False)
        self.ffn_2.weight = nn.Parameter(torch.diag(s[:rank]).clone())
        self.ffn_3 = nn.Linear(in_features=1, out_features=1, bias=False)
        self.ffn_3.weight = nn.Parameter(v[:rank].clone())

        self.to(device)

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        hidden_states = self.ffn_2(hidden_states)
        hidden_states = self.ffn_3(hidden_states)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states.cpu()


class DistMusicBERT(nn.Module):
    def __init__(self, cloud_pretrain_dir, edge_pretrain_dir, split_num, rank, devices):
        super(DistMusicBERT, self).__init__()
        self.front_ref = rpc.remote("cloud", MusicBERTFront, 
                                    args=(cloud_pretrain_dir, split_num, rank, devices['cloud']))
        self.back_ref = rpc.remote("edge", MusicBERTBack, 
                                   args=(edge_pretrain_dir, split_num, rank, devices['edge']))

    def forward(self, input_ids, attention_mask):
        hidden_states = self.front_ref.rpc_sync().forward(input_ids, attention_mask)
        prediction = self.back_ref.rpc_sync().forward(hidden_states, attention_mask)
        return prediction
