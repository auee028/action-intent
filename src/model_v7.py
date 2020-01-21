import os
import numpy as np
import math
import datetime
import time
import torch
import torch.nn as nn

'''
from i3d.i3d_v7 import I3D as FeatureExtractor
'''
import utils_v7 as utils
from config_v7 import FLAGS

'''
_CHECKPOINT_PATHS = {
    'rgb': './ckpt/pytorch_checkpoints/rgb_scratch.pkl',
    'flow': './ckpt/pytorch_checkpoints/flow_scratch.pkl',
    'rgb_imagenet': './ckpt/pytorch_checkpoints/rgb_imagenet.pkl',
    'flow_imagenet': './ckpt/pytorch_checkpoints/flow_imagenet.pkl',
}

class I3D(nn.Module):

    def __init__(self, i3d_type='rgb', imagenet_pretrained='true'):
        super(I3D, self).__init__()
        self.i3d = None
        if i3d_type in ['rgb', 'joint']:
            rgb_i3d = FeatureExtractor(input_channel=3)
            if imagenet_pretrained:
                state_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
                rgb_i3d.load_state_dict(state_dict)
                print('RGB checkpoint restored')
            else:
                state_dict = torch.load(_CHECKPOINT_PATHS['rgb'])
                rgb_i3d.load_state_dict(state_dict)
                print('RGB checkpoint restored')
            self.i3d = rgb_i3d
        if i3d_type in ['flow', 'joint']:
            flow_i3d = FeatureExtractor(input_channel=2)
            if imagenet_pretrained:
                state_dict = torch.load(_CHECKPOINT_PATHS['flow_imagenet'])
                flow_i3d.load_state_dict(state_dict)
                print('FLOW checkpoint restored')
            else:
                state_dict = torch.load(_CHECKPOINT_PATHS['flow'])
                flow_i3d.load_state_dict(state_dict)
                print('FLOW checkpoint restored')
            self.i3d = flow_i3d

    def forward(self, x):
        if FLAGS.mode == "train":
            print("I3D network ...\n")
        i3d_feats = self.i3d(x)
        i3d_feats = i3d_feats.squeeze(3).squeeze(3)

        return i3d_feats.permute(1, 2, 0)

    def print_paramcount(self):
        print(f'I3D parameters : {utils.count_parameters(self.i3d):,}')
'''
'''
class FeatsSampling(nn.Module):

    def __init__(self, load_idx):
        super(FeatsSampling, self).__init__()
        self.num_sampling = FLAGS.embedding_size
        self.sampling_idx = None
        if load_idx:
            self.sampling_idx = torch.load(load_idx)
        else:
            # make a new idx tensor
            values, indices = torch.multinomial(torch.ones((FLAGS.batch_size, FLAGS.i3d_dim)), FLAGS.embedding_size,
                                                replacement=False).sort(-1)
            self.sampling_idx = values

    def forward(self, x, device):
        # print(x.shape)      # torch.Size([1024, 7, 1])
        augmented_samples = torch.Tensor([]).to(device)
        for idx in self.sampling_idx:
            # print(x[idx].size())     # torch.Size([512, 7, 1])
            augmented_samples = torch.cat((augmented_samples, x[idx]))  # -> torch.Size([512, 7, 3]) : appended in loop

        return augmented_samples.transpose_(0, 2)
'''

class PositionalEncoding(nn.Module):
    # https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(FLAGS.dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)

class Transformer(nn.Module):

    def __init__(self, vocab_len, feats_dim):
        super(Transformer, self).__init__()

        self.fc_layer = nn.Linear(feats_dim, FLAGS.embedding_size)
        self.dropout_layer = nn.Dropout(FLAGS.dropout)
        self.embedding_layer = nn.Embedding(vocab_len, FLAGS.embedding_size, padding_idx=FLAGS.PAD)
        self.transformer = nn.Transformer(d_model=FLAGS.embedding_size,
                                          nhead=FLAGS.nhead,
                                          num_encoder_layers=FLAGS.num_encoder_layers,
                                          num_decoder_layers=FLAGS.num_decoder_layers,
                                          dim_feedforward=FLAGS.dim_feedforward,
                                          dropout=FLAGS.dropout,
                                          custom_encoder=None,
                                          custom_decoder=None)
        self.src_positional_encoding = PositionalEncoding(d_model=FLAGS.embedding_size, max_len=FLAGS.src_max_seq_len)
        self.trg_positional_encoding = PositionalEncoding(d_model=FLAGS.embedding_size, max_len=FLAGS.trg_max_seq_len)
        self.classifier = nn.Linear(FLAGS.embedding_size, vocab_len)

    def forward(self, src_, trg_, device):
        print("Transformer ...\n")
        src = self.fc_layer(src_)     # src: torch.Size([2, 8, 4096])

        trg = self.embedding_layer(trg_)

        src = self.src_positional_encoding(src.transpose_(0, 1))        # src : torch.Size([8, 24, 512]) -> (S, N, E)
        trg = self.trg_positional_encoding(trg.transpose_(0, 1))        # trg : torch.Size([16, 24, 512]) -> (T, N, E)

        dec_output = self.transformer(src, trg)      # src: (S, N, E) / tgt: (T, N, E) / output: (T, N, E)
        dec_output.transpose_(0, 1)  # output = torch.transpose(output, 0, 1)

        output = self.classifier(dec_output)
        # output = self.softmax(output)           # torch.Size([2, 16, 10403])

        return output

    def print_paramcount(self):
        print(f'Transformer parameters : {utils.count_parameters(self.transformer):,}')


'''
class Transformer(nn.Module):

    def __init__(self, d_model=FLAGS.embedding_size,
                          nhead=FLAGS.nhead,
                          num_encoder_layers=FLAGS.num_encoder_layers,
                          num_decoder_layers=FLAGS.num_decoder_layers,
                          dim_feedforward=FLAGS.dim_feedforward,
                          dropout=FLAGS.dropout,
                          custom_encoder=None,
                          custom_decoder=None):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=FLAGS.embedding_size,
                                          nhead=FLAGS.nhead,
                                          num_encoder_layers=FLAGS.num_encoder_layers,
                                          num_decoder_layers=FLAGS.num_decoder_layers,
                                          dim_feedforward=FLAGS.dim_feedforward,
                                          dropout=FLAGS.dropout,
                                          custom_encoder=None,
                                          custom_decoder=None)
        self.src_positional_encoding = PositionalEncoding(d_model=d_model, max_len=FLAGS.src_max_seq_len)
        self.trg_positional_encoding = PositionalEncoding(d_model=d_model, max_len=FLAGS.trg_max_seq_len)

    def forward(self, src, trg):
        if FLAGS.mode == "train":
            print("Transformer ...\n")

        src = self.src_positional_encoding(src.transpose_(0, 1))        # src : torch.Size([8, 24, 512]) -> (S, N, E)
        trg = self.trg_positional_encoding(trg.transpose_(0, 1))        # trg : torch.Size([16, 24, 512]) -> (T, N, E)

        dec_output = self.transformer(src, trg)         # dec_output : torch.Size([16, 24, 512])

        return dec_output

    def print_paramcount(self):
        print(f'Transformer parameters : {utils.count_parameters(self.transformer):,}')


class I3D_Transformer(nn.Module):

    def __init__(self, vocab_len):
        super(I3D_Transformer, self).__init__()
        self.i3d = I3D(imagenet_pretrained='true')

        self.feats_sampling = FeatsSampling(load_idx=FLAGS.idx_path)

        self.fc_layer = nn.Linear(FLAGS.i3d_dim, FLAGS.embedding_size)
        self.dropout_layer = nn.Dropout(FLAGS.dropout)

        self.src_positional_encoding = PositionalEncoding(d_model=FLAGS.embedding_size, max_len=FLAGS.src_max_seq_len)
        self.trg_positional_encoding = PositionalEncoding(d_model=FLAGS.embedding_size, max_len=FLAGS.trg_max_seq_len)

        self.embedding_layer = nn.Embedding(vocab_len, FLAGS.embedding_size, padding_idx=FLAGS.PAD)
        self.transformer = Transformer(d_model=FLAGS.embedding_size,
                                          nhead=FLAGS.nhead,
                                          num_encoder_layers=FLAGS.num_encoder_layers,
                                          num_decoder_layers=FLAGS.num_decoder_layers,
                                          dim_feedforward=FLAGS.dim_feedforward,
                                          dropout=FLAGS.dropout,
                                          custom_encoder=None,
                                          custom_decoder=None)

        self.classifier = nn.Linear(FLAGS.embedding_size, vocab_len)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_, trg_, device):
        i3d_feats = self.i3d(src_)      # src_: torch.Size([1, 3, 64, 224, 224]) -> i3d_feats: torch.Size([1024, 7, 1])

        # src = self.feats_sampling(i3d_feats, device)    # torch.Size([3, 7, 512])
        src = self.fc_layer(i3d_feats.transpose_(0, 2))     # src: torch.Size([2, 8, 512])

        trg = self.embedding_layer(trg_)

        trg[trg == FLAGS.PAD] = 0

        dec_output = self.transformer(src, trg)      # src: (S, N, E) / tgt: (T, N, E) / output: (T, N, E)
        dec_output.transpose_(0, 1)  # output = torch.transpose(output, 0, 1)

        output = self.classifier(dec_output)
        # output = self.softmax(output)           # torch.Size([2, 16, 10403])

        return output

    def get_idx(self):
        return self.feats_sampling.sampling_idx
'''
