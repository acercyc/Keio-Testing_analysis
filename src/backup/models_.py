import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from torchsummary import summary
from einops import rearrange, reduce, repeat

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# ---------------------------------------------------------------------------- #
#                                    LSTM_AE                                   #
# ---------------------------------------------------------------------------- #
class LSTM_AE(pl.LightningModule):
    def __init__(self, nhidden=16, nLayer=8, nFeature=32, dropout=0.02):
        super(LSTM_AE, self).__init__()

        # Encoder
        self.conv_enc = nn.Conv1d(2, nFeature, kernel_size=1)
        self.dropout_enc = nn.Dropout(dropout)
        self.encoder = Encoder_LSTM_AE(
            nLayer=nLayer, nFeature=nFeature, dropout=dropout)

        # Hidden
        self.preZ = nn.Linear(nFeature, nhidden)
        self.postZ = nn.Linear(nhidden, nFeature)

        # Decoder
        self.conv_dec1 = nn.Conv1d(nhidden, nFeature, kernel_size=1)
        self.decoder = Decoder_LSTM_AE(
            nLayer=nLayer, nFeature=nFeature, dropout=dropout)
        self.conv_dec2 = nn.Conv1d(nFeature, 2, kernel_size=1)

    @staticmethod
    def reparameterize(mu, log_var, nTime):
        mu = repeat(mu, 'b f -> b t f', t=nTime)
        std = torch.exp(0.5*log_var)  # standard deviation
        std = repeat(std, 'b f -> b t f', t=nTime)
        sample = torch.normal(mu, std)  # b f
        # sample = repeat(sample, 'b f -> b t f', t=nTime)
        return sample

    def forward(self, x):
        # x: b t f
        nTime = x.shape[1]
        # --------------------------------- encoding --------------------------------- #
        x = rearrange(x, 'b t f -> b f t')
        x = self.conv_enc(x)
        # x = self.dropout_enc(x)
        x = rearrange(x, 'b f t -> b t f')
        x = self.encoder(x)
        x = x[:, -1, :]

        # ---------------------------------- hidden ---------------------------------- #
        x = self.preZ(x)
        z = torch.tanh(x)
        y = self.postZ(z)
        y = torch.tanh(z)
        y = repeat(y, 'b f -> b t f', t=nTime)

        # --------------------------------- decoding --------------------------------- #
        y = rearrange(y, 'b t f -> b f t')
        y = self.conv_dec1(y)
        y = rearrange(y, 'b f t -> b t f')
        y = self.decoder(y)
        # y = y[:, :, 0:2]
        y = rearrange(y, 'b t f -> b f t')
        y = self.conv_dec2(y)
        y = rearrange(y, 'b f t -> b t f')

        return y

    def training_step(self, batch, batch_idx):
        y = self.forward(batch)
        loss = torch.nn.functional.mse_loss(y, batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class EncoderLayer_LSTM_AE(nn.Module):
    def __init__(self, nFeature=32, dropout=0.1):
        super(EncoderLayer_LSTM_AE, self).__init__()
        self.lstm = nn.LSTM(nFeature, nFeature, 1, batch_first=True)
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hc=None):
        # b t f
        res = x
        if hc is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hc)
        x = self.selu(x)
        x += res
        x = self.dropout(x)
        return x


class Encoder_LSTM_AE(nn.Module):
    def __init__(self, nLayer=8, nFeature=32, dropout=0.1):
        super(Encoder_LSTM_AE, self).__init__()
        self.seq = nn.Sequential()
        for i in range(nLayer):
            self.seq.add_module(
                f'lstm{i}', EncoderLayer_LSTM_AE(nFeature=nFeature))

    def forward(self, x):
        x = self.seq(x)
        return x


class DecoderLayer_LSTM_AE(nn.Module):
    def __init__(self, nFeature=32, dropout=0.1):
        super(DecoderLayer_LSTM_AE, self).__init__()
        self.lstm = nn.LSTM(nFeature, nFeature, 1, batch_first=True)
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hc=None):
        # b t f
        res = x
        if hc is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hc)
        x = self.selu(x)
        x += res
        x = self.dropout(x)
        return x


class Decoder_LSTM_AE(nn.Module):
    def __init__(self, nLayer=8, nFeature=32, dropout=0.1):
        super(Decoder_LSTM_AE, self).__init__()
        self.seq = nn.Sequential()
        for i in range(nLayer):
            self.seq.add_module(
                f'lstm{i}', DecoderLayer_LSTM_AE(nFeature=nFeature))

    def forward(self, x):
        x = self.seq(x)
        return x

# model = Decoder()
# model(torch.rand(5, 3, 32)).shape


# class TrajNet(nn.Module):
#     def __init__(self, nFeature=4, nhead=2, dim_feedforward=64, num_layers=2, dropout=0.1):
#         super(TrajNet, self).__init__()

#         # encoder
#         self.conv_enc = nn.Conv1d(2, nFeature, 1)
#         encoder_ = nn.TransformerEncoderLayer(d_model=nFeature,
#                                               nhead=nhead,
#                                               dim_feedforward=dim_feedforward,
#                                               batch_first=True, activation='gelu',
#                                               dropout=dropout)
#         self.encoder = nn.TransformerEncoder(encoder_, num_layers=num_layers)

#         # hidden bottleneck
#         self.hidden = nn.Linear(nFeature, nFeature)

#         # decoder
#         self.conv_dec1 = nn.Conv1d(2, nFeature, 1)
#         self.conv_dec2 = nn.Conv1d(nFeature, 2, 1)
#         decoder_ = nn.TransformerDecoderLayer(d_model=nFeature,
#                                               nhead=nhead,
#                                               dim_feedforward=dim_feedforward,
#                                               batch_first=True,
#                                               activation='gelu',
#                                               dropout=dropout)
#         self.decoder = nn.TransformerDecoder(decoder_, num_layers=num_layers)

#         # output
#         # self.output = nn.Linear(nFeature, 2)

#     @staticmethod
#     def timeShift(x, nShift=1):
#         # x.shape = (Batch, Time, Feature)
#         x = torch.roll(x, nShift, dims=1)
#         x[:, 0, :] = 0
#         return x

#     def forward(self, src, tgt=None, add_tgt_mask=True):
#         # src.shape = (batch_size, nSrc, nFeature=2)
#         # tgt.shape = (batch_size, nTgt, nFeature=2)

#         # ---------------------------------------------------------------------------- #
#         #                                   encoding                                   #
#         # ---------------------------------------------------------------------------- #
#         x = src

#         # expend feature dimension
#         #   x_expand = (batch, time, nFeature)
#         x = x.permute(0, 2, 1)
#         x = self.conv_enc(x)
#         x = x.permute(0, 2, 1)

#         x = self.encoder(x)
#         x = nn.functional.gelu(x)

#         # extract hidden control units
#         #   x.shape = (batch, head)
#         hidden = x[:, 0, :]
#         hidden = self.hidden(hidden)
#         memory = hidden
#         memory = nn.functional.gelu(memory)

#         # construct memory embedding
#         # memory.shape = (batch, time, feature)
#         memory = torch.unsqueeze(memory, 1)


#         # ---------------------------------------------------------------------------- #
#         #                                   decoding                                   #
#         # ---------------------------------------------------------------------------- #

#         if tgt is None:
#             tgt = self.timeShift(src)

#         y = tgt
#         # expend feature dimension
#         # x_expand = (batch, time, nFeature)
#         y = y.permute(0, 2, 1)
#         y = self.conv_dec1(y)
#         y = y.permute(0, 2, 1)

#         if add_tgt_mask:
#             tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
#         else:
#             tgt_mask = None

#         y = self.decoder(y, memory, tgt_mask=tgt_mask)

#         # y.shape = (batch_size, nTime, nFeature=2)
#         y = y.permute(0, 2, 1)
#         y = self.conv_dec2(y)
#         y = y.permute(0, 2, 1)

#         return y

#     def autoRegressiveDecoding(self, src, nTime):
#         tgt = torch.zeros((src.shape[0], 1, 2)).to(device).float()
#         for i in range(nTime):
#             y = self.forward(src, tgt)[:, -1:, :]
#             tgt = torch.concat((tgt, y), dim=1)
#         return tgt[:, 1:, :]
