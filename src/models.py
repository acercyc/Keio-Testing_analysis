# %%
import numpy as np
import pandas as pd
import utils
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce, repeat


class Transformer(nn.Module):
    def __init__(self, nFeature=64, nhead=16, dim_feedforward=128, num_layers=8, dropout=0.1):
        super(Transformer, self).__init__()
        encoder_ = nn.TransformerEncoderLayer(d_model=nFeature,
                                              nhead=nhead,
                                              dim_feedforward=dim_feedforward,
                                              batch_first=True,
                                              activation='gelu',
                                              dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_, num_layers=num_layers)

    def forward(self, x, mask=None):
        x = self.transformer(x, mask=mask)
        return x

# ---------------------------------------------------------------------------- #
#                                   tran2tran                                  #
# ---------------------------------------------------------------------------- #
class TrajNet_tran2tran(nn.Module):
    def __init__(self, nHidden=8, dim_posEnc=8,
                 nFeature=64, nhead=16, dim_feedforward=128, num_layers=8,  dropout=0.1,
                 nFeature_dec=64, nhead_dec=16, dim_feedforward_dec=128, num_layers_dec=8, dropout_dec=0.1):
        super(TrajNet_tran2tran, self).__init__()
        self.nhidden = nHidden
        self.nFeature = nFeature
        self.nFeature_dec = nFeature_dec
        self.dim_posEnc = dim_posEnc

        # encoding
        self.enc_conv = nn.Conv1d(2+dim_posEnc*2, nFeature, 1)
        self.encoder = Transformer(nFeature=nFeature, nhead=nhead,
                                   dim_feedforward=dim_feedforward, num_layers=num_layers, dropout=dropout)

        # hidden
        self.hidden = nn.Linear(nFeature, nHidden)

        # decode
        self.dec_conv1 = nn.Conv1d(nHidden, nFeature_dec-dim_posEnc*2-2, 1)
        self.decoder = Transformer(nFeature=nFeature_dec, nhead=nhead_dec,
                                   dim_feedforward=dim_feedforward_dec, num_layers=num_layers_dec, dropout=dropout_dec)
        self.dec_conv2 = nn.Conv1d(nFeature_dec, 2, 1)

    @staticmethod
    def positionEncoding(x):
        # x: b t f
        nBatch = x.shape[0]
        nTime = x.shape[1]
        p = torch.arange(0, nTime).type_as(x)
        p = p / 300
        p = repeat(p, 't -> b t f', b=nBatch, f=1)
        x = torch.concat([x, p], dim=2)
        return x

    @staticmethod
    def positionEncoding_sincos(x, dim=4):
        nBatch = x.shape[0]
        nTime = x.shape[1]
        mat = utils.DataProcessing.positionEncoding_sincos_mat(
            nTime, dim=dim)
        mat = repeat(mat, 't f -> b t f', b=nBatch)
        mat = torch.from_numpy(mat).type_as(x)
        x = torch.concat([x, mat], dim=2)
        return x

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5*log_var)  # standard deviation
        sample = torch.normal(mu, std).type_as(mu)  # b f
        return sample

    @staticmethod
    def kl_loss_fun(mu, log_var):
        return (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)

    def forward(self, x):
        # x: b t f
        # nBatch = x.shape[0]
        nTime = x.shape[1]
        xy_init = x[:, 0, :]

        # --------------------------------- encoding --------------------------------- #
        x = self.positionEncoding_sincos(x, dim=self.dim_posEnc)
        x = rearrange(x, 'b t f -> b f t')
        x = self.enc_conv(x)
        x = rearrange(x, 'b f t -> b t f')
        x = self.encoder(x)

        # ----------------------------- hidden bottleneck ---------------------------- #
        x = x[:, 0, :]
        x = self.hidden(x)
        x = torch.tanh(x)
        self.x_hidden = x

        # --------------------------------- decoding --------------------------------- #
        x = repeat(x, 'b f -> b t f', t=nTime)
        x = rearrange(x, 'b t f -> b f t')
        x = self.dec_conv1(x)
        x = rearrange(x, 'b f t -> b t f')

        xy_init = repeat(xy_init, 'b f -> b t f', t=nTime)
        x = torch.concat([x, xy_init], dim=2)
        x = self.positionEncoding_sincos(x, dim=self.dim_posEnc)
        mask = nn.Transformer.generate_square_subsequent_mask(nTime).type_as(x)
        x = self.decoder(x, mask=mask)

        x = rearrange(x, 'b t f -> b f t')
        x = self.dec_conv2(x)
        x = rearrange(x, 'b f t -> b t f')
        return x

# ---------------------------------------------------------------------------- #
#                                     LSTM                                     #
# ---------------------------------------------------------------------------- #


class LstmLayer(nn.Module):
    def __init__(self, nFeature=32, dropout=0.1):
        super(LstmLayer, self).__init__()
        self.lstm = nn.LSTM(nFeature, nFeature, 1, batch_first=True)
        self.linear = nn.Linear(nFeature*2, nFeature)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hc=None):
        # b t f
        res = x
        if hc is None:
            x, _ = self.lstm(x)
        else:
            x, _ = self.lstm(x, hc)
        x = rearrange([x, res], 'c b t f -> b t (f c)')
        x = self.linear(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


class Lstm(nn.Module):
    def __init__(self, nLayer=4, nFeature=16, dropout=0.1):
        super(Lstm, self).__init__()
        self.seq = nn.Sequential()
        for i in range(nLayer):
            self.seq.add_module(f'lstm{i}', LstmLayer(nFeature=nFeature, dropout=dropout))

    def forward(self, x):
        x = self.seq(x)
        return x


class TrajNet_tran2lstm(nn.Module):
    def __init__(self, nHidden=8,
                 nFeature=64, nhead=16, dim_feedforward=128, num_layers=8, dim_posEnc=8, dropout=0.1,
                 nFeature_lstm=32, nLayer_lstm=4, dropout_lstm=0.1):
        super(TrajNet_tran2lstm, self).__init__()
        self.nhidden = nHidden
        self.nFeature = nFeature
        self.dim_posEnc = dim_posEnc

        # encoding
        self.enc_conv = nn.Conv1d(2+dim_posEnc*2, nFeature, 1)
        self.encoder = Transformer(nFeature=nFeature, nhead=nhead,
                                   dim_feedforward=dim_feedforward, 
                                   num_layers=num_layers, 
                                   dropout=dropout)

        # hidden
        self.hidden = nn.Linear(nFeature, nHidden)

        # decode
        self.dec_conv1 = nn.Conv1d(nHidden+2, nFeature_lstm, 1)
        self.decoder = Lstm(nLayer=nLayer_lstm, nFeature=nFeature_lstm, dropout=dropout_lstm)
        self.dec_conv2 = nn.Conv1d(nFeature_lstm, 2, 1)

    @staticmethod
    def positionEncoding(x):
        # x: b t f
        nBatch = x.shape[0]
        nTime = x.shape[1]
        p = torch.arange(0, nTime).type_as(x)
        p = p / 300
        p = repeat(p, 't -> b t f', b=nBatch, f=1)
        x = torch.concat([x, p], dim=2)
        return x

    @staticmethod
    def positionEncoding_sincos(x, dim=4):
        nBatch = x.shape[0]
        nTime = x.shape[1]
        mat = utils.DataProcessing.positionEncoding_sincos_mat(
            nTime, dim=dim)
        mat = repeat(mat, 't f -> b t f', b=nBatch)
        mat = torch.from_numpy(mat).type_as(x)
        x = torch.concat([x, mat], dim=2)
        return x

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5*log_var)  # standard deviation
        sample = torch.normal(mu, std).type_as(mu)  # b f
        return sample

    @staticmethod
    def kl_loss_fun(mu, log_var):
        return (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)

    def forward(self, x):
        # x: b t f
        nBatch = x.shape[0]
        nTime = x.shape[1]
        xy_init = x[:, 0, :]

        # --------------------------------- encoding --------------------------------- #
        x = self.positionEncoding_sincos(x, dim=self.dim_posEnc)
        x = rearrange(x, 'b t f -> b f t')
        x = self.enc_conv(x)
        x = rearrange(x, 'b f t -> b t f')
        x = self.encoder(x)

        # ----------------------------- hidden bottleneck ---------------------------- #
        x = x[:, 0, :]
        x = self.hidden(x)
        x = torch.tanh(x)
        self.x_hidden = x        

        # --------------------------------- decoding --------------------------------- #
        x = torch.concat([x, xy_init], dim=1)
        x = repeat(x, 'b f -> b t f', t=nTime)
        x = rearrange(x, 'b t f -> b f t')
        x = self.dec_conv1(x)
        x = rearrange(x, 'b f t -> b t f')

        x = self.decoder(x)

        x = rearrange(x, 'b t f -> b f t')
        x = self.dec_conv2(x)
        x = rearrange(x, 'b f t -> b t f')
        return x
