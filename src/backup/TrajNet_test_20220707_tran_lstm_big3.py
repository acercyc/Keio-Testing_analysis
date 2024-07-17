#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import utils 
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from einops import rearrange, reduce, repeat

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="test_20220707", id='tran_lstm_big3')


# %%
class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, nTime=72, nBatch=128, seed=0, add_polar=False):
        x = utils.SynthData.spiral(nTime, nBatch, seed)
        if add_polar:
            x_, y_ = utils.DataProcessing.cart2pol(x[:, :, 0], x[:, :, 1])
            x_ = repeat(x_, 'b t -> b t f', f=1)
            y_ = repeat(y_, 'b t -> b t f', f=1)
            x = np.concatenate([x, x_, y_], axis=2)
        self.data = x
              
        
    def __getitem__(self, idx):
        return self.data[idx, :, :]
        
    def __len__(self):
        return self.data.shape[0]
    
    
dataset_train = SpiralDataset(nTime=72, nBatch=128, seed=0)
dataset_val = SpiralDataset(nTime=72, nBatch=128, seed=1)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=False)


#%%
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
            self.seq.add_module(f'lstm{i}', LstmLayer(nFeature=nFeature))
            
    def forward(self, x):
        x = self.seq(x)
        return x


class Transformer(nn.Module):
    def __init__(self, nFeature=64, nhead=16, dim_feedforward=128, num_layers=8, dropout=0.1):
        super(Transformer, self).__init__()
        encoder_ = nn.TransformerEncoderLayer(d_model=nFeature,
                                            nhead=nhead,
                                            dim_feedforward=dim_feedforward,
                                            batch_first=True, 
                                            activation='gelu',
                                            dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_, num_layers=num_layers)
    def forward(self, x):
        x = self.transformer(x)
        return x 
    
       
class TrajNet(nn.Module):
    def __init__(self, nHidden=8, nFeature=64, dropout=0.1):
        super(TrajNet, self).__init__()
        self.nhidden = nHidden
        self.nFeature = nFeature

        # encoding
        self.enc_conv = nn.Conv1d(3, 64, 1)
        self.encoder = Transformer(nFeature=64, dropout=dropout)
        
        # hidden
        self.hidden = nn.Linear(64, nHidden)
        # self.silu = nn.SiLU()
        # self.mu = nn.Linear(nFeature, nHidden)
        # self.log_var = nn.Linear(nFeature, nHidden)
        # self.alpha = torch.tensor(0.01)
                
        
        # decode
        self.dec_conv1 = nn.Conv1d(nHidden, 16, 1)
        self.decoder = Lstm(nFeature=16, dropout=dropout)
        self.dec_conv2 = nn.Conv1d(16, 2, 1)


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
    def reparameterize(mu, log_var):
        std = torch.exp(0.5*log_var)  # standard deviation
        sample = torch.normal(mu, std).type_as(mu) # b f
        return sample
    
    @staticmethod
    def kl_loss_fun(mu, log_var):
        return (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        
        
    def forward(self, x):
        # x: b t f 
        nBatch = x.shape[0]
        nTime = x.shape[1]
        
        # --------------------------------- encoding --------------------------------- #
        x = self.positionEncoding(x)
        x = rearrange(x, 'b t f -> b f t')
        x = self.enc_conv(x)
        x = rearrange(x, 'b f t -> b t f')
        x = self.encoder(x)

        # ----------------------------- hidden bottleneck ---------------------------- #
        x = x[:, 0, :]
        x = self.hidden(x)
        # hidden = torch.tanh(hidden)
        # hidden = self.silu(hidden)
        # mu = self.mu(hidden)
        # mu = torch.tanh(mu)
        # log_var = self.log_var(hidden)
        # log_var = torch.tanh(log_var)
        # y = self.reparameterize(mu, log_var)
        # self.kl_loss = self.kl_loss_fun(mu, log_var)
        
        # --------------------------------- decoding --------------------------------- #
        x = repeat(x, 'b f -> b t f', t=nTime)
        x = rearrange(x, 'b t f -> b f t')
        x = self.dec_conv1(x)
        x = rearrange(x, 'b f t -> b t f')
              
        x = self.decoder(x)
        
        x = rearrange(x, 'b t f -> b f t')
        x = self.dec_conv2(x)
        x = rearrange(x, 'b f t -> b t f')
        return x
        
        
class PL_model(pl.LightningModule):
    def __init__(self):
        super(PL_model, self).__init__()
        self.model = TrajNet()
        self.c = 0
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.train()
        y = self.forward(batch)
        # loss = torch.nn.functional.mse_loss(y, batch[:, :, 0:2]) # + self.model.kl_loss * self.model.alpha
        loss = torch.nn.functional.mse_loss(y, batch) 
        self.log('train_loss', loss)     
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        
    def training_epoch_end(self, training_step_outputs):        
        if self.c % 50 == 0:
            self.eval()
            
            # training set
            x = dataset_train[0:1]
            y = self.forward(torch.from_numpy(x).to(self.device).double())
            x = x.squeeze()
            y = y.squeeze()
            y = y.detach().cpu().numpy()
            ax[0].clear()
            ax[0].plot(x[:, 0], x[:, 1], '-')
            ax[0].plot(y[:, 0], y[:, 1], '-')
            ax[0].plot(0, 0, 'or')
            ax[0].axis('equal')
            
            # evaluation set
            x = dataset_val[0:1]
            y = self.forward(torch.from_numpy(x).to(self.device).double())
            x = x.squeeze()
            y = y.squeeze()
            y = y.detach().cpu().numpy()
            ax[1].clear()
            ax[1].plot(x[:, 0], x[:, 1], '-')
            ax[1].plot(y[:, 0], y[:, 1], '-')
            ax[1].plot(0, 0, 'or')
            ax[1].axis('equal')            
                        
            img = utils.Plot.fig2img(fig)
            wandb_logger.log_image('traj', [img])
            self.c = 0
        self.c += 1


# m = TrajNet().double()
# x = torch.from_numpy(dataset_train[0:1]).double()
# m(x).shape
#%%

fig, ax = plt.subplots(1, 2)
model = PL_model().double()
trainer = pl.Trainer(max_epochs=10000, 
                     logger=wandb_logger, 
                     log_every_n_steps=10,
                     accelerator='gpu', 
                     strategy='dp')
trainer.fit(model, dataloader_train)