# %%
from re import U
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import config_context
import utils
import torch
import models
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from einops import rearrange, reduce, repeat
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('subj', type=str, help='subject id')  
args = parser.parse_args()
# %%


# %%
def dummy(x):
   return x/2 + torch.rand(1) * 0.01

x = torch.linspace(0, 0.5, 128).unsqueeze(1)
y = dummy(x)
dataset_train = torch.utils.data.TensorDataset(x, y)

x = torch.linspace(0.5, 1, 128).unsqueeze(1)
y = dummy(x)
dataset_val = torch.utils.data.TensorDataset(x, y)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4)

class PL_model(pl.LightningModule):
    def __init__(self):
        super(PL_model, self).__init__()
        self.model = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)
    


# %%
if __name__ == '__main__':
    subj = args.subj
    task = utils.ExpInfo.taskName[0]
    savepath = utils.path_data / 'Autoencoder' 
    savepath.mkdir(parents=True, exist_ok=True)
    savepath_wb = savepath / f'{subj}_{task}'
    savepath_wb.mkdir(parents=True, exist_ok=True)

    callbacks = []
    callbacks.append(EarlyStopping('val_loss', patience=5, mode='min'))
    callbacks.append(ModelCheckpoint(monitor='train_loss', mode='min', verbose=True, dirpath=str(savepath), filename=f'{subj}_{task}_train'))
    callbacks.append(ModelCheckpoint(monitor='val_loss', mode='min', verbose=True, dirpath=str(savepath), filename=f'{subj}_{task}_val'))

    model = PL_model()
    wandb_logger = WandbLogger(project="test_20220713_logger_test", name=f'{subj}_{task}', save_dir=str(savepath_wb))
    trainer = pl.Trainer(max_epochs=3, 
                        logger=wandb_logger,
                        accelerator='gpu',
                        auto_select_gpus=True, 
                        gpus=1,
                        callbacks=callbacks,
                        log_every_n_steps=10)
    trainer.fit(model, dataloader_train, dataloader_val)
    # wandb_logger.finalize('finished')
    # wandb.finish()




