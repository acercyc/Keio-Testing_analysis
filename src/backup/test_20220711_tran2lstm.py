# %%
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


# %%

# exp
config = {}
config['wSize'] = 40
config['minTime'] = 10
config['batch_size'] = 64
config['pos'] = False
config['multiGPU'] = True

# lr
config['lr'] = 0.001
config['lr_scheduler'] = True
config['lr_scheduler_factor'] = 0.9
config['lr_scheduler_patience'] = 10

# training
config['hidden_loss'] = 0

# model
config_model = {'nHidden': 16, 'dim_posEnc': 8,
                'nFeature': 32, 'nhead': 32, 'dim_feedforward': 32, 'num_layers': 4, 'dropout': 0.1,
                'nFeature_lstm': 64, 'nLayer_lstm': 2, 'dropout_lstm': 0.1}
config.update(config_model)

tags = ['16_hidden']
wandb_logger = WandbLogger(project="test_20220711", config=config, tags=tags, group='tran2lstm')


# %%
dataset_train, dataset_val = utils.LoadData.mouseMovementRollingData(wSize=config['wSize'], pos=config['pos'], seed=1)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config['batch_size'], shuffle=False, num_workers=4)

# %%
class PL_model(pl.LightningModule):
    def __init__(self):
        super(PL_model, self).__init__()
        self.model = models.TrajNet_tran2lstm(**config_model)
        self.fig = plt.figure()
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
        if config['lr_scheduler']:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                      patience=config['lr_scheduler_patience'], 
                                                                      factor=config['lr_scheduler_factor'], 
                                                                      verbose=True)
            lr_scheduler_config = {'scheduler': lr_scheduler, 'monitor': 'train_loss'}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            return optimizer
    
    def training_step(self, batch, batch_idx):
        batch = utils.DataProcessing.seqTrim(batch, minTime=config['minTime'])
        y = self.forward(batch)
        
        # hidden_loss
        if config['hidden_loss'] > 0:
            x_hidden = self.model.x_hidden
            corr = torch.corrcoef(x_hidden.T)
            eye = torch.eye(corr.shape[0]).type_as(x_hidden)
            hidden_loss = torch.mean((corr - eye)**2) * config['hidden_loss'] 
        else:
            hidden_loss = 0        
        
        # mse loss    
        loss_mse = torch.nn.functional.mse_loss(y, batch)
        self.log('train_loss', loss_mse)
        
        # sum loss
        loss = loss_mse + hidden_loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        print(batch.shape)
        y = self.forward(batch)
                
        # val_loss
        val_loss = torch.nn.functional.mse_loss(y, batch)
        self.log('val_loss', val_loss)
        
        return val_loss
    
    def validation_epoch_end(self, validation_step_outputs):
        self.fig.clear()
        if type(validation_step_outputs) is list:
            validation_step_outputs = validation_step_outputs[0]
        
        x_train = dataset_train[0:1]
        y_train = self.forward(torch.from_numpy(x_train).type_as(validation_step_outputs))
        y_train = y_train.detach().cpu().numpy()
        x_train = x_train.squeeze()
        y_train = y_train.squeeze()
        if config['pos'] is not True:
            x_train = x_train.cumsum(axis=0)
            y_train = y_train.cumsum(axis=0)
        ax = self.fig.add_subplot(1, 2, 1)
        utils.Plot.traj_and_Reconstruc(x_train, y_train, ax, legend=False)        
        
        x_val = dataset_val[0:1]
        y_val = self.forward(torch.from_numpy(x_val).type_as(validation_step_outputs))
        y_val = y_val.detach().cpu().numpy()
        x_val = x_val.squeeze()
        y_val = y_val.squeeze()
        if config['pos'] is not True:
            x_val = x_val.cumsum(axis=0)
            y_val = y_val.cumsum(axis=0)        
        ax = self.fig.add_subplot(1, 2, 2)
        utils.Plot.traj_and_Reconstruc(x_val, y_val, ax, legend=False)
        
        img = utils.Plot.fig2img(self.fig)
        wandb_logger.log_image('traj', [img])
    
# %%
callbacks = []
callbacks.append(EarlyStopping('val_loss', patience=50, mode='min'))
callbacks.append(ModelCheckpoint(monitor='val_loss', mode='min', verbose=True))

# %%
model = PL_model().double()

if config['multiGPU']:
    trainer = pl.Trainer(max_epochs=100000, 
                        logger=wandb_logger,
                        accelerator='gpu', 
                        strategy="ddp",
                        callbacks=callbacks)
else:
    trainer = pl.Trainer(max_epochs=100000, 
                        logger=wandb_logger,
                        accelerator='gpu',
                        auto_select_gpus=True, 
                        gpus=1,
                        callbacks=callbacks)
trainer.fit(model, dataloader_train, dataloader_val)


