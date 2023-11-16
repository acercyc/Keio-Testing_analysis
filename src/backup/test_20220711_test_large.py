# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
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
config = {'nHidden': 8, 'nFeature': 128, 'nhead': 32, 'dim_feedforward': 256, 'num_layers': 8, 'dim_posEnc': 8}
wandb_logger = WandbLogger(project="test_20220711", config=config)
# wandb_logger.experiment.config.update(config)

# %%
dataset_train, dataset_val = utils.LoadData.mouseMovementRollingData()
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=4)

# %%
class PL_model(pl.LightningModule):
    def __init__(self):
        super(PL_model, self).__init__()
        self.model = models.TrajNet_tran2tran(**config)
        self.fig = plt.figure()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        batch = utils.DataProcessing.seqTrim(batch, 24)
        y = self.forward(batch)
        loss = torch.nn.functional.mse_loss(y, batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print(batch.shape)
        y = self.forward(batch)
        loss = torch.nn.functional.mse_loss(y, batch)
        self.log('val_loss', loss)
        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        self.fig.clear()
        if type(validation_step_outputs) is list:
            validation_step_outputs = validation_step_outputs[0]
        
        x_train = dataset_train[0:1]
        y_train = self.forward(torch.from_numpy(x_train).type_as(validation_step_outputs))
        y_train = y_train.detach().cpu().numpy()
        x_train = x_train.squeeze().cumsum(axis=0)
        y_train = y_train.squeeze().cumsum(axis=0)
        ax = self.fig.add_subplot(1, 2, 1)
        utils.Plot.traj_and_Reconstruc(x_train, y_train, ax, legend=False)        
        
        x_val = dataset_val[0:1]
        y_val = self.forward(torch.from_numpy(x_val).type_as(validation_step_outputs))
        y_val = y_val.detach().cpu().numpy()
        x_val = x_val.squeeze().cumsum(axis=0)
        y_val = y_val.squeeze().cumsum(axis=0)        
        ax = self.fig.add_subplot(1, 2, 2)
        utils.Plot.traj_and_Reconstruc(x_val, y_val, ax, legend=False)
        
        img = utils.Plot.fig2img(self.fig)
        wandb_logger.log_image('traj', [img])
        
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    



# %%
callbacks = []
# callbacks.append(EarlyStopping('val_loss', patience=30, mode='min'))
callbacks.append(ModelCheckpoint(monitor='val_loss', mode='min', verbose=True))

# %%
model = PL_model().double()
trainer = pl.Trainer(max_epochs=100000, 
                     logger=wandb_logger,
                     accelerator='gpu', 
                     strategy="ddp",
                     callbacks=callbacks)
trainer.fit(model, dataloader_train, dataloader_val)


