import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pytorch_lightning as pl
import TrajNet_train
import utils
import torch


subj = 'K-Reg-S-19'
task = 'one_dot'
model_type = 'val'
df = utils.LoadData.mouseMovement(subj, task)
model = TrajNet_train.PL_model()
path_cp = utils.path_data / 'TrajNet_train' / f'{subj}_one_dot_{model_type}.ckpt'
model = model.load_from_checkpoint(path_cp).double()

for trialno in np.arange(1, 3):
    fig, ax = utils.Plot.traj_and_Reconstruc_from_trial(df, trialno, model)