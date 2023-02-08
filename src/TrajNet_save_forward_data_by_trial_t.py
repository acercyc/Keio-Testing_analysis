# %%
import numpy as np
import pandas as pd
import torch
import utils


# %%
savepath = utils.path_data / 'TrajNet_xhy' 
savepath.mkdir(exist_ok=True, parents=True)

# %%
# wSize = 60
subjs = utils.ExpInfo.getSubjIDs()
task = utils.ExpInfo.taskName[0]


# %%
for wSize in [60, 50, 40, 30, 20, 10]:
    for subj in subjs:
        df = utils.LoadData.mouseMovement(subj, task)
        d = utils.DataProcessing.rollingWindow_from_df(df, wSize, 1, returnWithTrial=True)
        model = utils.Model.load(subj, task, path='TrajNet_train_onUse').eval()
        model = model.cuda()
        H = []
        Y = []
        for x in d:
            x_ = torch.from_numpy(x).double().cuda()
            y = model(x_)
            H.append(model.model.x_hidden.detach().cpu().numpy())
            Y.append(y.detach().cpu().numpy())
            
        H = np.array(H, dtype=object)
        Y = np.array(Y, dtype=object)
        savepath_ = savepath / f'{subj}_{task}_xhy_{wSize}.npz'
        np.savez(str(savepath_), x=d, y=Y, h=H)
        print(f'{subj}_{task}_xhy_{wSize}.npz saved')


