# %%
import utils
import torch
import numpy as np


# %%
savepath = utils.path_data / 'TrajNet_xhy' 
savepath.mkdir(exist_ok=True, parents=True)

# %%
wSize = 60
subjs = utils.ExpInfo.getSubjIDs()
task = utils.ExpInfo.taskName[0]


# %%
for subj in subjs:
    model = utils.Model.load(subj, task, path='TrajNet_train_onUse').eval().cuda()
    motor, d = utils.LoadData.mouseMovement_array(subj, task, velocity=True)
    X = []
    H = []
    Y = []
    for x in d:
        x = utils.DataProcessing.rollingWindow(x, wSize, 1)
        x_ = torch.from_numpy(x).double().cuda()
        y = model(x_)
        X.append(x)
        H.append(model.model.x_hidden.detach().cpu().numpy())
        Y.append(y.detach().cpu().numpy())    
        
    X = np.array(X, dtype=object)
    H = np.array(H, dtype=object)
    Y = np.array(Y, dtype=object)

    savepath_ = savepath / f'{subj}_{task}_xhy_disp.npz'
    np.savez(str(savepath_), x=X, y=Y, h=H)
    print(f'{subj}_{task}_xhy_disp.npz saved')



