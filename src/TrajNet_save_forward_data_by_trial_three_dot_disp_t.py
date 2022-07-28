# %%
import utils
import torch
import numpy as np

# %%
import importlib
importlib.reload(utils)


# %%
savepath = utils.path_data / 'TrajNet_xhy' 
savepath.mkdir(exist_ok=True, parents=True)

# %%
# wSize = 60
subjs = utils.ExpInfo.getSubjIDs()
task = utils.ExpInfo.taskName[1]


# %%
for wSize in [50, 40, 30, 20, 10]:
    for subj in subjs:
        model = utils.Model.load(subj, 'one_dot', path='TrajNet_train_onUse').eval().cuda()
        motor, d = utils.LoadData.mouseMovement_array(subj, task, velocity=True)

        X = []
        H = []
        Y = []
        for x in d:
            x = utils.DataProcessing.rollingWindow(x, wSize, 1)
            X_ = []
            H_ = []
            Y_ = []
            
            for i in range(3):
                x_ = x[:, :, i*2:i*2+2]
                x_ = torch.from_numpy(x_).double().cuda()
                y = model(x_)
                X_.append(x_.detach().cpu().numpy())
                H_.append(model.model.x_hidden.detach().cpu().numpy())
                Y_.append(y.detach().cpu().numpy())
            X.append(X_)
            H.append(H_)
            Y.append(Y_)
            
            
                
        X = np.array(X, dtype=object)
        H = np.array(H, dtype=object)
        Y = np.array(Y, dtype=object)

        savepath_ = savepath / f'{subj}_{task}_xhy_disp_{wSize}.npz'
        np.savez(str(savepath_), x=X, y=Y, h=H)
        print(f'{subj}_{task}_xhy_disp_{wSize}.npz saved')


