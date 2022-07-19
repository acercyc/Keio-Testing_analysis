# Preprocessing
`Preprocessing.ipynb`: Converting mouse.mat to csv  
Data save to *..\data\Preprocessing*

# Examine mouse trajectory 
## individual difference
`TrajSegViz.ipynb`

## trial by trial nice trajectory plot and save
`Plot_shift.ipynb`

# Autoencoder: TrajNet
``/TrajNet_testing/``: some testing files

## Training
`TrajNet_train.py`: Training script. Take subjID as CL argument 
`TrajNet_train_run2.py` & `TrajNet_train_run2.py`: Run TrajNet_train

## Forward swap 
`TrajNet_save_forward_data_by_trial` run forward_call by trial and save x, h, y to data\TrajNet_xhy