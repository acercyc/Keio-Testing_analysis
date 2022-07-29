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
`TrajNet_save_forward_data_by_trial_{task}` run forward_call by trial for motor data and save x, h, y to data\TrajNet_xhy
`TrajNet_save_forward_data_by_trial_{task}_disp.py` run forward_call for disp data by trial and save x, h, y to data\TrajNet_xhy

# Analysis
`test_seqLen_dim.ipynb`: Examine dimensionality on different sequence lengths 

## Three dot exp
`ana_three_dot_predicting_individual_beh_profile.ipynb`: Predicting choice by position and velocity of action plans

# Behavioral data
`ana_behavioral_results.ipynb`: Behavioral result summary