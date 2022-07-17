import utils
import os

subjs = ['K-Reg-S-20',
         'K-Reg-H-23',
         'K-Reg-S-15']

# subjs = utils.ExpInfo.getSubjIDs()[46:47]
for subj in subjs:
    gDict = {}
    gDict['subj'] = subj
    os.system(
        f"python '/home/acercyc/projects/Keio Testing_analysis/src/TrajNet_train_60s_retrain.py' {subj}")
