import utils
import os

subjs = utils.ExpInfo.getSubjIDs()[0:25]
for subj in subjs:
    gDict = {}
    gDict['subj'] = subj
    os.system(f"python '/home/acercyc/projects/Keio Testing_analysis/src/TrajNet_train_60s.py' {subj}")