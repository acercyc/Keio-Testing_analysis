import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import TrajNet_train
import utils
import torch

subj = utils.ExpInfo.getSubjIDs()[25]
task = 'one_dot'
print(subj, task)

trialno = 1
df = utils.LoadData.mouseMovement(subj, task).query(f'trialno == {trialno}').head()
utils.DataProcessing.rollingWindow_from_df(df, 30, 1)
