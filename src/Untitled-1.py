import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
subj = utils.ExpInfo.getSubjIDs()[0]
task = utils.ExpInfo.taskName[0]
df_beh = utils.LoadData.behaviorData(subj, task)
trialno = 11
df = utils.LoadData.mouseMovement(subj, task, trialno+1)
x = df['x-shift'].cumsum()
y = df['y-shift'].cumsum()

fig, ax = plt.subplots()
utils.Plot.traj_withColour(x, y, fig, ax)