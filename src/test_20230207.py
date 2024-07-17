
import numpy as np
import pandas as pd

import utils
from sklearn.metrics.pairwise import paired_distances

import matplotlib.pyplot as plt
import seaborn as sns 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotnine as pn

from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from sklearn.preprocessing import StandardScaler, scale
from scipy.spatial import distance

from einops import rearrange, reduce, repeat
subj = utils.ExpInfo.getSubjIDs()[0]
task = utils.ExpInfo.taskName[1]
df_beh = utils.LoadData.behaviorData(subj, task)
# df_beh.head()
utils.LoadData.mouseMovement(subj, task)
utils.LoadData.mouseMovement_array(subj, task)