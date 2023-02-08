import numpy as np
import pandas as pd
import utils

import matplotlib.pyplot as plt 
import seaborn as sns
import seaborn.objects as so
import plotnine as pn
 
 
fn = utils.path_data / 'ana_three_dot_predicting_individual_beh_profile/prediction.csv'
df = pd.read_csv(fn)
df['accuracy'] = df['accuracy']=='correct'
predVar = 'auc_pred_accuracy'
df_ = df.copy()
df_ = df_.melt(['participant', 'actual control', 'angular bias', 'group'], ['accuracy', predVar], 
               var_name='acc_type',
               value_name='acc')
df_ = df_.groupby(['participant', 'actual control', 'angular bias', 'group', 'acc_type']).mean()
df_ = df_.reset_index() 


df_['angular bias'] = df_['angular bias'].astype('str')
df_['actual control'] = df_['actual control'].astype('str')
df_2 = df_.copy()
df_2['condition'] = df_2['actual control'] + "_" + df_2['angular bias']
df_2.head()


df_['angular bias'] = df_['angular bias'].astype('str')
df_['actual control'] = df_['actual control'].astype('str')

gap = 0.1
byVar = ['acc', 'actual control', 'angular bias', 'acc_type']

g = (
    so.Plot(df_, x='acc_type', y='acc', color='angular bias')
    # .add(so.Dots(alpha=0.2), so.Jitter(x=.05, y=.05))
    # .pair(x=["actual control", 'angular bias'])
    # .facet(row="group")    
    .add(so.Line(alpha=0.2, marker='.'), so.Agg(),  so.Jitter(x=.05, y=.05), group="participant")
    # .add(so.Range(), so.Est(errorbar="se"))
    .add(so.Dot(pointsize=10), so.Agg(), so.Dodge(), marker='acc_type', fill='acc_type')
    .add(so.Range(), so.Est(errorbar="se"), so.Dodge())
    .facet(row="group", col='actual control')
    .layout(size=(6, 10))
    .limit(y=(-0.1, 1.1))
    # .on(f)
  
)

g

(
    pn.ggplot(df_2, pn.aes(x='condition', y='condition', color='participant')) 
    + pn.geom_line()
    + pn.scale_color_brewer('Spectral')
    
)


pn.scale_shape_manual