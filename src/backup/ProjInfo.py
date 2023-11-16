
#%%
from pathlib import Path
path_project = Path(__file__).parent.parent
path_raw = [path_project / 'data/raw/Keio Results/',
            path_project / 'data/raw/Komagino Results/']
bad_subj = ['K-Reg-H-1', 'K-Reg-H-2', 'K-Reg-S-5']
taskList = ['one_dot', 'three_dot', 'reaching']