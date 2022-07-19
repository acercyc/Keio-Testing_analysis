# %%
import numpy as np
import pandas as pd
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
import torch
from einops import rearrange, reduce, repeat
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------- #
#                                     info                                     #
# ---------------------------------------------------------------------------- #

# path
path_project = Path(__file__).parent.parent
path_data = path_project / 'data'
path_data_raw = [path_data/'raw'/'Keio Results',
                 path_data/'raw'/'Komagino Results']

def expon(x, a):
    return a * np.exp(-a*x)

class ExpInfo:
    bad_subj = ['K-Reg-H-1', 'K-Reg-H-2', 'K-Reg-S-5']
    taskName = ['one_dot', 'three_dot', 'reaching']

    @staticmethod
    def getScreenSise(df):
        if 'dot-x1' in df.columns:
            screenSize = df.loc[:, 'dot-x1':'dot-y3'].max().max()
        else:
            screenSize = df.loc[:, 'dot-x':'dot-y'].max().max()
        return screenSize

    @staticmethod
    def getSubjIDs():
        files = []
        for datapath in path_data_raw:
            files += glob.glob(str(datapath) + '/*')

        ids = []
        for file in files:
            id = re.search(r'((K-Reg)|(Reg))-(S|H)-\d+', file)
            if id is not None:
                ids.append(id.group())
        ids = set(ids).difference(ExpInfo.bad_subj)
        ids = list(ids)
        ids.sort()
        return ids


class LoadData:
    def __init__(self) -> None:
        pass

    @staticmethod
    def mouseMovement(subjID, task, trialno=None):
        fname = f'{subjID}_{task}.csv'
        fpath = path_data / 'Preprocessing' / 'mouseMovement' / fname
        df = pd.read_csv(fpath)
        if trialno is not None:
            try:
                df = df.loc[df['trialno'].isin(trialno)]
            except:
                try:
                    df = df.loc[df['trialno'] == trialno]
                except:
                    raise ValueError('trialno is not valid')
        return df

    @staticmethod
    def mouseMovementRollingData(subjID='K-Reg-S-18', task='one_dot', wSize=48, interval=1, pos=False, nTrial_val=6,seed=0):
        # load data
        df = LoadData.mouseMovement(subjID, task)
        
        # Split data into train and test
        trial_train, trials_val = DataProcessing.split_train_val_trials(
            df, nTrial_val=nTrial_val, seed=seed)
        df_train = df.query(f'trialno in @trial_train')
        df_val = df.query(f'trialno in @trials_val')
        
        # rolling
        d_train = DataProcessing.rollingWindow_from_df(df_train, wSize, interval, pos=pos)
        d_val = DataProcessing.rollingWindow_from_df(df_val, wSize, interval, pos=pos)

        class TrajDataset(torch.utils.data.Dataset):
            def __init__(self, d):
                self.d = d

            def __len__(self):
                return self.d.shape[0]

            def __getitem__(self, idx):
                return self.d[idx]

        dataset_train = TrajDataset(d_train)
        dataset_val = TrajDataset(d_val)
        return dataset_train, dataset_val

    @staticmethod
    def behaviorData(subjID, task):
        files = []
        for datapath in path_data_raw:
            files += list(datapath.glob('*.*'))

        for file in files:
            if file.match(f'*{subjID}*{task}_results.csv'):
                return pd.read_csv(file, index_col=False)


class DataProcessing:
    
    @staticmethod
    def diff(x, measure='euclidean', offset=1):
        from scipy.spatial import distance
        dist = distance.pdist(x, measure)
        dist = distance.squareform(dist)
        dist = np.diagonal(dist, offset=offset)
        return dist

    @staticmethod
    def split_train_val_trials(df, nTrial_val=6, seed=0):
        trials = set(df['trialno']).difference([0])
        nTrial = len(trials)
        rng = np.random.default_rng(seed)
        trials_val = rng.choice(nTrial, nTrial_val, replace=False)
        trial_train = trials.difference(trials_val)
        return trial_train, trials_val

    @staticmethod
    def rollingWindow(d, wSize=60, interval=1, pos=False):
        """ Rolling window function along time dim
        Args:
            d (np.array): time x feature
            wSize (int): window size
            interval (int): interval size

        Returns:
            np.array: rolling windowed array
        """
        if d.shape[0] < wSize:
            raise ValueError('Data length is shorter than window size')
        d_ = []
        S = 0
        E = S + wSize
        while E <= (len(d)-1):
            d_.append(d[S:E, :])
            S += interval
            E = S + wSize
        d_ = np.stack(d_, axis=0)
        if pos:
            d_ = d_.cumsum(axis=1)        
        return d_

    @staticmethod
    def rollingWindow_from_df(df, wSize, interval=1, pos=False, bySubj=False):
        ''' Run rolling window function based on trial number
        '''
        screensize = ExpInfo.getScreenSise(df)
        d = []
        trials = list(set(df['trialno']).difference([0]))
        for trial in trials:
            df_ = df.query(f'trialno == {trial}').copy()
            df_ = df_[["x-shift", "y-shift"]].values / screensize
            d.append(DataProcessing.rollingWindow(df_, wSize, interval, pos=pos))
        
        if bySubj:
            return d
        else:
            return np.concatenate(d, axis=0)

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    @staticmethod
    def positionEncoding_sincos_mat(nTime, dim=4, max_length=300):
        x = np.arange(nTime) * 2 * np.pi / max_length
        x = np.tile(x, (dim, 1)).T  # t f
        x = x * np.arange(1, dim+1)
        x = np.hstack((np.sin(x), np.cos(x)))
        return x

    @staticmethod
    def seqTrim(x, minTime):
        # x: b t f
        tLen = np.random.randint(minTime, x.shape[1])
        sTime = np.random.randint(0, x.shape[1]-tLen)
        eTime = sTime + tLen
        return x[:, sTime:eTime, :]


class SynthData:
    def __init__(self) -> None:
        pass

    @staticmethod
    def spiral(nTime=72, nBatch=64, seed=0, add_polar=False):
        rng = np.random.default_rng(seed)
        XY = []
        for i in range(nBatch):
            theta = np.linspace(rng.uniform(0.5, 2*np.pi*4),
                                rng.uniform(0.5, 2*np.pi*4), nTime)
            r = np.linspace(rng.uniform(), 1, nTime)
            transform = rng.random((2, 2))

            def polar2z(r, theta):
                compx = r * np.exp(1j * theta)
                xy = np.vstack([np.real(compx), np.imag(compx)]).T
                return xy

            xy = polar2z(r, theta)
            if rng.random() > 0.5:
                r = -r

            xy = polar2z(r, theta)
            if rng.random() > 0.5:
                xy[:, 0] = -xy[:, 0]

            if rng.random() > 0.5:
                xy[:, 1] = -xy[:, 1]

            xy = np.roll(xy, 1, axis=1)
            xy = np.matmul(xy, transform)

            xy = xy / np.max(np.abs(xy))
            XY.append(xy)
        XY = np.stack(XY, 2)
        XY = np.transpose(XY, (2, 0, 1))

        if add_polar:
            x_, y_ = DataProcessing.cart2pol(XY[:, :, 0], XY[:, :, 1])
            x_ = repeat(x_, 'b t -> b t f', f=1)
            y_ = repeat(y_, 'b t -> b t f', f=1)
            XY = np.concatenate([XY, x_, y_], axis=2)
        return XY

    @staticmethod
    def spiral_dataset(**kwargs):
        x = SynthData.spiral(**kwargs)
        return TensorDataset(torch.from_numpy(x))

    @staticmethod
    def sin(nTime=72, nBatch=8, plot=False):
        data = [np.linspace(0, 2*np.pi, nTime),
                np.linspace(np.pi/2, np.pi/2+2*np.pi, nTime)]
        data = np.vstack(data).T
        data = np.sin(data)
        data = np.tile(data, (nBatch, 1, 1))
        data = data + np.random.random(data.shape)/2
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1)
            ax.plot(data[0, :, 1])
            ax.plot(data[0, :, 0])
        return data


class Plot:

    @staticmethod
    def traj_withColour(x, y, fig, ax):
        colors = np.linspace(0, 1, len(x))
        ax.plot(x, y, '-k', alpha=0.2)
        ax.scatter(x, y, c=colors, cmap='turbo')
        ax.plot(x[0], y[0], 'Dr', label='start', markersize=8)
        ax.axis('equal')
        norm = mpl.colors.Normalize(vmin=0, vmax=len(x))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
            cmap='turbo', norm=norm), ax=ax)
        cbar.set_label('Time step')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

    @staticmethod
    def fig2img(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = PIL.Image.fromarray(img)
        return img
    
    # ---------------------------------------------------------------------------- #
    #                           Plot traj_and_Reconstruc                           #
    # ---------------------------------------------------------------------------- #
    @staticmethod
    def traj_and_Reconstruc_from_batch(x, y, x_full=None, fig=None, nSegment=24, nCol=5, cmap='viridis'):
        ''' First order function for plotting trajectory and reconstructed trajectory
        '''
        nBatch = x.shape[0]
        wSize = x.shape[1]
        plot_offset = 0
        
        # compute starting points of segments
        start_idx = np.linspace(0, nBatch, nSegment + 1).astype(int)[:-1]
        nRow = np.ceil((nSegment+1) / nCol).astype(int)
        
        # setup colormap
        if x_full is None:
            t_len = 301
        else:
            t_len = x_full.shape[0]
        colors = np.linspace(0, 1, t_len)
        cmap = mpl.cm.get_cmap(cmap)

        # setup figure
        if fig is None:
            fig = plt.figure(figsize=(4*nRow, 4*nCol))
        ax = fig.subplots(nRow, nCol)
        if ax.ndim == 1:
            ax = ax.reshape(1, -1)

        # plot full trajectory
        if x_full is not None:
            ax[0, 0].plot(x_full[:, 0], x_full[:, 1], '-k', alpha=0.2)
            ax[0, 0].scatter(x_full[:, 0], x_full[:, 1], c=colors, cmap=cmap)
            ax[0, 0].plot(0, 0, 'Dr', label='start', markersize=8)
            ax[0, 0].axis('equal')
            plot_offset = 1

        for i, si in enumerate(start_idx):
            iRow, iCol = np.unravel_index(i+plot_offset, (nRow, nCol))
            
            # plot ground Truth
            ax[iRow, iCol].scatter(x[si, :, 0], x[si, :, 1], c=cmap(colors[si:si+wSize]), marker='o')
            ax[iRow, iCol].plot(x[si, :, 0], x[si, :, 1], 'k', alpha=0.5)
            ax[iRow, iCol].axis('equal')
            
            # plot reconstructed
            ax[iRow, iCol].plot(y[si, 0, 0], y[si, 0, 1], 'ro', mfc='none', markersize=10)
            ax[iRow, iCol].plot(y[si, :, 0], y[si, :, 1], color='red', alpha=0.5)
            ax[iRow, iCol].plot(y[si, :, 0], y[si, :, 1], '.', color='red', alpha=0.5)
            ax[iRow, iCol].axis('equal')
            ax[iRow, iCol].set_title(f'{si/60:.1f}s~{(si+wSize)/60:.1f}s')

        return fig, ax

    @staticmethod
    def traj_and_Reconstruc_from_trial(df, trialno, model, wSize=30, **kwargs):
        '''
        Second order function for plotting trajectory and reconstructed trajectory
        Model is run at this level to get the reconstructed trajectory
        '''
        # extract data
        df = df.query(f'trialno == {trialno}')
        x = DataProcessing.rollingWindow_from_df(df, wSize, 1)
        
        # run reconstruction
        model.eval()
        x_ = torch.from_numpy(x).double()
        y = model(x_).detach().cpu().numpy()
        
        # cumsum
        x_cum = x.cumsum(axis=1)
        y_cum = y.cumsum(axis=1)
        x_full = df[['x-shift', 'y-shift']].values
        x_full = x_full.cumsum(axis=0)
        
        return Plot.traj_and_Reconstruc_from_batch(x_cum, y_cum, x_full=x_full, **kwargs)
    
    @staticmethod
    def traj_and_Reconstruc_quick_check(subj, task, trialno, path='TrajNet_train', model_type='val', **kwargs):
        '''Third order function for plotting trajectory and reconstructed trajectory
        '''
        # load data
        df = LoadData.mouseMovement(subj, task)
        
        # load model
        model = Model.load(subj=subj, task=task, path=path, model_type=model_type)
        return Plot.traj_and_Reconstruc_from_trial(df, trialno=trialno, model=model, **kwargs)


    @staticmethod
    def traj_and_Reconstruc(x, y, ax, legend=True):
        """ plot trajectory and reconstructed trajectory simple version 
        Args:
            x: Ground true trajectory
            y: Reconstructed
            ax: matplotlib axis
        """

        x = np.vstack([np.zeros((1, 2)), x])
        y = np.vstack([np.zeros((1, 2)), y])
        ax.plot(x[:, 0], x[:, 1], '-')
        ax.plot(y[:, 0], y[:, 1], '-')
        ax.plot(0, 0, 'or')
        ax.axis('equal')
        if legend:
            ax.legend(['Ground true trajectory', 'Reconstructed trajectory', 'orig'],
                    bbox_to_anchor=(1.05, 1), loc=2)


class Model:
    
    @staticmethod
    def load(subj='K-Reg-S-18', task='one_dot', model_type='val', path='TrajNet_train'):
        ''' Load model from checkpoint
        '''
        import TrajNet_train
        model = TrajNet_train.PL_model()
        path_cp = path_data / path / f'{subj}_{task}_{model_type}.ckpt'
        model = model.load_from_checkpoint(path_cp).double().eval()
        return model
    


class Analysis:
    @staticmethod
    def pca(x, n_components=None, plot_explained_variance=False):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(x)
        if plot_explained_variance:
            n = len(pca.explained_variance_ratio_)
            plt.bar(range(n), pca.explained_variance_ratio_.cumsum())
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.show()
        return pca.transform(x)
    
    @staticmethod
    def fit_function(x, y, fun=expon, plot=False):
        from scipy.optimize import curve_fit
        para = curve_fit(fun, x, y, 0.5)
        if plot:
            plt.plot(x, fun(x, *para[0]))
            plt.plot(x, y)
        return para
    
    @staticmethod
    def pca(x, n_components=None, normalise=True, plot_explained_variance=False):
        from sklearn.decomposition import PCA
        if normalise:
            from sklearn.preprocessing import scale
            x = scale(x, axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(x)
        if plot_explained_variance:
            n = len(pca.explained_variance_ratio_)
            plt.bar(range(n), pca.explained_variance_ratio_.cumsum())
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.show()
        return pca
    