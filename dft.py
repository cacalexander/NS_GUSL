import numpy as np
from lib_loss import cal_weighted_H
from tqdm import tqdm
import matplotlib.pyplot as plt

class Disc_Feature_Test():
    def __init__(self, num_class=10, num_Candidate=16, loss='entropy'):
        self.num_class = num_class
        self.B_ = num_Candidate
        self.loss = loss
        self.loss_list = []

    def binning(self,x,y):
        if np.max(x) ==  np.min(x):
            return 1 

        # B bins (B-1) candicates of partioning point
        candidates = np.arange(np.min(x),np.max(x),(np.max(x)-np.min(x))/(self.B_))
        candidates = candidates[1:]
        candidates = np.unique(candidates)

        loss_i = np.zeros(candidates.shape[0])
        # if self.loss == 'cross_entropy':
        #     for idx in range(candidates.shape[0]):
        #         loss_i[idx] = cal_weighted_CE(x, y, candidates[idx],num_cls=self.num_class)
        # elif self.loss == 'entropy':
        for idx in range(candidates.shape[0]):
            loss_i[idx] = cal_weighted_H(x, y, candidates[idx],num_cls=self.num_class)

        best_loss = np.min(loss_i)

        return best_loss

    def loss_estimation(self, x, y):
        x = x.astype('float64')
        y = y.astype('int64')
        y = y.squeeze()
        minimum_loss = self.binning(x.squeeze(), y)
        return minimum_loss

    def get_all_loss(self, X, Y):
        '''
        Parameters
        ----------
        X : shape (N, P).
        Y : shape (N).

        Returns
        -------
        feat_loss: DFT loss for all the feature dimensions. The smaller, the better.
        '''
        feat_loss = np.zeros(X.shape[-1])
        for k in tqdm(range(X.shape[-1])):
            feat_loss[k] = self.loss_estimation(X[:,[k]], Y)
        return feat_loss
    
    


def feature_selection(tr_X, tr_y, FStype='DFT_entropy', thrs=1.0, B=16):
    """
    This is the main function for feature selection using DFT.
    
    Parameters
    ----------
    tr_X : shape (N, P).
    tr_y : shape (N).
    FStype: feature selection criteria
    thrs: the percentage of kept dimension (0-1), K = thrs*P
    B: the number of bins. Default=16.

    Returns
    -------
    selected_idx: selected feature dimension index based on thrs;
    feat_score: the feature importance/ DFT loss for each of the P dimensions.
    """
    
    NUM_CLS = np.unique(tr_y).size
    if FStype == 'DFT_entropy': # lower the better # more loss options will be added
        dft = Disc_Feature_Test(num_class=NUM_CLS, num_Candidate=B, loss='entropy')
        feat_score = dft.get_all_loss(tr_X, tr_y)
        feat_sorted_idx = np.argsort(feat_score)

    selected_idx = feat_sorted_idx[:int(thrs*feat_score.size)]

    return selected_idx, feat_score


def plot_loss_scatter(rft_loss_tr, rft_loss_val, name, saveroot):
    plt.scatter(rft_loss_tr, rft_loss_val)
    plt.xlabel('Train loss')
    plt.ylabel('Validation loss')
    plt.title('DFT loss scatter')
    plt.savefig(saveroot + f'DFT_scatter_{name}' + '.png')
    plt.close()
    return
    
def plot_loss_curve(rft_loss, name, saveroot):
    loss = rft_loss.copy()
    idx = np.arange(1, len(loss)+1)
    plt.plot(idx, loss)
    loss.sort()
    plt.plot(idx, loss)
    plt.xlabel('Feature index')
    plt.ylabel('DFT loss')
    plt.title('DFT loss curve')
    plt.savefig(saveroot + 'DFT_curve_' + str(name) + '_' + '.png')
    plt.close()
    return