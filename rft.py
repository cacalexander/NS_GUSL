import numpy as np
import matplotlib.pyplot as plt
from rft_lib import Reletive_Feat_Test


class RFT:
    def __init__(self, num_bins=4, name='RFT'):
        self.num_bins = num_bins
        self.name = name
        self.clf = Reletive_Feat_Test()
        return
    
    def fit(self, x, y):
        rft_loss = self.clf.get_rftloss_multidim(x, y, num_bins=self.num_bins)
        return np.array(rft_loss)
    
    def plot_loss_scatter(self, rft_loss_tr, rft_loss_val, saveroot):
        plt.scatter(rft_loss_tr, rft_loss_val)
        plt.xlabel('Train loss')
        plt.ylabel('Validation loss')
        plt.title('RFT loss scatter')
        plt.savefig(saveroot + 'RFT_scatter_' + str(self.name) + '.png')
        plt.close()
        return
    
    def plot_loss_curve(self, rft_loss, name, saveroot):
        loss = rft_loss.copy()
        idx = np.arange(1, len(loss)+1)
        plt.plot(idx, loss)
        loss.sort()
        plt.plot(idx, loss)
        plt.xlabel('Feature index')
        plt.ylabel('RFT loss')
        plt.title('RFT loss curve')
        plt.savefig(saveroot + 'RFT_curve_' + str(name) + '_' + str(self.name) + '.png')
        plt.close()
        return
    
    def plot_loss_hist(self, rft_loss, feat_idx, saveroot):
        sorted_indices = np.argsort(rft_loss)
        rft_loss_sort = rft_loss[sorted_indices]
        feat_idx_sort = feat_idx[sorted_indices]

        # Divide loss into different layers
        loss = []
        for n in range(1, 7):
            temp_idx = np.where(feat_idx_sort==n)[0]
            loss.append(rft_loss_sort[temp_idx])

        plt.plot(np.arange(len(loss[0])), loss[0], label='3x3x3', color='blue')
        plt.plot(np.arange(len(loss[1])), loss[1], label='5x5x5', color='red')
        plt.plot(np.arange(len(loss[2])), loss[2], label='7x7x7', color='yellow')
        plt.plot(np.arange(len(loss[3])), loss[3], label='9x9x9 ', color='black')
        plt.plot(np.arange(len(loss[4])), loss[4], label='xyz coordinate ', color='purple')
        plt.plot(np.arange(len(loss[5])), loss[5], label='interpolation ', color='green')

        plt.xlabel('Feature index')
        plt.ylabel('Loss of tarining set')
        plt.title('RFT loss curve')
        plt.legend()
        plt.savefig(saveroot + 'RFT_curve_seperate_' + str(self.name) + '.png')
        plt.close()
        return