# v2021.05.24
# data transformations

import numpy as np
from sklearn.decomposition import PCA
import sklearn
# print('The scikit-learn version is {}.'.format(sklearn.__version__))


def pqr_decomposition(data, forma='PQR', color_pca=None, bias=None):
    '''
    data: input RGB image (channel last: N,H,W,C)
    forma: 'PQR','PQ','P','Q'
    color_pca: the trained PCA model
    bias: the bias term (scalar) to make the response positive
    '''

    if data.shape[1] < 5:
        data = np.moveaxis(data, 1, 3)
    new_data = np.zeros((data.shape))
    data_tr = data[np.array(range(0,len(data),10))]
    if color_pca is None:
        color_pca = PCA(n_components=3, svd_solver='full').fit(
            data_tr.reshape(-1, 3))
    new_data = color_pca.transform(data.reshape(-1, 3)).reshape(new_data.shape)
    # new_data = abs(new_data)
    if bias is None:
        # shift up the negative values (actually it doesn't matter, if a minmax normalization is done after that)
        bias = -1*np.min(new_data)
    new_data += bias

    if forma == 'PQ':
        new_data = new_data[:, :, :, :2]
    elif forma == 'P':
        new_data = new_data[:, :, :, [0]]
    elif forma == 'Q':
        new_data = new_data[:, :, :, [1]]
    elif forma == 'PQR':
        new_data = new_data[:, :, :]

    return new_data, color_pca, bias


def minmax_normalize(data, single=False):
    '''
    single: normalize each single image or not
    '''
    if single == False:
        data_new = data - data.min()
        return data_new/(data_new.max()+1e-5)
    elif single == True:
        data_new = np.zeros((data.shape))
        for n in range(data.shape[0]):
            data_new[n] = data[n] - data[n].min()
            data_new[n] = data_new[n]/(data_new[n].max()+1e-5)
        return data_new


# %%
if __name__ == "__main__":
    from keras.datasets import cifar10
    import matplotlib.pyplot as plt

    (tr_rgb_img, _), (te_rgb_img, _) = cifar10.load_data()
    print(tr_rgb_img.shape)
    # use only 5000 images as an example
    tr_PQR, color_pca_model, bias_term = pqr_decomposition(
        tr_rgb_img[:5000], forma='P', color_pca=None, bias=None)
    te_PQR, _, _ = pqr_decomposition(
        te_rgb_img, forma='P', color_pca=color_pca_model, bias=bias_term)

    tr_PQR = minmax_normalize(tr_PQR, single=True)
    te_PQR = minmax_normalize(te_PQR, single=True)

    print(color_pca_model.explained_variance_ratio_)

    plt.imshow(te_rgb_img[0])
    plt.show()
    plt.imshow(te_PQR[0].squeeze(), cmap='gray')
    plt.show()
