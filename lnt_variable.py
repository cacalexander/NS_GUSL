import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb


class LNT:
    def __init__(self, depth, num_tree, mode, saveroot=None, name=None):
        self.svd = []
        self.depth = depth
        self.num_tree = num_tree
        self.mode = mode
        self.saveroot = saveroot
        self.name = name
        return


    def fit(self, x, y):
        print('Mode {}'.format(self.mode))
        if self.mode == 'gpu':
            params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'objective': 'reg:squarederror',
                'max_depth': self.depth,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 1,
                'n_estimators': self.num_tree,
                'learning_rate': 0.07,
            }
        else:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': self.depth,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 1,
                'n_estimators': self.num_tree,
                'learning_rate': 0.07,
            }
        evalset = [(x, y)]
        model = xgb.XGBRegressor(**params)
        model.fit(x, y, verbose = False, eval_set = evalset)
        plt.cla();plt.clf()
        plt.figure()
        
        plt.plot(model.evals_result_['validation_0']['rmse'], label='train')
        plt.savefig(self.saveroot+'LNT')
        plt.cla();plt.clf()
        trees_df = model.get_booster().trees_to_dataframe()
        tree_features = set()
        for tree_index in range(self.num_tree):
            df = trees_df[trees_df['Tree'] == tree_index]
            fs = df[df['Feature'] != 'Leaf']['Feature'].unique()
            fs = tuple(sorted(int(f[1:]) for f in fs))
            tree_features.add(fs)
        tree_features = [np.array(fs) for fs in tree_features]
        print('LNT feature dimension {}'.format(len(tree_features)))

        num_features = x.shape[-1]
        num_tree_features = 0
        for i in range(len(tree_features)):
            selected = tree_features[i]
            selected = [int(idx) for idx in selected]
            if len(selected) > 0:
                num_tree_features += 1
        
        sub_last = int(num_tree_features/2)
        sub = [3]
        sub.extend([i for i in range(5,sub_last,5)])
        sub.append(sub_last)

        count = 0
        feat_subsets = []
        for s in sub:
            feat_subsets.extend(np.array_split(np.asarray(tree_features).reshape(-1),s))
        A = np.zeros((num_features, len(feat_subsets)))
        for i in range(len(feat_subsets)):
            selected = feat_subsets[i]
            selected = [int(idx) for idx in selected]
            if len(selected) > 0:
                X_sel = x[:, selected]

                lin_reg = LinearRegression()
                lin_reg.fit(X_sel, y)

                theta = np.zeros(num_features)
                theta[selected] = lin_reg.coef_

                A[:, count] = theta
                count += 1

        self.svd.append(A)
        return

    def transform(self, x):
        res = []
        for n in range(0, len(self.svd)):
            temp_kernel = self.svd[n]
            res.append(x @ temp_kernel)
        res = np.concatenate(res, axis=-1)
        return res
    