import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, '/home/cc98905/NS_GUSL/patch_alignment')

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np 
import cv2
import random
#import pywt
import itertools
from itertools import combinations
#from codecarbon import EmissionsTracker
from numba import njit
from RFT_new import FeatureTest

from HE_Norm import normalizeStaining, normalizeStaining_aug
from skimage.measure import block_reduce
import mahotas
from data_patch_loader import image_patch_loader, generate_patches, image_patch_loader_test, generate_patches_new, image_patch_loaderc, image_patch_loadercons, image_patch_loadercpm, image_patch_loadertnbc
from skimage import exposure, morphology
from skimage import io, color
from skimage.feature import hog
from skimage.feature import blob_dog, blob_log, blob_doh
from lib_data_transformation import pqr_decomposition, minmax_normalize
#from rft import RFT
from dft import feature_selection, plot_loss_curve, plot_loss_scatter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from skimage.util import view_as_windows, view_as_blocks
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import binary_fill_holes
from scipy.stats import entropy

import xgboost as xgb
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage import measure, segmentation
from skimage.feature import peak_local_max
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import matplotlib.pyplot as plt
import pickle
import gc
import time
from mean_shift_post_processing_v2 import hole_fill, postProc2

from utils import load_model, save_model, binarize, truncation, duplicate_1d, calculate_mse, find_aji, \
    duplicate, select_comb, plot, plot_residue, energy_cal, plot_scatter, thresh, upscale_pred, rec_img, \
    add_to_hue, add_to_brightness, add_to_contrast, add_to_saturation, gaussian_blur, median_blur, find_aji_w, lnt_kernel, find_aji_m, rec_imgo, plot_train_val_rank, upsample, dice_coef, \
    random_brightness_shift, random_gamma, random_contrast, update_pred, recover_fn, find_aji32, rec_img512
from xgboost_funcs import train_iter, train_clf, train_iter_init, train_clf_inst
from mean_shift_post_processing_v2 import remove_noise
from feat_extractor1 import Shrink, Concat, im_to_blocks, im_to_blocks_pad, im_to_blocks_te, channels, \
    channels_test, get_feat, save_feat, add_neighbor_feat,Shrink_patch, im_to_blocks_padte, im_to_blocks_gtte, \
    im_to_blocks_gtte16, im_to_blocks_padte16
from lnt_variable import LNT
from sklearn.metrics import precision_score, recall_score, f1_score
#from patch_alignment.class_data_modifier import DATA_MODIFIER
from mean_shift_post_processing_v2 import recover_img_patch, recover_img_patcho
import utils
from torch_SSL.torch_ssl import sslModel, torchSaab

class Model:
    def __init__(self, args):
        self.args = args
        
        self.pixelhop_all = dict()
        self.pixelhop_prev = dict()
        self.pixelhop_haar = dict()
        self.rft_all = dict()
        self.dft_all = dict()
        self.lnt_all = dict()
        self.xgb_all = dict()
        self.pred_tr = dict()
        self.pred_tra = dict()
        self.pred_te = dict()
        self.feat_mean = dict()
        self.all_combs = dict()
        self.feat_std = dict()
        self.thresh = dict()
        self.eps = dict()
        self.xgb_clf = None
        self.K_Means = None
        self.sel_entropy = None
        
    def load_model(self):
        model_paths = self.args['model_paths']
        for i, path in enumerate(model_paths):
            if not os.path.exists(path):
                print(f'Path {path} not exists')
                break
            level = load_model(path)
            self.pixelhop_all[f'inplevel_{i + 1}'] = level['pixelHop']
            self.rft_all[f'level_{i + 1}'] = level['rft']
            self.xgb_all[f'level_{i + 1}'] = level['xgb']
            #self.lnt_all[f'level_{i + 1}'] = level['lnt']
            self.pixelhop_prev[f'prevlevel_{i + 1}'] = level['pixelHop_prev']
            self.K_Means = level['Kmeans']
            self.thresh = level['thresh']
            
    def load_level(self, model_path, level):
        
            if not os.path.exists(model_path):
                print(f'Path {model_path} not exists')
                return
            else:
                model_l = load_model(model_path)
                if('pred_tr' in model_l.keys()):
                    self.pred_tr[f'level_{level}'] = model_l['pred_tr']
                self.pred_te[f'level_{level}'] = model_l['pred_te']
                if(level!=0):
                    self.pixelhop_all[f'inplevel_{level}'] = model_l['pixelHop']
                    self.rft_all[f'level_{level}'] = model_l['rft']
                    self.xgb_all[f'level_{level}'] = model_l['xgb']
                    if('lnt' in model_l.keys()):
                       self.lnt_all[f'level_{level}'] = model_l['lnt']
                    if('all_combs' in model_l.keys()):
                       self.all_combs[f'level_{level}'] = model_l['all_combs']
                    self.pixelhop_prev[f'prevlevel_{level}'] = model_l['pixelHop_prev']
                    #self.pixelhop_haar[f'{level}'] = model_l['pixelHop_haar']
                    #self.K_Means = model_l['Kmeans']
                    self.feat_mean[f'level_{level}'] = model_l['feat_mean']
                    self.feat_std[f'level_{level}'] = model_l['feat_std']
                    #self.thresh[f'level_{level}'] = model_l['thresh']
                if(level ==0):
                    self.rft_all[f'level_init'] = model_l['rft']
                    self.lnt_all[f'level_init'] = model_l['lnt']
                    self.xgb_all[f'initlevel_{4}'] = model_l['xgb']
                    self.feat_mean[f'init_{4}'] = model_l['feat_mean']
                    self.feat_std[f'init_{4}'] = model_l['feat_std']
                    self.pixelhop_all[f'inplevel_{4}_init'] = model_l['pixelHop']
                    self.pixelhop_all[f'inplevel_{4}'] = model_l['pixelHop_inp']
                    if('pixelHop_prev' in model_l.keys()):
                        self.pixelhop_prev[f'prevlevel_{4}'] = model_l['pixelHop_prev']
    
    def load_level_clf(self, model_path, level):
        
            if not os.path.exists(model_path):
                print(f'Path {model_path} not exists')
                return
            else:
                model_l = load_model(model_path)
                self.rft_all[f'level_clf'] = model_l['rft']
                self.lnt_all[f'level_clf'] = model_l['lnt']
                self.xgb_clf1 = model_l['xgb']
                self.pixelhop_all[f'inplevel_clf'] = model_l['pixelHop']
                if('pixelHop_prev' in model_l.keys()):
                    self.pixelhop_prev[f'prevlevel_clf'] = model_l['pixelHop_prev']

    def save_level(self, cur_level, save_path):
        if(cur_level>0):
            res = {
                'pixelHop': self.pixelhop_all[f'inplevel_{cur_level}'],
                'rft': self.rft_all[f'level_{cur_level}'],
                'xgb': self.xgb_all[f'level_{cur_level}'],
                'lnt': self.lnt_all[f'level_{cur_level}'],
                'pixelHop_prev': self.pixelhop_prev[f'prevlevel_{cur_level}'],
                #'pixelHop_haar': self.pixelhop_haar[f'{cur_level}'],
                'Kmeans': self.K_Means, 
                'pred_tr' : self.pred_tr[f'level_{cur_level}'],
                'pred_te' : self.pred_te[f'level_{cur_level}'], 
                'feat_mean' : self.feat_mean[f'level_{cur_level}'], 
                'feat_std' : self.feat_std[f'level_{cur_level}'], 
                #'thresh' : self.thresh[f'level_{cur_level}']
                #'args' : self.args
                #'all_combs': self.all_combs[f'level_{cur_level}']
                
            }
        else:
            res = {
                'pred_tr' : self.pred_tr[f'level_{cur_level}'],
                'pred_te' : self.pred_te[f'level_{cur_level}'],
                'pixelHop': self.pixelhop_all[f'inplevel_{4}_init'],
                'pixelHop_inp': self.pixelhop_all[f'inplevel_{4}'],
                'pixelHop_prev': self.pixelhop_prev[f'prevlevel_{4}'],
                'feat_mean' : self.feat_mean[f'init_{4}'], 
                'feat_std' : self.feat_std[f'init_{4}'],
                'xgb': self.xgb_all[f'initlevel_{4}'],
                'rft': self.rft_all[f'level_init'],
                'lnt': self.lnt_all[f'level_init']
                #'args': self.args
            }
        save_model(res, os.path.join(save_path, f"level_{cur_level}.pkl"))
        self.args['model_paths'].append(os.path.join(save_path, f"level_{cur_level}.pkl"))

    def save_level_clf(self, cur_level, save_path):
        res = {
                
                'pixelHop': model.pixelhop_all[f'inplevel_clf'],
                'xgb': model.xgb_clf1,
                'pixelHop_prev': model.pixelhop_prev[f'prevlevel_clf'],
                'rft': self.rft_all[f'level_clf'],
                'lnt': self.lnt_all[f'level_clf']
                
            } 
        save_model(res, os.path.join(save_path, f"level_clf.pkl"))
    
    def patch_alignment(self, X):
        input_patches_ori = X.copy()
        data_modifier = DATA_MODIFIER()
        input_patches_updated, grad_hist_arr_ori, grad_hist_arr_updated, ud_idx, rot_idx = \
        data_modifier.align_patches(input_patches_ori,
                        mode='alignedByGradHist',
                        get_source_arr_updated_flag=True)
        
        X_aug = np.concatenate([X, input_patches_updated], axis = 0)
        

        return X_aug, ud_idx, rot_idx
    
    def patch_alignment_test(self, X, ud_idx, rot_idx):
        input_patches_ori = X.copy()
        data_modifier = DATA_MODIFIER()
        input_patches_updated= \
        data_modifier.align_patches(input_patches_ori,
                        mode='alignedByGradHisttest',
                        ud_idx = ud_idx, rot_idx = rot_idx)
        
        X_aug = np.concatenate([input_patches_ori, input_patches_updated], axis = 0)
        

        return X_aug



    def energy_split_train(self, X ,prev_pred, level):
        if(level =='clf'):
            ld = 1
            win = 7
        else: 
            ld = 2**(level-1)  
            wind = [5,5,5,5]
            win = wind[level-1]
        #win = self.args['win'][level-1]
        X_shrink = utils.Shrink_patch(X, ld, win , 1, int(win//2))
        X_shrinkr = X_shrink.reshape(-1, win*win)
        pp_shrink = utils.Shrink_patch(prev_pred, 1, win , 1, int(win//2))
        pp_shrinkr = pp_shrink.reshape(-1, win*win)
        #res = (1-X_shrinkr) - pp_shrinkr
        dc = np.mean(pp_shrinkr, axis = 1, keepdims = True)
        X_ac = pp_shrinkr - dc
        energy = energy_cal(X_ac)
        #_, eva = pca_cal(X_shrink)
        #eva = eva/ np.sum(eva)
        inds = np.argsort(energy)

        energy_s = energy[inds]
        thresh = energy_s[int(len(inds)/2)]
        #thresh = 0.12
        """
        p = prev_pred.reshape(-1)
        uni = (p >= 0.25) & (p <= 0.75) 
        u = np.where(uni)[0]
        g = np.where(energy<=thresh)[0]
        g1 = np.intersect1d(g,u)
        u = np.where(uni==False)[0]
        g3 = np.intersect1d(g,u)
        g2 = np.where(energy>thresh)[0]
        """
        g1 = np.where(energy<=thresh)[0]
        g2 = np.where(energy>thresh)[0]
        
        #g1 = inds[np.where(energy<=thresh)[0]]
        #g2 = inds[np.where(energy>thresh)[0]]
        
        #dm = DATA_MODIFIER()
        #patches_0, patches_1 = dm.partition(X_shrink[g2].copy())

        return [g1, g2], thresh

        print(" ")

    def energy_split_test(self, X ,prev_pred, level, thresh,mode):
        if(level =='clf'):
            ld = 1
            win = 7
        else: 
            ld = 2**(level-1)  
            wind = [5,5,5,5]
            win = wind[level-1]
        #win = self.args['win'][level-1]
        if(mode == 'orig'):
            X_shrink = utils.Shrink_patch(X, ld, win , 1, int(win//2))
        #elif(mode =='prev' ):
        #    X_shrink = utils.Shrink_patch(X, 1, win, 1, int(win//2))
        #elif(mode =='inp'):
        #    X_shrink = utils.Shrink_patch(X, 1, win , 1, 0)
        X_shrinkr = X_shrink.reshape(-1, win*win)
        pp_shrink = utils.Shrink_patch(prev_pred, 1, win , 1, int(win//2))
        pp_shrinkr = pp_shrink.reshape(-1, win*win)
        #res = (1-X_shrinkr) - pp_shrinkr
        dc = np.mean(pp_shrinkr, axis = 1, keepdims = True)
        X_ac = pp_shrinkr - dc
        energy = energy_cal(X_ac)
        #_, eva = pca_cal(X_shrink)
        #eva = eva/ np.sum(eva)
        #inds = np.argsort(energy)
        #energy_s = energy[inds]
        #thresh = energy_s[int(len(inds)/2)]
        #thresh = 0.12
        """
        p = prev_pred.reshape(-1)
        uni = (p >= 0.25) & (p <= 0.75) 
        u = np.where(uni)[0]
        g = np.where(energy<=thresh)[0]
        g1 = np.intersect1d(g,u)
        u = np.where(uni==False)[0]
        g3 = np.intersect1d(g,u)
        g2 = np.where(energy>thresh)[0]
        """
        
        g1 = np.where(energy<=thresh)[0]
        g2 = np.where(energy>thresh)[0]
        
        #g1 = inds[np.where(energy<=thresh)[0]]
        #g2 = inds[np.where(energy>thresh)[0]]
        #dm = DATA_MODIFIER()
        #patches_0, patches_1 = dm.partition(X_shrink[g2].copy())

        return [g1, g2]

        print(" ")

    def train_level_haar(self, level, X0, X, Y, prev_pred = None, sel = None):
        win = self.args['win'][level-1]
        ld = 2**(level-1)           
            
        Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
        idx = np.arange(0, len(X), 1)
        if(self.args['downsample_patches'][level-1]!=1 and sel == None):
            sel = random.sample(idx.tolist(), int(self.args['downsample_patches'][level-1]*len(X)))
            X_sel = X[sel]; Y_sel = Y[sel]; prev_pred_sel = prev_pred[sel]; X0_sel = X0[sel]
            res = Y_sel - duplicate(prev_pred_sel)
        else:
            X_sel = X; Y_sel = Y; prev_pred_sel = prev_pred; X0_sel = X0

        X_samples = utils.Shrink_patch(X0_sel, ld, win, 1, int(win//2))  
        _, ud_idxs, rot_idxs = self.patch_alignment(X_samples)
        self.pixelhop_all[f'inplevel_{level}'] = dict()
        self.pixelhop_prev[f'prevlevel_{level}'] = dict()
        self.pixelhop_haar[f'{level}'] = dict()
        self.rft_all[f'level_{level}'] = dict()
        self.lnt_all[f'level_{level}'] = dict()
        self.xgb_all[f'level_{level}'] = dict()
        self.feat_mean[f'level_{level}'] = dict(); self.feat_std[f'level_{level}'] = dict()

        if(level<4):
                prev_pred_sel = duplicate(prev_pred_sel)
        grp,self.thresh[f'level_{level}'] = self.energy_split_train(X0_sel,np.expand_dims(prev_pred_sel,-1), level) 
        img_grp = np.zeros(len(X0_sel.reshape(-1)))
        img_grp[grp[0]] = 0; img_grp[grp[1]] = 1
        img_grp = img_grp.reshape(X0_sel.shape)
        X0_block = np.squeeze(view_as_blocks(X0_sel, (1,2,2,1)))
        blocks = view_as_blocks(img_grp, (1,2,2,1)).reshape(-1, 2, 2)
        grp_blocks = np.where(np.sum(blocks.reshape(-1,4), axis = -1)==0, 0,1)
        grp_blocks = grp_blocks.reshape(X0_block.shape[0],X0_block.shape[1],X0_block.shape[2])
        grp_blocks = duplicate(np.squeeze(grp_blocks))
        
        g0 = np.where(grp_blocks.reshape(-1) == 0); g1 = np.where(grp_blocks.reshape(-1) ==1)
        g = [g0,g1]
        g_name = 'grp_0'
        ud_idx = None; rot_idx = None;
        Y_selg = Y_sel.copy()
        """
        print(f"Training Group {g_name}")
        self.Train_PixelHop(level, X_sel, X0_sel,np.expand_dims(prev_pred_sel,-1), 'inp', 3, self.args['win'][level - 1],grp, g_name, ud_idx, rot_idx)
        feat_inp = self.Get_PixelHopFeat(level, X_sel, 'inp', 0,self.args['win'][level - 1],grp,g_name , ud_idx, rot_idx)
        feat_inp = self.Add_SpatialFeat(feat_inp, X0_sel, level,self.args['win'][level - 1],'inp', ud_idx, rot_idx )
        print(f"Shape of input features: {feat_inp.shape}")
        N,h,w,c = feat_inp.shape
        feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g0]
                
        self.Train_PixelHop(level, np.expand_dims(prev_pred_sel,-1),X0_sel,np.expand_dims(prev_pred_sel,-1), 'prev', 3, self.args['win'][level - 1],grp, g_name,  ud_idx, rot_idx)
        feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred_sel,-1), 'prev', 0,self.args['win'][level - 1],grp, g_name, ud_idx, rot_idx )
        feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred_sel,-1), level,self.args['win'][level - 1],'prev' , ud_idx, rot_idx)
        print(f"Shape of prev pred features: {feat_prev.shape}")
        feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g0]
        feat = np.concatenate([feat_inp, feat_prev], axis = -1)

        
        prev_pred_selg = prev_pred_sel.copy()
        Y_selg = Y_selg.reshape(-1)[g0]; prev_pred_selg = prev_pred_selg.reshape(-1)[g0]
        res = Y_selg - prev_pred_selg
        feat_idx = np.arange(0, feat.shape[-1], 1)

        #RFT 
                
        self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][0], f'level_{level}_{g_name}')
        feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
        
        #LNT

        feat_tr = feat
        #XGBoost 
        self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
        self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
        feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
        print(f"Shape of features after RFT {feat_tr.shape}")
        feat_lnt = None
        gc.collect()
        feat_prev = None
        gc.collect()
        feat_inp = None
        gc.collect()
        feat = None 
        gc.collect()
        self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg, Y_selg, level, g_name)
        """
        g_name = 'grp_1'
        print(f"Training Group {g_name}")
        #X0_eq = exposure.equalize_adapthist(1-X0_sel)
        X0_eq = block_reduce(X0_sel, (1, 2**(level-1), 2**(level-1),1), np.mean)
        X0_eq = 1-X0_eq
        resd = np.squeeze(X0_eq) - prev_pred_sel
        #Y_haar = np.squeeze(view_as_blocks(Y_selg, (1,2,2)))
        #Y_haar = Y_haar.reshape(-1, 2 ,2)
        #prev_pred = np.squeeze(view_as_blocks(prev_pred_sel, (1,2,2)))
        #prev_pred = prev_pred.reshape(-1, 2 ,2)
        res_haar = Y_sel - prev_pred_sel
        grp2 = np.squeeze(view_as_blocks(grp_blocks, (1,2,2)))
        grp2 = grp2.reshape(-1, 2 ,2)
        resd = np.squeeze(utils.Shrink_patch(np.expand_dims(resd,-1), 1, 6, 2, 2))
        #resd = np.squeeze(utils.Shrink_patch(np.expand_dims(resd,-1), 1, 6, 2, 2))
        res_haar = np.squeeze(utils.Shrink_patch(np.expand_dims(res_haar,-1), 1, 6, 2, 2))
        wt_0 = []; wt_g = []
        y_0 = []; y_g = []
        g1_ind = []
        for n in range(len(grp2)):
            if(np.sum(grp2[n])==4):
                g1_ind.append(n)
                """
                coeff = pywt.dwt2(resd[n], 'haar')
                wt_0.append(np.squeeze(coeff[0]))
                wt_g.append(np.squeeze(coeff[1]))
                coeff_y = pywt.dwt2(res_haar[n], 'haar')
                y_0.append(np.squeeze(coeff_y[0]))
                y_g.append(np.squeeze(coeff_y[1]))
                """

        coeff = pywt.dwt2(resd, 'haar')
        wt_0 = np.squeeze(coeff[0])
        wt_g = np.squeeze(coeff[1])
        coeff_y = pywt.dwt2(res_haar, 'haar')
        y_0 = np.squeeze(coeff_y[0])
        y_g = np.squeeze(coeff_y[1])
        wt_0 = np.asarray(wt_0);wt_g = np.asarray(wt_g)
        y_0 = np.asarray(y_0);y_g = np.asarray(y_g)

        self.Train_PixelHop(level, np.expand_dims(resd,-1), X0_sel,np.expand_dims(prev_pred_sel,-1), 'patches', 3, self.args['win'][level - 1],grp, 'resd_1', ud_idx, rot_idx)
        feat_inp = self.Get_PixelHopFeat(level, np.expand_dims(resd,-1), 'patches', 0,self.args['win'][level - 1],grp,'resd_1' , ud_idx, rot_idx)
        feat_inp = self.Add_SpatialFeat(feat_inp, resd, level,self.args['win'][level - 1],'patches', ud_idx, rot_idx )
        feat_inp = feat_inp
        print(f"Shape of input features: {feat_inp.shape}")
        #N,h,w,c = feat_inp.shape
        #feat_inp = feat_inp.reshape(-1, h*w)

        for chan in range(4):
            print(f'Feature Extraction Channel {chan}')
            if(chan ==0):
                X_chan = wt_0
            else:
                X_chan = wt_g[chan-1]
            g_name = f'chan_{chan}1'
            self.Train_PixelHop(level, np.expand_dims(X_chan,-1), X0_sel,np.expand_dims(prev_pred_sel,-1), 'patches', 3, self.args['win'][level - 1],grp, g_name, ud_idx, rot_idx)
            feat_chan = self.Get_PixelHopFeat(level, np.expand_dims(X_chan,-1), 'patches', 0,self.args['win'][level - 1],grp,g_name , ud_idx, rot_idx)
            feat_chan = self.Add_SpatialFeat(feat_chan, X_chan, level,self.args['win'][level - 1],'patches', ud_idx, rot_idx )
            print(f"Shape of input features: {feat_chan.shape}")
            #N,h,w,c = feat_chan.shape
            #feat_chan = feat_chan.reshape(-1, h*w)
            feat_inp = np.concatenate([feat_inp, feat_chan], axis = -1)
            feat_chan = None
            gc.collect()


        #LNT
        g_name = 'grp_1'
        if(self.args['lnt'][level-1]):
            self.Train_LNT(feat_inp, res, level, g_name)
            feat_lnt = self.Get_LNT(feat_inp, level, g_name)
            feat_tr = np.concatenate([feat_inp, feat_lnt], axis = -1)
            plot(feat_inp, res, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
            feat_idx = np.arange(0, feat_tr.shape[-1], 1)
            self.rft_feat_selection(level, feat_tr, res,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}LNT_{g_name}')
            feat_tr = feat_tr[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
        else:
            feat_tr = feat_inp
        #XGBoost 
        self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
        self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
        feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
        print(f"Shape of features after RFT {feat_tr.shape}")
        feat_lnt = None
        gc.collect()
        feat_prev = None
        gc.collect()
        feat_inp = None
        gc.collect()
        feat = None 
        gc.collect()

        for chan in range(4):
            g_name = f'chan_{chan+1}'
            print(f'XGBoost Training {g_name}')
            if(chan == 0):
                Y_haar = y_0[:,2,2]
            else: 
                Y_haar = y_g[chan-1,:,2,2]
            Y_haar = np.asarray(Y_haar)
            feat_idx = np.arange(0, feat_tr.shape[-1], 1)
            self.rft_feat_selection(level, feat_tr, Y_haar,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], 0.25, f'level_{level}_{g_name}')
            feat_rft = feat_tr[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            
            self.Train_XGB_Haar(feat_rft.reshape(-1, feat_rft.shape[-1]), Y_haar, level, g_name)

            
        feat_inp = None
        gc.collect()
        feat_tr = None
        gc.collect()
        print("")
        return sel

    def test_level_haar(self, level, X0, X, Y, mode, prev_pred = None):

        ud_idx = None
        rot_idx = None
        Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            
        pred = np.zeros(len(Y.reshape(-1)))
        if(level <4):
            prev_pred = duplicate(prev_pred)
        pred = prev_pred.copy()
        pred = pred.reshape(-1)
        grp = self.energy_split_test(X0,np.expand_dims(prev_pred,-1), level,'orig')
        img_grp = np.zeros(len(X0.reshape(-1)))
        img_grp[grp[0]] = 0; img_grp[grp[1]] = 1
        img_grp = img_grp.reshape(X0.shape)
        X0_block = np.squeeze(view_as_blocks(X0, (1,2,2,1)))
        blocks = view_as_blocks(img_grp, (1,2,2,1)).reshape(-1, 2, 2)
        grp_blocks = np.where(np.sum(blocks.reshape(-1,4), axis = -1)==0, 0,1)
        grp_blocks = grp_blocks.reshape(X0_block.shape[0],X0_block.shape[1],X0_block.shape[2])
        grp_blocks = duplicate(np.squeeze(grp_blocks))
        N, h, w = grp_blocks.shape
        g0 = np.where(grp_blocks.reshape(-1) == 0); g1 = np.where(grp_blocks.reshape(-1) ==1)
        g = [g0,g1]
        """
        g_name = 'grp_0'
        ud_idx = None; rot_idx = None;
        Y_selg = Y.copy()
        print(f"Testing Group {g_name}")
        feat_inp = self.Get_PixelHopFeat(level, X, 'inp', 0,self.args['win'][level - 1] ,g, g_name, ud_idx, rot_idx)
        feat_inp = self.Add_SpatialFeat(feat_inp, X0, level,self.args['win'][level - 1],'inp' ,ud_idx, rot_idx)
        N,h,w,c = feat_inp.shape
        feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g0]
                
        feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred,-1), 'prev', 0,self.args['win'][level - 1] ,g, g_name, ud_idx, rot_idx)
        feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred,-1), level,self.args['win'][level - 1],'prev' ,ud_idx, rot_idx)
        feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g0]
        feat = np.concatenate([feat_inp, feat_prev], axis = -1)
        feat_lnt = None
        gc.collect()
        feat_prev = None
        gc.collect()
        feat_inp = None
        gc.collect()
        feat_rft = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
        
                
        pred[g0] = self.Get_XGBoost_Pred(feat_rft.reshape(-1, feat_rft.shape[-1]), prev_pred.reshape(-1)[g0], Y.reshape(-1)[g0], level, mode, g_name)
        """
        feat_rft = None
        gc.collect()
        g_name = 'grp_1'
        print(f"Testing Group {g_name}")
        #X0_eq = exposure.equalize_adapthist(1-X0)
        X0_eq = block_reduce(X0, (1, 2**(level-1), 2**(level-1),1), np.mean)
        N, h, w,_ = X0_eq.shape
        X0_eq = 1-X0_eq
        resd = np.squeeze(X0_eq) - prev_pred
        #Y_haar = np.squeeze(view_as_blocks(Y_selg, (1,2,2)))
        #Y_haar = Y_haar.reshape(-1, 2 ,2)
        #prev_pred = np.squeeze(view_as_blocks(prev_pred_sel, (1,2,2)))
        #prev_pred = prev_pred.reshape(-1, 2 ,2)
        res_haar = Y - prev_pred
        grp2 = np.squeeze(view_as_blocks(grp_blocks, (1,2,2)))
        grp2 = grp2.reshape(-1, 2 ,2)
        resd = np.squeeze(utils.Shrink_patch(np.expand_dims(resd,-1), 1, 6, 2, 2))
        #resd = np.squeeze(utils.Shrink_patch(np.expand_dims(resd,-1), 1, 6, 2, 2))
        res_haar = np.squeeze(utils.Shrink_patch(np.expand_dims(res_haar,-1), 1, 6, 2, 2))
        wt_0 = []; wt_g = []
        y_0 = []; y_g = []
        g1_ind = []; g0_ind = []
        for n in range(len(grp2)):
            if(np.sum(grp2[n])==4):
                g1_ind.append(n)
            
                """
                coeff = pywt.dwt2(resd[n], 'haar')
                wt_0.append(np.squeeze(coeff[0]))
                wt_g.append(np.squeeze(coeff[1]))
                coeff_y = pywt.dwt2(res_haar[n], 'haar')
                y_0.append(np.squeeze(coeff_y[0]))
                y_g.append(np.squeeze(coeff_y[1]))
                """

        coeff = pywt.dwt2(resd, 'haar')
        wt_0 = coeff[0]
        wt_g = coeff[1]
        coeff_y = pywt.dwt2(res_haar, 'haar')
        y_0 = coeff_y[0]
        y_g = coeff_y[1]
        
        wt_0 = np.asarray(wt_0);wt_g = np.asarray(wt_g)
        y_0 = np.asarray(y_0);y_g = np.asarray(y_g)

        feat_inp = self.Get_PixelHopFeat(level, np.expand_dims(resd,-1), 'patches', 0,self.args['win'][level - 1],grp,'resd_1' , ud_idx, rot_idx)
        feat_inp = self.Add_SpatialFeat(feat_inp, resd, level,self.args['win'][level - 1],'patches', ud_idx, rot_idx )
        feat_inp = feat_inp
        print(f"Shape of input features: {feat_inp.shape}")

        for chan in range(4):
            print(f'Feature Extraction Channel {chan}')
            if(chan ==0):
                X_chan = wt_0
            else:
                X_chan = wt_g[chan-1]
            g_name = f'chan_{chan}1'
            feat_chan = self.Get_PixelHopFeat(level, np.expand_dims(X_chan,-1), 'patches', 0,self.args['win'][level - 1],grp,g_name , ud_idx, rot_idx)
            feat_chan = self.Add_SpatialFeat(feat_chan, X_chan, level,self.args['win'][level - 1],'patches', ud_idx, rot_idx )
            print(f"Shape of input features: {feat_chan.shape}")
            #N,h,w,c = feat_chan.shape
            #feat_chan = feat_chan.reshape(-1, h*w)
            feat_inp = np.concatenate([feat_inp, feat_chan], axis = -1)
            feat_chan = None
            gc.collect()
        
        g_name = 'grp_1'
        if(self.args['lnt'][level-1]):
            feat_lnt = self.Get_LNT(feat_inp, level, g_name)
            feat_rft = np.concatenate([feat_inp, feat_lnt], axis = -1)
            feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
        else:
            feat_rft = feat_inp
        feat_rft = (feat_rft - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
        feat_inp = None 
        gc.collect()
        coeff = []
        for chan in range(4):
            g_name = f'chan_{chan+1}'
            print(f'XGBoost Testing {g_name}')
            if(chan == 0):
                Y_haar = y_0[:,1,1]
                Y_haar = np.asarray(Y_haar)
                feat_tr = feat_rft[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                coeff0 = self.Get_XGB_Haar(feat_tr.reshape(-1, feat_tr.shape[-1]), Y_haar, level, g_name, mode)
            else: 
                Y_haar = y_g[chan-1,:,1,1]
                Y_haar = np.asarray(Y_haar)
                feat_tr = feat_rft[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                coeff.append(self.Get_XGB_Haar(feat_tr.reshape(-1, feat_tr.shape[-1]), Y_haar, level, g_name, mode))
        
        feat_rft = None 
        gc.collect()
        #coeff = np.asarray(coeff)
        coeffs = (np.asarray(coeff0)[:, np.newaxis, np.newaxis], (np.asarray(coeff[0])[:, np.newaxis, np.newaxis], np.asarray(coeff[1])[:, np.newaxis, np.newaxis], np.asarray(coeff[2])[:, np.newaxis, np.newaxis]))
        haar_pred = pywt.idwt2(coeffs, 'haar')
        haar_pred = np.asarray(haar_pred)
        
        pred = pred.reshape(N, h,w)
        pred = np.squeeze(view_as_blocks(pred, (1,2,2)))
        pred = pred.reshape(-1,2,2)
        #pred[g1_ind] = pred[g1_ind] + haar_pred[g1_ind] 
        pred= pred + haar_pred
        

        pred = pred.reshape(N, int(h/2), int(w/2), 2,2)
        pred = pred.transpose(0, 1, 3, 2, 4).reshape(N, h, w)

        return truncation(pred)
    
    
    def train_level_cs(self,level, X0, X, Y, prev_pred, pred_b = None, sel = None, idx_hard = None):     
        
        if(level == 0):
            sel = np.arange(0,len(X))
            idx = random.sample(sel.tolist(), int(len(sel)/2))
            self.Train_KMeans_InitPred(X[idx], 2)
            pred = self.Get_KMeans_InitPred(X, 2)
            
        else:
            win = self.args['win'][level-1]
            ld = 2**(level-1)           
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            Y_grad = []
            for im in range(len(Y)):
                Y_grad.append(utils.get_edges(Y[im]))
            Y_grad = np.asarray(Y_grad)
            Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            """
            if(level!=1):
                for l in range(level-1):
                    X = self.lp(X)
                    X0 =self.lp(X0)
                    Y = np.squeeze(self.lp(Y))
            """
            idx = np.arange(0, len(X), 1)
            if(self.args['downsample_patches'][level-1]!=1 and sel == None):
                #patch_err = []
                #a = duplicate(prev_pred)
                #for patch in range(len(a)):
                #        patch_err.append(calculate_mse(a[patch].reshape(-1), Y[patch].reshape(-1)))
                #idx = np.argsort(patch_err)[::-1] 
                #sel = idx[:int(self.args['downsample_patches'][level-1]*len(X))]
                sel = random.sample(idx.tolist(), int(self.args['downsample_patches'][level-1]*len(X)))
                X_sel = X[sel]; Y_sel = Y[sel]; prev_pred_sel = prev_pred[sel]; X0_sel = X0[sel]; #pred_b_sel = pred_b[sel]
                res = Y_sel - duplicate(prev_pred_sel)
            else:
                X_sel = X; Y_sel = Y; prev_pred_sel = prev_pred; X0_sel = X0; #pred_b_sel = pred_b
            X_samples = utils.Shrink_patch(X0_sel, ld, win, 1, int(win//2)) 
            pp_samples = utils.Shrink_patch(np.expand_dims(prev_pred_sel,-1), 1, win , 1, int(win//2)) 
            #_, ud_idxs, rot_idxs = self.patch_alignment(X_samples)
            
            self.pixelhop_all[f'inplevel_{level}'] = dict()
            self.pixelhop_prev[f'prevlevel_{level}'] = dict()
            self.rft_all[f'level_{level}'] = dict()
            self.lnt_all[f'level_{level}'] = dict()
            self.xgb_all[f'level_{level}'] = dict()
            self.feat_mean[f'level_{level}'] = dict(); self.feat_std[f'level_{level}'] = dict()
            
            
            if(level<4 and idx_hard is None):
                #prev_pred_sel = duplicate(prev_pred_sel)
                #prev_pred_sel = upscale_pred(prev_pred_sel, (2*prev_pred_sel.shape[-1], 2*prev_pred_sel.shape[-1]), order= 'lanczos')
                prev_pred_sel = upsample(prev_pred_sel,2*prev_pred_sel.shape[-1] )
                prev_pred_sel = truncation(prev_pred_sel)
                #pred_b_sel = upscale_pred(pred_b_sel, (2*pred_b_sel.shape[-1], 2*pred_b_sel.shape[-1]), order= 'lanczos')
            grp,self.thresh[f'level_{level}'] = self.energy_split_train(X0_sel,np.expand_dims(prev_pred_sel,-1), level) 
            #pred = np.zeros(len(Y_sel.reshape(-1)))
            pred = np.zeros(len(Y_sel.reshape(-1))*2)
            """
            grp= []
            
            if(level>4): 
                a = np.where(Y_sel.reshape(-1)==0)[0]
                b = np.where(Y_sel.reshape(-1)!=0)[0]
                grp.append(np.concatenate([b, random.sample(a.tolist(),int(0.75*len(a) ))]))
            else: 
                grp.append(np.arange(0, len(prev_pred_sel.reshape(-1)), 1))
                self.thresh[f'level_{level}'] = 1
            """
            for g_ind in range(len(grp)):
                
                g = grp[g_ind]
                g_name = f'grp_{g_ind}'

                if(g_ind ==3):
                    a = np.where(prev_pred_sel.reshape(-1)[g] >0.05)[0]
                    b = np.where(prev_pred_sel.reshape(-1)[g] <0.95)[0]
                    ab = np.intersect1d(a,b)
                    g = g[ab]
                """
                if(level==3 or level ==4):
                    ud_idx = ud_idxs; rot_idx = rot_idxs
                    g = np.insert(g, len(g), g+len(X_samples))
                    Y_selg = np.concatenate([Y_sel, Y_sel])
                else:
                    ud_idx = None; rot_idx = None
                    Y_selg = Y_sel.copy()
                """

                print(f"Training Group {g_name}")
                ud_idx = None; rot_idx = None
                g = grp[g_ind]
                Y_selg = Y_sel.copy()
                #res = (1-X_sel) - prev_pred_sel
                if(idx_hard is not None):
                    g = np.intersect1d(g, idx_hard)
                    g_name = f'grp2_{g_ind}'
                self.Train_PixelHop(level, X_sel, X0_sel,np.expand_dims(prev_pred_sel,-1), 'inp', 3, self.args['win'][level - 1],g, g_name, self.thresh[f'level_{level}'] , ud_idx, rot_idx)
                feat_inp = self.Get_PixelHopFeat(level, X_sel, 'inp', 0,self.args['win'][level - 1],g_name , ud_idx, rot_idx)
                feat_inp = self.Add_SpatialFeat(feat_inp, X0_sel, level,self.args['win'][level - 1],'inp', ud_idx, rot_idx )
                print(f"Shape of input features: {feat_inp.shape}")
                N,h,w,c = feat_inp.shape

                feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
                
                self.Train_PixelHop(level, np.expand_dims(prev_pred_sel,-1),X0_sel,np.expand_dims(prev_pred_sel,-1), 'prev', 3, self.args['win'][level - 1],g, g_name, self.thresh[f'level_{level}'] , ud_idx, rot_idx)
                feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred_sel,-1), 'prev', 0,self.args['win'][level - 1], g_name, ud_idx, rot_idx )
                feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred_sel,-1), level,self.args['win'][level - 1],'prev' , ud_idx, rot_idx)
                print(f"Shape of prev pred features: {feat_prev.shape}")
                feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g]
                
                """
                self.Train_PixelHop(level, np.expand_dims(pred_b_sel,-1),X0_sel,np.expand_dims(prev_pred_sel,-1), 'prevb', 3, self.args['win'][level - 1],g, g_name, self.thresh[f'level_{level}'], ud_idx, rot_idx)
                feat_predb = self.Get_PixelHopFeat(level, np.expand_dims(pred_b_sel,-1), 'prevb', 0,self.args['win'][level - 1], g_name, ud_idx, rot_idx )
                feat_predb = self.Add_SpatialFeat(feat_predb, np.expand_dims(pred_b_sel,-1), level,self.args['win'][level - 1],'prev' , ud_idx, rot_idx)
                print(f"Shape of prev pred boundary features: {feat_predb.shape}")
                feat_predb = feat_predb.reshape(-1, feat_predb.shape[-1])[g]
                """
                feat = np.concatenate([feat_inp, feat_prev], axis = -1)
                if(ud_idx is not None):
                    prev_pred_selg = np.concatenate([prev_pred_sel, prev_pred_sel])  
                else: 
                    prev_pred_selg = prev_pred_sel.copy()
                Y_selg = Y_selg.reshape(-1)[g]; prev_pred_selg = prev_pred_selg.reshape(-1)[g]
                res = Y_selg - prev_pred_selg
                feat_idx = np.arange(0, feat.shape[-1], 1)

                save_path = args['img_root']
                X_train, X_val, y_train, y_val = train_test_split(feat, res, test_size=0.2,
                                                          random_state=42)
                """
                #RFT 
                rft = FeatureTest('rmse')
                rft.fit(X_train, y_train, n_bins=16, outliers=True)
                rft.plot(path=os.path.join(save_path, f"train_rft_{level}_{g_name}.png"))

                #logger.info(f'Val RFT, shape: {X_val.shape}')
                rft_val = FeatureTest('rmse')
                rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
                rft_val.plot(path=os.path.join(save_path, f"val_rft_{level}_{g_name}.png"))

                plot_train_val_rank(rft, rft_val, path=os.path.join(save_path, f"joint_{level}_{g_name}.png"))
                rft.n_selected = int(self.args['FS'][level-1][g_ind]*len(feat_idx))
                self.rft_all[f'level_{level}'][f'before_lnt_{g_name}'] = rft
                """
                self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                #feat = rft.transform(feat, n_selected=rft.n_selected)
                #X_val = rft.transform(X_val, n_selected=self.args['num_rft_features'][cur_level - 1])
                #LNT
                
                if(self.args['lnt'][level-1]):
                    self.Train_LNT(feat, res, level, g_name)
                    feat_lnt = self.Get_LNT(feat, level, g_name)
                    #feat_tr = feat_lnt.copy()
                    feat_tr = np.concatenate([feat, feat_lnt], axis = -1)
                    plot(feat, res, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
                    feat_idx = np.arange(0, feat_tr.shape[-1], 1)
                    """
                    X_train, X_val, y_train, y_val = train_test_split(feat_tr, res, test_size=0.2,
                                                          random_state=42)
                    #RFT 
                    rft = FeatureTest('rmse')
                    rft.fit(X_train, y_train, n_bins=16, outliers=True)
                    rft.plot(path=os.path.join(save_path, f"train_rft_lnt_{level}_{g_name}.png"))

                    #logger.info(f'Val RFT, shape: {X_val.shape}')
                    rft_val = FeatureTest('rmse')
                    rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
                    rft_val.plot(path=os.path.join(save_path, f"val_rft_lnt_{level}_{g_name}.png"))

                    plot_train_val_rank(rft, rft_val, path=os.path.join(save_path, f"jointlnt_{level}_{g_name}.png"))
                    rft.n_selected = int(self.args['FS'][level-1][g_ind]*len(feat_idx))
                    self.rft_all[f'level_{level}'][f'after_lnt_{g_name}'] = rft
                    """
                    #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                    #feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                    #feat_tr = rft.transform(feat_tr, n_selected=rft.n_selected)
                    
                    self.rft_feat_selection(level, feat_tr, res,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}LNT_{g_name}')
                    feat_tr = feat_tr[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
                else:
                    feat_tr = feat
                #XGBoost 
                self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
                self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
                feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
                print(f"Shape of features after RFT {feat_tr.shape}")
                feat_lnt = None
                gc.collect()
                feat_prev = None
                gc.collect()
                feat_inp = None
                gc.collect()
                feat = None 
                gc.collect()
                if(g_ind == 0): 
                    g_sel = np.where(np.abs(Y_selg)<0.01)[0]
                    ix = np.arange(0, len(Y_selg),1)
                    a = np.setdiff1d(ix, g_sel)
                    sel_g = np.concatenate([a, random.sample(g_sel.tolist(), int(0.75*len(g_sel)))])
                else: 
                    sel_g = np.arange(0, len(Y_selg),1).tolist()
                self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1])[sel_g], prev_pred_selg[sel_g], Y_selg[sel_g], level, g_name)
                pred[g], _ = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg, Y_selg, level, 'tra', g_name)
            
            pred_all = pred.reshape(len(X_sel)*2, Y.shape[1], Y.shape[2])
            
            pred = pred_all[:len(X_sel)]
            #pp_samples = utils.Shrink_patch(np.expand_dims(pred,-1), 1, win , 1, int(win//2)) 
            #_, ud_idxs, rot_idxs = self.patch_alignment(pp_samples)
            #ud_idx = ud_idxs; rot_idx = rot_idxs
            next_pred = pred.reshape(-1)
            a = np.where(next_pred>0.1)[0]
            b = np.where(next_pred<0.9)[0]
            g = np.intersect1d(a,b)
            #g = np.insert(g, len(g), g+len(pp_samples))
            """
            grp0,self.thresh[f'level_{level}2'] = self.energy_split_train(X0_sel,np.expand_dims(pred,-1), level) 
            #g = grp0[1]
            g = np.arange(0, len(prev_pred_sel.reshape(-1)))
            g_name = f'grp_{21}'
            ud_idx = None; rot_idx = None
            #g = np.insert(g, len(g), g+len(pp_samples))
            #Y_selg = np.concatenate([Y_sel, Y_sel])

            print(f"Training Group {g_name}")
            #res = (1-X_sel) - prev_pred_sel
            self.Train_PixelHop(level, X_sel, X0_sel,np.expand_dims(pred,-1), 'inp', 3, self.args['win'][level - 1],g, g_name, self.thresh[f'level_{level}2'],ud_idx, rot_idx)
            feat_inp = self.Get_PixelHopFeat(level, X_sel, 'inp', 0,self.args['win'][level - 1],g,g_name , ud_idx, rot_idx)
            feat_inp = self.Add_SpatialFeat(feat_inp, X0_sel, level,self.args['win'][level - 1],'inp', ud_idx, rot_idx )
            print(f"Shape of input features: {feat_inp.shape}")
            N,h,w,c = feat_inp.shape
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
                
            self.Train_PixelHop(level, np.expand_dims(pred,-1),X0_sel,np.expand_dims(pred,-1), 'prev', 3, self.args['win'][level - 1],g, g_name, self.thresh[f'level_{level}2'], ud_idx, rot_idx)
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred,-1), 'prev', 0,self.args['win'][level - 1],g, g_name, ud_idx, rot_idx )
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred,-1), level,self.args['win'][level - 1],'prev' , ud_idx, rot_idx)
            print(f"Shape of prev pred features: {feat_prev.shape}")
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g]
            feat = np.concatenate([feat_inp, feat_prev], axis = -1)
            #feat = feat_prev
            #prev_pred_selg = np.concatenate([pred, pred])  
            prev_pred_selg = pred.copy()
            Y_selg = Y_sel.reshape(-1); prev_pred_selg = prev_pred_selg.reshape(-1)
            res = Y_selg - prev_pred_selg
            feat_idx = np.arange(0, feat.shape[-1], 1)

            #RFT 
                
            self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], 0.75, f'level_{level}_{g_name}')
            feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                
            #LNT
            
            if(self.args['lnt'][level-1]):
                self.Train_LNT(feat, res, level, g_name)
                feat_lnt = self.Get_LNT(feat, level, g_name)
                feat_tr = np.concatenate([feat, feat_lnt], axis = -1)
                plot(feat, res, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
                feat_idx = np.arange(0, feat_tr.shape[-1], 1)
                self.rft_feat_selection(level, feat_tr, res,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}LNT_{g_name}')
                feat_tr = feat_tr[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            else:
                feat_tr = feat
            

            feat_tr = feat
            
            #XGBoost 
            self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
            self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
            feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
            print(f"Shape of features after RFT {feat_tr.shape}")
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            feat = None 
            gc.collect()
            self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg[g], Y_selg[g], level, 'grp_2')
            """               
            """
            g_name = f'grp_{2}'
            if(idx_hard is not None):
                g = np.intersect1d(g, idx_hard)
                g_name = f'grp2_{2}'
            #grp.append(g)
            self.Train_PixelHop(level, X_sel, X0_sel,np.expand_dims(pred,-1), 'inp', 3, self.args['win'][level - 1],grp, g_name, None,ud_idx, rot_idx)
            feat_inp = self.Get_PixelHopFeat(level, X_sel, 'inp', 0,self.args['win'][level - 1],g_name , ud_idx, rot_idx)
            feat_inp = self.Add_SpatialFeat(feat_inp, X0_sel, level,self.args['win'][level - 1],'inp', ud_idx, rot_idx )
            print(f"Shape of input features: {feat_inp.shape}")
            N,h,w,c = feat_inp.shape
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
                
            self.Train_PixelHop(level, np.expand_dims(pred,-1),X0_sel,np.expand_dims(pred,-1), 'prev', 3, self.args['win'][level - 1],grp, g_name, None, ud_idx, rot_idx)
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred,-1), 'prev', 0,self.args['win'][level - 1], g_name, ud_idx, rot_idx )
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred,-1), level,self.args['win'][level - 1],'prev' , ud_idx, rot_idx)
            print(f"Shape of prev pred features: {feat_prev.shape}")
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g]
            feat = np.concatenate([feat_inp, feat_prev], axis = -1)
            #feat = feat_inp
            if(level ==3 or level ==4):
                prev_pred_selg = np.concatenate([pred, pred])  
                Y_selg = np.concatenate([Y_sel, Y_sel]).reshape(-1)
            else: 
                prev_pred_selg = pred.copy()
                Y_selg = Y_sel.copy().reshape(-1)
            prev_pred_selg = prev_pred_selg.reshape(-1)
            res = Y_selg[g] - prev_pred_selg[g]
            feat_idx = np.arange(0, feat.shape[-1], 1)

            #RFT 
                
            self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], 0.6, f'level_{level}_{g_name}')
            feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                
            #LNT
            
            if(self.args['lnt'][level-1]):
                self.Train_LNT(feat, res, level, g_name)
                feat_lnt = self.Get_LNT(feat, level, g_name)
                feat_tr = np.concatenate([feat, feat_lnt], axis = -1)
                plot(feat, res, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
                feat_idx = np.arange(0, feat_tr.shape[-1], 1)
                self.rft_feat_selection(level, feat_tr, res,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}LNT_{g_name}')
                feat_tr = feat_tr[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            else:
                feat_tr = feat
            

            #feat_tr = feat
            
            #XGBoost 
            self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
            self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
            feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
            print(f"Shape of features after RFT {feat_tr.shape}")
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            feat = None 
            gc.collect()
            self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg[g], Y_selg[g], level, 'grp_2')
            #pred = pred.reshape(N, h,w)
            
            feat_tr = None
            gc.collect()
            """
        return sel

    #@njit
    def test_level_cs(self, level, X0, X, Y, mode, prev_pred, pred_b= None, idx_hard = None):
        
        if(level == 0):
            pred = self.Get_KMeans_InitPred(X, 2)
        else:
            
            ud_idx = None
            rot_idx = None
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            """
            Y_grad = []
            for im in range(len(Y)):
                Y_grad.append(utils.get_edges(Y[im]))
            Y_grad = np.asarray(Y_grad)
            """
            Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            """
            if(level!=1):
                for l in range(level-1):
                    X = self.lp(X)
                    X0 =self.lp(X0)
                    Y = np.squeeze(self.lp(Y))
            """
            if(level <4 and idx_hard is None):
                #prev_pred = duplicate(prev_pred)
                prev_pred = upscale_pred(prev_pred, (2*prev_pred.shape[-1], 2*prev_pred.shape[-1]), order= 'lanczos')
                #prev_pred = upsample(prev_pred,2*prev_pred.shape[-1] )
                prev_pred = truncation(prev_pred)
                """
                if(pred_b is not None):
                    pred_b = upscale_pred(pred_b, (2*pred_b.shape[-1], 2*pred_b.shape[-1]), order= 'lanczos')
                """
            
            if(pred_b is None):
                grp = self.energy_split_test(X0,np.expand_dims(prev_pred,-1), level,self.thresh[f'level_{level}'] ,'orig') 
            else:
                grp= []
                grp.append(np.arange(0, len(prev_pred.reshape(-1)), 1))
            pred = np.zeros(len(prev_pred.reshape(-1)))
                
            #grp= []
            #grp.append(np.arange(0, len(prev_pred.reshape(-1)), 1))
            
            for g_ind in range(len(grp)):
                g = grp[g_ind]
                g_name = f'grp_{g_ind}'
                if(idx_hard is not None):
                    g = np.intersect1d(g, idx_hard)
                    g_name = f'grp2_{g_ind}'
                """
                if(g_ind ==3):
                    a = np.where(prev_pred.reshape(-1)[g] >0.05)[0]
                    b = np.where(prev_pred.reshape(-1)[g] <0.95)[0]
                    ab = np.intersect1d(a,b)
                    g = g[ab]
                """
                print(f"Testing Group {g_name}")
                #res = (1-X) - prev_pred
                feat_inp = self.Get_PixelHopFeat(level, X, 'inp', 0,self.args['win'][level - 1] , g_name, ud_idx, rot_idx)
                feat_inp = self.Add_SpatialFeat(feat_inp, X0, level,self.args['win'][level - 1],'inp' ,ud_idx, rot_idx)
                N,h,w,c = feat_inp.shape
                feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
                
                feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred,-1), 'prev', 0,self.args['win'][level - 1] , g_name, ud_idx, rot_idx)
                feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred,-1), level,self.args['win'][level - 1],'prev' ,ud_idx, rot_idx)
                feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g]
                
                if(pred_b is not None):
                    feat_predb = self.Get_PixelHopFeat(level, np.expand_dims(pred_b,-1), 'prevb', 0,self.args['win'][level - 1], g_name, ud_idx, rot_idx )
                    feat_predb = self.Add_SpatialFeat(feat_predb, np.expand_dims(pred_b,-1), level,self.args['win'][level - 1],'prev' , ud_idx, rot_idx)
                
                    feat_predb = feat_predb.reshape(-1, feat_predb.shape[-1])[g]
                
                    feat = np.concatenate([feat_inp, feat_prev, feat_predb], axis = -1)
                    
                else: 
                    
                    feat = np.concatenate([feat_inp, feat_prev], axis = -1)
                    
                feat_lnt = None
                #gc.collect()
                feat_prev = None
                #gc.collect()
                feat_inp = None
                #gc.collect()
                feat_rft = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                #rft = self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']
                #feat_rft = rft.transform(feat, n_selected = rft.n_selected)
                if(self.args['lnt'][level-1]):
                    feat_lnt = self.Get_LNT(feat_rft, level, g_name)
                    #feat_rft = feat_lnt.copy()
                    feat_rft = np.concatenate([feat_rft, feat_lnt], axis = -1)
                    feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
                    #rft = self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']
                    #feat_rft = rft.transform(feat_rft, n_selected = rft.n_selected)
                feat_rft = (feat_rft - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
                feat_lnt = None
                #gc.collect()
                pred[g], _ = self.Get_XGBoost_Pred(feat_rft.reshape(-1, feat_rft.shape[-1]), prev_pred.reshape(-1)[g], Y.reshape(-1)[g], level, mode, g_name)
                feat_rft = None
                #gc.collect()
            pred = pred.reshape(N, h,w)

            #fin_pred = pred.copy().reshape(-1)
            #a = np.where(fin_pred>0.1)[0]
            #b = np.where(fin_pred<0.9)[0]
            #g = np.intersect1d(a,b)
            """
            fin_pred = pred.copy().reshape(-1)
            grp0 = self.energy_split_test(X0,np.expand_dims(pred,-1), level, self.thresh[f'level_{level}2'],'orig') 
            g = grp0[1]
            g_name = f'grp_{21}'
            print(f"Testing Group {g_name}")
            #res = (1-X) - prev_pred
            feat_inp = self.Get_PixelHopFeat(level, X, 'inp', 0,self.args['win'][level - 1] ,g, g_name, ud_idx, rot_idx)
            feat_inp = self.Add_SpatialFeat(feat_inp, X0, level,self.args['win'][level - 1],'inp' ,ud_idx, rot_idx)
            N,h,w,c = feat_inp.shape
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
                
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred,-1), 'prev', 0,self.args['win'][level - 1] ,g, g_name, ud_idx, rot_idx)
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred,-1), level,self.args['win'][level - 1],'prev' ,ud_idx, rot_idx)
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g]
            feat = np.concatenate([feat_inp, feat_prev], axis = -1)
            #feat = feat_prev
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            feat_rft = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            
            if(self.args['lnt'][level-1]):
                feat_lnt = self.Get_LNT(feat_rft, level, g_name)
                feat_rft = np.concatenate([feat_rft, feat_lnt], axis = -1)
                feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
        
            feat_rft = (feat_rft - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
                
            fin_pred[g] = self.Get_XGBoost_Pred(feat_rft.reshape(-1, feat_rft.shape[-1]), pred.reshape(-1)[g], Y.reshape(-1)[g], level, mode, 'grp_2')
            fin_pred = fin_pred.reshape(N, h,w)
            """
            """
            g_name = f'grp_{2}'
            if(idx_hard is not None):
                g = np.intersect1d(g, idx_hard)
                g_name = f'grp2_{2}'
            print(f"Testing Group {g_name}")
            #res = (1-X) - prev_pred
            feat_inp = self.Get_PixelHopFeat(level, X, 'inp', 0,self.args['win'][level - 1], g_name, ud_idx, rot_idx)
            feat_inp = self.Add_SpatialFeat(feat_inp, X0, level,self.args['win'][level - 1],'inp' ,ud_idx, rot_idx)
            N,h,w,c = feat_inp.shape
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
                
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred,-1), 'prev', 0,self.args['win'][level - 1] , g_name, ud_idx, rot_idx)
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred,-1), level,self.args['win'][level - 1],'prev' ,ud_idx, rot_idx)
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])[g]
            feat = np.concatenate([feat_inp, feat_prev], axis = -1)
            #feat = feat_inp
            feat_lnt = None
            #gc.collect()
            feat_prev = None
            #gc.collect()
            feat_inp = None
            #gc.collect()
            feat_rft = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            
            if(self.args['lnt'][level-1]):
                feat_lnt = self.Get_LNT(feat_rft, level, g_name)
                feat_rft = np.concatenate([feat_rft, feat_lnt], axis = -1)
                feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
        
            feat_rft = (feat_rft - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
                
            fin_pred[g], _ = self.Get_XGBoost_Pred(feat_rft.reshape(-1, feat_rft.shape[-1]), pred.reshape(-1)[g], Y.reshape(-1)[g], level, mode, 'grp_2')
            fin_pred = fin_pred.reshape(N, h,w)
            
            feat_rft = None
            #gc.collect()
            """
            
        return pred
    
    def hard_sample_select(self, level, pred, Y,res_pred):
        
        X_train, X_val, y_train, y_val, res_train,res_val = train_test_split(pred, Y, res_pred, test_size=0.2,
                                                          random_state=42)

        eps = np.arange(0.005, 0.25, 0.005)
        mse_eps = []
        for e in eps: 
            a = np.where(res_val>-e)[0]
            b = np.where(res_val<e)[0]
            samp = np.intersect1d(a,b)
            mse_eps.append(calculate_mse(X_val[samp], y_val[samp]))

        plt.cla();plt.clf();
        plt.plot(eps, mse_eps)
        plt.title(f"Epsilon vs. MSE plot Level{level}")
        plt.xlabel("Epsilon")
        plt.ylabel("MSE")
        plt.savefig(self.args['img_root'] + f"Epsilon vs MSE plot Level {level}")

        sel = np.max(np.where(np.asarray(mse_eps)<=0.017)[0])
        sel_eps = eps[sel]
        self.eps[f'{level}'] = sel_eps
        a = np.where(res_pred<-sel_eps)[0]
        b = np.where(res_pred>sel_eps)[0]
        samp = np.concatenate([a,b])
        print("Round 2 training samples: ", len(samp))
        return samp
    
    def select_roi(self, X0, Y, level, mode):
        win = 3
        pad = int(win//2)
        neigh = utils.Shrink_patch(Y, 1, win, 1, pad )
        neigh_ = neigh.reshape(-1, win*win)
        dc = (np.sum(neigh_, axis = -1) - neigh[:,0,1,1])/8.0
        cen = neigh[:,0,1,1] - dc
        #cen = cen.reshape(Y.shape)

        sel = thresh(cen, -0.025, 0.025)

        plt.cla();plt.clf()
        plt.hist(cen, bins = 64)
        plt.title(f"ROI Selection Histogram Level {level} {mode}")
        plt.savefig(f"ROI Selection Histogram Level {level} {mode}")
        plt.cla();plt.clf();
        return sel

    def select_roi_im(self, X0, Y, level, mode):
        win = 3
        pad = int(win//2)
        neigh = utils.Shrink_patch(Y, 1, win, 1, pad )
        neigh_ = neigh.reshape(-1, win*win)
        dc = (np.sum(neigh_, axis = -1) - neigh[:,0,1,1])/8.0
        cen = neigh[:,0,1,1] - dc
        #cen = cen.reshape(Y.shape)

        sel = thresh(cen, -0.03, 0.03)

        a = np.zeros(len(Y.reshape(-1)))
        a[sel] = 1
        a = a.reshape(Y.shape)
                
        return a
    
    def train_level(self,level, X0, X, Y, prev_pred = None, sel = None, idx_hard = None):     
        
        if(level == 0):
            sel = np.arange(0,len(X))
            idx = random.sample(sel.tolist(), int(len(sel)/2))
            self.Train_KMeans_InitPred(X[idx], 2)
            pred = self.Get_KMeans_InitPred(X, 2)
            
        else:

            win = self.args['win'][level-1]
            ld = 2**(level-1) 
            if(idx_hard is None): 
                self.pixelhop_all[f'inplevel_{level}'] = dict()
                self.pixelhop_prev[f'prevlevel_{level}'] = dict()
                self.rft_all[f'level_{level}'] = dict()
                
                self.lnt_all[f'level_{level}'] = dict()
                self.xgb_all[f'level_{level}'] = dict()
                self.feat_mean[f'level_{level}'] = dict(); self.feat_std[f'level_{level}'] = dict()
            Y_orig = Y.copy()
            Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            N,h,w = Y.shape
            idx = np.arange(0, len(X), 1)
            """
            if(self.args['downsample_patches'][level-1]!=1 and sel == None):
                sel = random.sample(idx.tolist(), int(self.args['downsample_patches'][level-1]*len(X)))
                X_sel = X[sel]; Y_sel = Y[sel]; prev_pred_sel = prev_pred[sel]; X0_sel = X0[sel]; Y_orig_sel = Y_orig[sel];
            else:
                X_sel = X; Y_sel = Y; prev_pred_sel = prev_pred; X0_sel = X0; Y_orig_sel = Y_orig;
            """
            X_sel = X; Y_sel = Y; prev_pred_sel = prev_pred; X0_sel = X0; Y_orig_sel = Y_orig;
            #X_samples = utils.Shrink_patch(X0_sel, ld, win, 1, int(win//2)) 
            #pp_samples = utils.Shrink_patch(np.expand_dims(prev_pred_sel,-1), 1, win , 1, int(win//2)) 
            #_, ud_idxs, rot_idxs = self.patch_alignment(X_samples)
            if(level <4 and idx_hard is None): 
                prev_pred_sel = upscale_pred(prev_pred_sel, (2*prev_pred_sel.shape[-1], 2*prev_pred_sel.shape[-1]), order= 'lanczos')
                prev_pred_sel = truncation(prev_pred_sel)
            #sel = self.select_roi(X0_sel, np.expand_dims(prev_pred_sel,-1), level, 'tra')
            #print(f"Selected number of samples: {len(sel)}")
            #g = np.arange(0, len(X_samples), 1)
            
            #ud_idx = ud_idxs; rot_idx = rot_idxs
            #g = np.insert(g, len(g), g+len(X_samples))
            #Y_selg = np.concatenate([Y_sel, Y_sel])
            Y_selg = Y_sel
            if(idx_hard is not None):
                g = idx_hard
                g_name = '2grp2_1'
            else: 
                g = np.arange(0, len(Y_sel.reshape(-1)), 1)
                g_name = '2grp2_0'

            res_tr = (Y_selg.reshape(-1)- prev_pred_sel.reshape(-1))
            abs_res = np.abs(Y_selg.reshape(-1)- prev_pred_sel.reshape(-1))
            sort_absrestr = np.argsort(abs_res)
            
            sel_tr_all = sort_absrestr[int(len(sort_absrestr)/2):]
            self.thresh[f'level_{l}'] = np.min(abs_res[sel_tr_all])
            sel_pos = np.where(res_tr>self.thresh[f'level_{l}'])[0]
            sel_neg = np.where(res_tr<-self.thresh[f'level_{l}'] )[0]
            sel_neg = random.sample(sel_neg.tolist(),1500000 )
            sel_pos = random.sample(sel_pos.tolist(),1500000 )
            sel_tr = np.concatenate([sel_neg, sel_pos])
            sel_id = [i for i in range(len(sort_absrestr))]
            #sel_tr = random.sample(sort_absrestr.tolist(), int(0.33*len(sort_absrestr)))
            sel_easy = random.sample(sort_absrestr[:int(len(sort_absrestr)/2)].tolist(), int(0.15*len(sort_absrestr)) )
            sel_tr_all = np.concatenate([sel_neg, sel_pos, sel_easy])
            sel_id = np.where(Y.reshape(-1)>0.9)
            #X_sel3 = X_sel[:,(3-l+1): X_sel.shape[2] - (3-l+1), (3-l+1): X_sel.shape[2] - (3-l+1) ]
            self.Train_PixelHop(level, X_sel, X0_sel,np.expand_dims(prev_pred_sel,-1), 'inp', 3, self.args['win'][level - 1],[g], g_name, None)
           
            feat_inp = self.Get_PixelHopFeat(level, X_sel, 'inp', 0,self.args['win'][level - 1],g_name)
            feat_inp = self.Add_SpatialFeat(feat_inp, X_sel, level,self.args['fov'][level - 1],'inp' )
            feat_laws = self.Add_LawsFeat(level, X_sel, 'tr')
            feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])
            print(f"Shape of input features: {feat_inp.shape}")
            
            
            self.Train_PixelHop(level, np.expand_dims(prev_pred_sel,-1),X0_sel,np.expand_dims(prev_pred_sel,-1), 'prev', 3, self.args['win'][level - 1],[g], g_name, None)
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred_sel,-1), 'prev', 0,self.args['win'][level - 1], g_name )
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred_sel,-1), level,self.args['fov'][level - 1],'prev' )
            #feat_laws = self.Add_LawsFeat(level, prev_pred_sel, 'tr')
            #feat_prev = np.concatenate([feat_prev, feat_laws], axis = -1)
            #feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])
            #print(f"Shape of prev pred features: {feat_prev.shape}")
            feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1])], axis = -1)[g]
            #print(f"Shape of all features: {feat.shape}")
            #feat = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
            print(f"Shape of all features: {feat.shape}")
            #prev_pred_selg = np.concatenate([prev_pred_sel, prev_pred_sel])  
            feat = feat[sel_tr_all]; 
            
            Y_all = list(Y_selg.reshape(-1)[sel_tr_all])
            prev_pred_selg =list(prev_pred_sel.reshape(-1)[sel_tr_all])

            
            for aug in range(1,5):
                if(aug%2==0):
                    place_holder = np.zeros(prev_pred_sel.shape).reshape(-1)
                    place_holder[sel_easy] = 1
                    place_holder = place_holder.reshape(prev_pred_sel.shape)
                else: 
                    place_holder = np.zeros(prev_pred_sel.shape).reshape(-1)
                    place_holder[sel_tr] = 1
                    place_holder = place_holder.reshape(prev_pred_sel.shape)
                place_holder_aug = self.transform_images_onetype(np.squeeze(place_holder), aug,Y_all )
                #sel_aug = np.where(place_holder_aug.reshape(-1)==1)[0]
                place_holder_windows = view_as_windows(np.pad(place_holder_aug, ((0,0), (1,1),(1,1))), (1,3,3), (1,1,1))
                if(aug<8):
                    #sel_aug = np.where(np.sum(place_holder_windows.reshape(-1,9), axis = -1)>1)[0]
                    sel_aug = np.where(place_holder_aug.reshape(-1)==1)[0]
                else: 
                    sel_aug = sel_tr.copy()
                
                #X0_aug = self.transform_images_onetype(np.squeeze(X0_sel), aug)
                X_aug = self.transform_images_onetype(np.squeeze(X_sel), aug,Y_all)

                if(aug<8):
                    Y_aug = self.transform_images_onetype(np.squeeze(Y_selg), aug,Y_all)
                    prev_pred_aug = self.transform_images_onetype(np.squeeze(prev_pred_sel), aug,Y_all)
                    #Y_orig_sel_aug = self.transform_images_onetype(np.squeeze(Y_orig_sel), aug)
                else: 
                    Y_aug = np.squeeze(Y_selg.copy())
                    #Y_orig_sel_aug = np.squeeze(Y_orig_sel.copy())
                    prev_pred_aug = prev_pred_sel.copy()
                #prev_pred_aug = self.load_prevlevel(level, np.expand_dims(X_aug,-1), np.expand_dims(X0_aug,-1), Y_orig_sel_aug)
                #if(level <4 and idx_hard is None): 
                #    prev_pred_aug = upscale_pred(prev_pred_aug, (2*prev_pred_aug.shape[-1], 2*prev_pred_aug.shape[-1]), order= 'lanczos')
                #    prev_pred_aug = truncation(prev_pred_aug)
                #prev_aug = self.transform_images_onetype(np.squeeze(prev_pred_aug), aug)
                feat_inp = self.Get_PixelHopFeat(level, np.expand_dims(X_aug,-1), 'inp', 0,self.args['win'][level - 1], g_name)
                feat_inp = self.Add_SpatialFeat(feat_inp,  np.expand_dims(X_aug,-1), level,self.args['fov'][level - 1],'inp')
                feat_laws = self.Add_LawsFeat(level,  np.expand_dims(X_aug,-1), 'te')
                
                feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred_aug,-1), 'prev', 0,self.args['win'][level - 1], g_name )
                feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred_aug,-1), level,self.args['fov'][level - 1],'prev' )
                feat_inp = np.concatenate([feat_inp, feat_laws, feat_prev], axis = -1)
                feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[sel_aug]
                feat = np.concatenate([feat, feat_inp], axis = 0)
                Y_all.extend(Y_aug.reshape(-1)[sel_aug])
                prev_pred_selg.extend(prev_pred_aug.reshape(-1)[sel_aug])
            
            Y_selg = np.asarray(Y_all)
        
            prev_pred_selg = np.asarray(prev_pred_selg)
            #pred = prev_pred_selg.reshape(-1).copy()
            Y_selg = Y_selg.reshape(-1); prev_pred_selg = prev_pred_selg.reshape(-1)
            
            
            res = Y_selg - prev_pred_selg
            
            feat_idx = np.arange(0, feat.shape[-1], 1)
            
            """
            a = np.where(abs(res)<0.1)[0]
            idx = np.arange(0, len(res),1) 
            tr_idx = np.setdiff1d(idx, a).tolist()
            ds = random.sample(a.tolist(), int(0.4*len(a)))
            tr_idx.extend(ds)
            """
            
            plot_residue([res], f"Residue Histogram Level {level}", self.args['img_root'])
            
            #res_new = np.sign(res)*np.log(1+np.abs(res))

            #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}_{g_name}')
            #feat_tr = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            
            X_train, X_val, y_train, y_val = train_test_split(feat, Y_selg.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
            rft = FeatureTest('rmse')
            rft.fit(X_train, y_train, n_bins=16, outliers=True)
            rft.plot(path=os.path.join(self.args['img_root'], f"train_rft_{level}_{g_name}.png"))

                    #logger.info(f'Val RFT, shape: {X_val.shape}')
            rft_val = FeatureTest('rmse')
            rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
            rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rft_{level}_{g_name}.png"))

            plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"joint_{level}_{g_name}.png"))
            rft.n_selected = int(0.8*len(feat_idx))
            self.rft_all[f'level_{level}'][f'before_lnt_{g_name}'] = rft
                    #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                    #feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            feat_tr = rft.transform(feat, n_selected=rft.n_selected)
            
            if(self.args['lnt'][level-1]):
                
                feat_trl = np.copy(feat_tr)
               
                    
                self.Train_LNT(feat_tr, Y_selg, level, g_name)
                feat_lnt = self.Get_LNT(feat_tr, level, g_name)
                feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
                plot(feat_tr, Y_selg, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
                feat_idx = np.arange(0, feat_trl.shape[-1], 1)
                #self.rft_feat_selection(level, feat_trl, res,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level - 1][int(g_name[-1])], f'level_{level}LNT_{g_name}')
                #feat_tr = feat_trl[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
                
                X_train, X_val, y_train, y_val = train_test_split(feat_trl, Y_selg.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
                rft = FeatureTest('rmse')
                rft.fit(X_train, y_train, n_bins=16, outliers=True)
                rft.plot(path=os.path.join(self.args['img_root'], f"train_rftlnt_{level}_{g_name}.png"))

                        #logger.info(f'Val RFT, shape: {X_val.shape}')
                rft_val = FeatureTest('rmse')
                rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
                rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rftlnt_{level}_{g_name}.png"))

                plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"jointlnt_{level}_{g_name}.png"))
                rft.n_selected = int(self.args['FS'][level - 1][int(g_name[-1])]*len(feat_idx))
                self.rft_all[f'level_{level}'][f'after_lnt_{g_name}'] = rft
                feat_tr = rft.transform(feat_trl, n_selected=rft.n_selected)
                
            #else:
            #    feat_tr = feat
            self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
            self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
            feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
            print(f"Shape of features after RFT {feat_tr.shape}")
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg.reshape(-1), Y_selg.reshape(-1), level, g_name = g_name)
            pred, res_pred = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg.reshape(-1), Y_selg.reshape(-1), level, 'train_0', g_name = g_name)
            #sel_eps = self.hard_sample_select(level, pred, Y_selg.reshape(-1),res_pred )
            #Energy based hard sample selection 
            #res = Y_selg.reshape(-1) - pred
            #g = self.energy_split(level, X0_sel, pred.reshape(N,h,w), 'tr' )
            #sel_eps = g[1]
            #plot_residue([res[sel_eps]], f"Residue Histogram Round 2 Level {level}", self.args['img_root'])
            #self.rft_feat_selection(level, feat[sel_eps], res[sel_eps],'round_2', feat_idx, self.args['img_root'], 0.8, f'level_{level}')
            #feat_tr = feat[:, self.rft_all[f'level_{level}']['round_2']]
            #self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1])[sel_eps], prev_pred_selg[sel_eps], Y_selg.reshape(-1)[sel_eps], level, g_name = 'grp_1')
            #pred,res_pred = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg.reshape(-1), Y_selg.reshape(-1), level, 'train_1', g_name = 'grp_1')
            #pred = pred.reshape(N, h,w)
            feat_tr = None
            gc.collect()
            
        return 
    def test_level(self, level, X0, X, Y, mode, prev_pred = None, idx_hard = None):
        
        if(level == 0):
            pred = self.Get_KMeans_InitPred(X, 2)
        else:
            Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            N,h,w = Y.shape
            #prev_pred = prev_pred.reshape(N,int(h/2),int(w/2))
            if(idx_hard is not None):
                sel = idx_hard
                g_name = '2grp2_1'
            else: 
                sel = np.arange(0, len(Y.reshape(-1)), 1)
                g_name = '2grp2_0'

            feat_inp = self.Get_PixelHopFeat(level, X, 'inp', 0,self.args['win'][level - 1], g_name =g_name )
            feat_inp = self.Add_SpatialFeat(feat_inp, X, level,self.args['fov'][level - 1],'inp' )
            feat_laws = self.Add_LawsFeat(level, X, 'te')
            feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
            if(level <4 and idx_hard is None):
                prev_pred = upscale_pred(prev_pred, (2*prev_pred.shape[-1], 2*prev_pred.shape[-1]), order= 'lanczos')
                prev_pred = truncation(prev_pred)
            pred = prev_pred.reshape(-1).copy()
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred,-1), 'prev', 0,self.args['win'][level - 1], g_name = g_name )
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred,-1), level,self.args['fov'][level - 1],'prev' )
            #feat_laws = self.Add_LawsFeat(level, prev_pred, 'te')
            #feat_prev = np.concatenate([feat_prev, feat_laws], axis = -1)
            feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1])], axis = -1)
            #feat = feat_inp.reshape(-1, feat_inp.shape[-1])
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            #feat_rft = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            rft = self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']
            
            feat_rft = rft.transform(feat, n_selected = rft.n_selected)
            feat = None
            gc.collect()
            if(self.args['lnt'][level-1]):
                
                feat_trl = np.copy(feat_rft)
                
                feat_lnt = self.Get_LNT(feat_rft, level, g_name)
                feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
                rft = self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']
                feat_rft = rft.transform(feat_trl, n_selected = rft.n_selected)
                #feat_rft = feat_trl[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
        
            feat_rft = (feat_rft - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
            """
            if(level == 4):
                sel = np.arange(0, len(prev_pred.reshape(-1)), 1)
            else: 
            
                sel = self.select_roi(X0, np.expand_dims(prev_pred,-1), level, mode)
            """
            
            print(f"Selected number of samples: {len(sel)}")
            pred[sel],res_pred  = self.Get_XGBoost_Pred(feat_rft.reshape(-1, feat_rft.shape[-1])[sel], prev_pred.reshape(-1)[sel], Y.reshape(-1)[sel], level, mode, g_name = g_name)
            #a = np.where(res_pred<-self.eps[f'{level}'])[0]
            #b = np.where(res_pred>self.eps[f'{level}'])[0]
            #samp = np.concatenate([a,b])
            #g = self.energy_split(level, X0, pred.reshape(N,h,w), 'te' )
            #samp = g[1]
            #feat_tr = feat[:, self.rft_all[f'level_{level}']['round_2']]
            #pred,res_pred = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred.reshape(-1), Y.reshape(-1), level, mode, g_name = 'grp_1')
            #pred = pred.reshape(N, h,w)
            feat_rft = None
            gc.collect()
        return pred.reshape(N,h,w)

    def train_level16(self,level, X0, X, Y, prev_pred = None, sel = None, idx_hard = None):     
        
        if(level == 0):
            sel = np.arange(0,len(X))
            idx = random.sample(sel.tolist(), int(len(sel)/2))
            self.Train_KMeans_InitPred(X[idx], 2)
            pred = self.Get_KMeans_InitPred(X, 2)
            
        else:

            win = self.args['win'][level-1]
            ld = 2**(level-1) 
            if(idx_hard is None): 
                self.pixelhop_all[f'inplevel_{level}'] = dict()
                self.pixelhop_prev[f'prevlevel_{level}'] = dict()
                self.rft_all[f'level_{level}'] = dict()
                
                self.lnt_all[f'level_{level}'] = dict()
                self.xgb_all[f'level_{level}'] = dict()
                self.feat_mean[f'level_{level}'] = dict(); self.feat_std[f'level_{level}'] = dict()
            Y_orig = Y.copy()
            Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            N,h,w = Y.shape
            idx = np.arange(0, len(X), 1)
            """
            if(self.args['downsample_patches'][level-1]!=1 and sel == None):
                sel = random.sample(idx.tolist(), int(self.args['downsample_patches'][level-1]*len(X)))
                X_sel = X[sel]; Y_sel = Y[sel]; prev_pred_sel = prev_pred[sel]; X0_sel = X0[sel]; Y_orig_sel = Y_orig[sel];
            else:
                X_sel = X; Y_sel = Y; prev_pred_sel = prev_pred; X0_sel = X0; Y_orig_sel = Y_orig;
            """
            X_sel = X; Y_sel = Y; prev_pred_sel = prev_pred; X0_sel = X0; Y_orig_sel = Y_orig;
            #X_samples = utils.Shrink_patch(X0_sel, ld, win, 1, int(win//2)) 
            #pp_samples = utils.Shrink_patch(np.expand_dims(prev_pred_sel,-1), 1, win , 1, int(win//2)) 
            #_, ud_idxs, rot_idxs = self.patch_alignment(X_samples)
            #if(level <4 and idx_hard is None): 
            #    prev_pred_sel = upscale_pred(prev_pred_sel, (2*prev_pred_sel.shape[-1], 2*prev_pred_sel.shape[-1]), order= 'lanczos')
            #    prev_pred_sel = truncation(prev_pred_sel)
            #sel = self.select_roi(X0_sel, np.expand_dims(prev_pred_sel,-1), level, 'tra')
            #print(f"Selected number of samples: {len(sel)}")
            #g = np.arange(0, len(X_samples), 1)
            
            #ud_idx = ud_idxs; rot_idx = rot_idxs
            #g = np.insert(g, len(g), g+len(X_samples))
            #Y_selg = np.concatenate([Y_sel, Y_sel])
            Y_selg = Y_sel
            if(idx_hard is not None):
                g = idx_hard
                g_name = '2grp2_1'
            else: 
                g = np.arange(0, len(Y_sel.reshape(-1)), 1)
                g_name = '2grp2_0'
            """
            res_tr16 = np.squeeze(Y_selg) - prev_pred_sel

            mse_im = []
            for im in range(len(prev_pred_sel)):
                mse_im.append(calculate_mse(np.squeeze(Y_selg), prev_pred_sel))
            """
            

            #abs_res = np.abs(Y_selg.reshape(-1)- prev_pred_sel.reshape(-1))
            #sort_absrestr = np.argsort(abs_res)

            pred_sel1000, _ = rec_img(prev_pred_sel, prev_pred_sel.reshape(len(prev_pred_sel),args['patch_size'], args['patch_size'] ), args['patch_size'], len(prev_pred_sel), 30)
            blobs_dog = []
            for im in pred_sel1000:
                blob = blob_dog(im, max_sigma=30, threshold=0.1)
                center_map = np.zeros_like(im, dtype=np.uint8)

                for y, x, _ in blob:
                    center_map[int(y), int(x)] = 1
                blobs_dog.append(center_map)
            

            blobs_dog = np.asarray(blobs_dog)
            blobs_dog = view_as_windows(np.asarray(blobs_dog), (1,16,16), (1, 12, 12))
            blobs_dog = blobs_dog.reshape(-1, 16,16)[sel]

            pred_sel16 = view_as_windows(np.asarray(pred_sel1000), (1,16,16), (1, 12, 12))
            pred_sel16 = pred_sel16.reshape(-1, 16,16)[sel]
            
            """
            sel_tr_all = sort_absrestr[int(len(sort_absrestr)/2):]
            self.thresh[f'level_{l}'] = np.min(abs_res[sel_tr_all])
            sel_pos = np.where(res_tr>self.thresh[f'level_{l}'])[0]
            sel_neg = np.where(res_tr<-self.thresh[f'level_{l}'] )[0]
            sel_neg = random.sample(sel_neg.tolist(),1200000 )
            sel_pos = random.sample(sel_pos.tolist(),1200000 )
            sel_tr = np.concatenate([sel_neg, sel_pos])
            sel_id = [i for i in range(len(sort_absrestr))]
            #sel_tr = random.sample(sel_tr_all.tolist(), int(0.15*len(sel_tr_all)))
            sel_easy = random.sample(sort_absrestr[:int(len(sort_absrestr)/2)].tolist(), int(0.1*len(sort_absrestr)) )
            sel_tr = np.concatenate([sel_neg, sel_pos, sel_easy])
            sel_id = np.where(Y.reshape(-1)>0.9)
            """
            #X_sel3 = X_sel[:,(3-l+1): X_sel.shape[2] - (3-l+1), (3-l+1): X_sel.shape[2] - (3-l+1) ]
            self.Train_PixelHop(level, X_sel, X0_sel,np.expand_dims(pred_sel16,-1), 'inp', 3, self.args['win'][level - 1],[g], g_name, None)
           
            feat_inp = self.Get_PixelHopFeat(level, X_sel, 'inp', 0,self.args['win'][level - 1],g_name)
            feat_inp = self.Add_SpatialFeat(feat_inp, X_sel, level,self.args['fov'][level - 1],'inp' )
            feat_laws = self.Add_LawsFeat(level, X_sel, 'tr')
            feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])
            print(f"Shape of input features: {feat_inp.shape}")
            
            
            self.Train_PixelHop(level, np.expand_dims(pred_sel16,-1),X0_sel,np.expand_dims(pred_sel16,-1), 'prev', 3, self.args['win'][level - 1],[g], g_name, None)
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred_sel16,-1), 'prev', 0,self.args['win'][level - 1], g_name )
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred_sel16,-1), level,self.args['fov'][level - 1],'prev' )
            #feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(blobs_dog,-1), level,self.args['fov'][level - 1],'prev' )

            #feat_laws = self.Add_LawsFeat(level, prev_pred_sel, 'tr')
            #feat_prev = np.concatenate([feat_prev, feat_laws], axis = -1)
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])
            print(f"Shape of prev pred features: {feat_prev.shape}")
            feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1])], axis = -1)[g]
            print(f"Shape of all features: {feat.shape}")
            #feat = feat_inp.reshape(-1, feat_inp.shape[-1])[g]
            #prev_pred_selg = np.concatenate([prev_pred_sel, prev_pred_sel])  
            
            
            feat = feat; 
            
            Y_all = list(Y_selg.reshape(-1))
            prev_pred_selg =list(pred_sel16.reshape(-1))

            """
            for aug in range(1,3):
                if(aug%2==0):
                    place_holder = np.zeros(prev_pred_sel.shape).reshape(-1)
                    place_holder[sel_pos] = 1
                    place_holder = place_holder.reshape(prev_pred_sel.shape)
                else: 
                    place_holder = np.zeros(prev_pred_sel.shape).reshape(-1)
                    place_holder[sel_neg] = 1
                    place_holder = place_holder.reshape(prev_pred_sel.shape)
                place_holder_aug = self.transform_images_onetype(np.squeeze(place_holder), aug,Y_all )
                #sel_aug = np.where(place_holder_aug.reshape(-1)==1)[0]
                place_holder_windows = view_as_windows(np.pad(place_holder_aug, ((0,0), (1,1),(1,1))), (1,3,3), (1,1,1))
                if(aug<8):
                    sel_aug = np.where(np.sum(place_holder_windows.reshape(-1,9), axis = -1)>1)[0]
                else: 
                    sel_aug = sel_tr.copy()
                
                #X0_aug = self.transform_images_onetype(np.squeeze(X0_sel), aug)
                X_aug = self.transform_images_onetype(np.squeeze(X_sel), aug,Y_all)

                if(aug<8):
                    Y_aug = self.transform_images_onetype(np.squeeze(Y_selg), aug,Y_all)
                    prev_pred_aug = self.transform_images_onetype(np.squeeze(prev_pred_sel), aug,Y_all)
                    #Y_orig_sel_aug = self.transform_images_onetype(np.squeeze(Y_orig_sel), aug)
                else: 
                    Y_aug = np.squeeze(Y_selg.copy())
                    #Y_orig_sel_aug = np.squeeze(Y_orig_sel.copy())
                    prev_pred_aug = prev_pred_sel.copy()
                #prev_pred_aug = self.load_prevlevel(level, np.expand_dims(X_aug,-1), np.expand_dims(X0_aug,-1), Y_orig_sel_aug)
                #if(level <4 and idx_hard is None): 
                #    prev_pred_aug = upscale_pred(prev_pred_aug, (2*prev_pred_aug.shape[-1], 2*prev_pred_aug.shape[-1]), order= 'lanczos')
                #    prev_pred_aug = truncation(prev_pred_aug)
                #prev_aug = self.transform_images_onetype(np.squeeze(prev_pred_aug), aug)
                feat_inp = self.Get_PixelHopFeat(level, np.expand_dims(X_aug,-1), 'inp', 0,self.args['win'][level - 1], g_name)
                feat_inp = self.Add_SpatialFeat(feat_inp,  np.expand_dims(X_aug,-1), level,self.args['fov'][level - 1],'inp')
                feat_laws = self.Add_LawsFeat(level,  np.expand_dims(X_aug,-1), 'te')
                
                feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(prev_pred_aug,-1), 'prev', 0,self.args['win'][level - 1], g_name )
                feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(prev_pred_aug,-1), level,self.args['fov'][level - 1],'prev' )
                feat_inp = np.concatenate([feat_inp, feat_laws, feat_prev], axis = -1)
                feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[sel_aug]
                feat = np.concatenate([feat, feat_inp], axis = 0)
                Y_all.extend(Y_aug.reshape(-1)[sel_aug])
                prev_pred_selg.extend(prev_pred_aug.reshape(-1)[sel_aug])
            """
            Y_selg = np.asarray(Y_all)
        
            prev_pred_selg = np.asarray(prev_pred_selg)
            #pred = prev_pred_selg.reshape(-1).copy()
            Y_selg = Y_selg.reshape(-1); prev_pred_selg = prev_pred_selg.reshape(-1)
            
            
            res = Y_selg - prev_pred_selg
            
            feat_idx = np.arange(0, feat.shape[-1], 1)
            
            """
            a = np.where(abs(res)<0.1)[0]
            idx = np.arange(0, len(res),1) 
            tr_idx = np.setdiff1d(idx, a).tolist()
            ds = random.sample(a.tolist(), int(0.4*len(a)))
            tr_idx.extend(ds)
            """
            
            plot_residue([res], f"Residue Histogram Level Hard {level}", self.args['img_root'])
            
            #res_new = np.sign(res)*np.log(1+np.abs(res))

            #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}_{g_name}')
            #feat_tr = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            
            X_train, X_val, y_train, y_val = train_test_split(feat, Y_selg.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
            rft = FeatureTest('rmse')
            rft.fit(X_train, y_train, n_bins=16, outliers=True)
            rft.plot(path=os.path.join(self.args['img_root'], f"train_rft_{level}_{g_name}.png"))

                    #logger.info(f'Val RFT, shape: {X_val.shape}')
            rft_val = FeatureTest('rmse')
            rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
            rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rft_{level}_{g_name}.png"))

            plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"joint_{level}_{g_name}.png"))
            rft.n_selected = int(0.8*len(feat_idx))
            self.rft_all[f'level_{level}'][f'before_lnt_{g_name}'] = rft
                    #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                    #feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            feat_tr = rft.transform(feat, n_selected=rft.n_selected)
            
            if(self.args['lnt'][level-1]):
                
                feat_trl = np.copy(feat_tr)
               
                    
                self.Train_LNT(feat_tr, Y_selg, level, g_name)
                feat_lnt = self.Get_LNT(feat_tr, level, g_name)
                feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
                plot(feat_tr, Y_selg, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
                feat_idx = np.arange(0, feat_trl.shape[-1], 1)
                #self.rft_feat_selection(level, feat_trl, res,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level - 1][int(g_name[-1])], f'level_{level}LNT_{g_name}')
                #feat_tr = feat_trl[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
                
                X_train, X_val, y_train, y_val = train_test_split(feat_trl, Y_selg.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
                rft = FeatureTest('rmse')
                rft.fit(X_train, y_train, n_bins=16, outliers=True)
                rft.plot(path=os.path.join(self.args['img_root'], f"train_rftlnt_{level}_{g_name}.png"))

                        #logger.info(f'Val RFT, shape: {X_val.shape}')
                rft_val = FeatureTest('rmse')
                rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
                rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rftlnt_{level}_{g_name}.png"))

                plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"jointlnt_{level}_{g_name}.png"))
                rft.n_selected = int(self.args['FS'][level - 1][int(g_name[-1])]*len(feat_idx))
                self.rft_all[f'level_{level}'][f'after_lnt_{g_name}'] = rft
                feat_tr = rft.transform(feat_trl, n_selected=rft.n_selected)
                
            #else:
            #    feat_tr = feat
            self.feat_mean[f'level_{level}'][g_name] = np.mean(feat_tr,axis=0,keepdims=True)
            self.feat_std[f'level_{level}'][g_name] = np.std(feat_tr,axis=0,keepdims=True)
            feat_tr = (feat_tr - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
            print(f"Shape of features after RFT {feat_tr.shape}")
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg.reshape(-1), Y_selg.reshape(-1), level, g_name = g_name)
            pred, res_pred = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg.reshape(-1), Y_selg.reshape(-1), level, 'train_0', g_name = g_name)
            #sel_eps = self.hard_sample_select(level, pred, Y_selg.reshape(-1),res_pred )
            #Energy based hard sample selection 
            #res = Y_selg.reshape(-1) - pred
            #g = self.energy_split(level, X0_sel, pred.reshape(N,h,w), 'tr' )
            #sel_eps = g[1]
            #plot_residue([res[sel_eps]], f"Residue Histogram Round 2 Level {level}", self.args['img_root'])
            #self.rft_feat_selection(level, feat[sel_eps], res[sel_eps],'round_2', feat_idx, self.args['img_root'], 0.8, f'level_{level}')
            #feat_tr = feat[:, self.rft_all[f'level_{level}']['round_2']]
            #self.Train_XGBoost(feat_tr.reshape(-1, feat_tr.shape[-1])[sel_eps], prev_pred_selg[sel_eps], Y_selg.reshape(-1)[sel_eps], level, g_name = 'grp_1')
            #pred,res_pred = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred_selg.reshape(-1), Y_selg.reshape(-1), level, 'train_1', g_name = 'grp_1')
            #pred = pred.reshape(N, h,w)
            feat_tr = None
            gc.collect()
            
        return 
    def test_level16(self, level, X0, X, Y, mode, prev_pred = None, idx_hard = None):
        
        if(level == 0):
            pred = self.Get_KMeans_InitPred(X, 2)
        else:
            Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
            N,h,w = Y.shape
            #prev_pred = prev_pred.reshape(N,int(h/2),int(w/2))
            """
            if(idx_hard is not None):
                sel = idx_hard
                g_name = '2grp2_1'
            else: 
                sel = np.arange(0, len(Y.reshape(-1)), 1)
                g_name = '2grp2_0'
            """
            if(mode =='tr'):
                pred_sel1000, _ = rec_img(prev_pred, prev_pred.reshape(len(prev_pred),args['patch_size'], args['patch_size'] ), args['patch_size'], len(prev_pred), 30)
            else: 
                pred_sel1000, _ = rec_img(prev_pred, prev_pred.reshape(len(prev_pred),args['patch_size'], args['patch_size'] ), args['patch_size'], len(prev_pred), 14)
            g_name = '2grp2_0'

            blobs_dog = []
            for im in pred_sel1000:
                blob = blob_dog(im, max_sigma=30, threshold=0.1)
                center_map = np.zeros_like(im, dtype=np.uint8)

                for y, x, _ in blob:
                    center_map[int(y), int(x)] = 1
                blobs_dog.append(center_map)
            

            blobs_dog = np.asarray(blobs_dog)
            blobs_dog = view_as_windows(np.asarray(blobs_dog), (1,16,16), (1, 12, 12))
            blobs_dog = blobs_dog.reshape(-1, 16,16)[idx_hard]
            pred_sel16 = view_as_windows(np.asarray(pred_sel1000), (1,16,16), (1, 12, 12))
            pred_sel16 = pred_sel16.reshape(-1, 16,16)[idx_hard]

            feat_inp = self.Get_PixelHopFeat(level, X, 'inp', 0,self.args['win'][level - 1], g_name =g_name )
            feat_inp = self.Add_SpatialFeat(feat_inp, X, level,self.args['fov'][level - 1],'inp' )
            feat_laws = self.Add_LawsFeat(level, X, 'te')
            feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
            """
            if(level <4 and idx_hard is None):
                prev_pred = upscale_pred(prev_pred, (2*prev_pred.shape[-1], 2*prev_pred.shape[-1]), order= 'lanczos')
                prev_pred = truncation(prev_pred)
            """
            #pred = prev_pred.reshape(-1).copy()
            feat_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred_sel16,-1), 'prev', 0,self.args['win'][level - 1], g_name = g_name )
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred_sel16,-1), level,self.args['fov'][level - 1],'prev' )
            #feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(blobs_dog,-1), level,self.args['fov'][level - 1],'prev' )

            #feat_laws = self.Add_LawsFeat(level, prev_pred, 'te')
            #feat_prev = np.concatenate([feat_prev, feat_laws], axis = -1)
            feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1])], axis = -1)
            #feat = feat_inp.reshape(-1, feat_inp.shape[-1])
            feat_lnt = None
            gc.collect()
            feat_prev = None
            gc.collect()
            feat_inp = None
            gc.collect()
            #feat_rft = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
            rft = self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']
            
            feat_rft = rft.transform(feat, n_selected = rft.n_selected)
            feat = None
            gc.collect()
            if(self.args['lnt'][level-1]):
                
                feat_trl = np.copy(feat_rft)
                
                feat_lnt = self.Get_LNT(feat_rft, level, g_name)
                feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
                rft = self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']
                feat_rft = rft.transform(feat_trl, n_selected = rft.n_selected)
                #feat_rft = feat_trl[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
        
            feat_rft = (feat_rft - self.feat_mean[f'level_{level}'][g_name])/self.feat_std[f'level_{level}'][g_name]
            """
            if(level == 4):
                sel = np.arange(0, len(prev_pred.reshape(-1)), 1)
            else: 
            
                sel = self.select_roi(X0, np.expand_dims(prev_pred,-1), level, mode)
            """
            
            #print(f"Selected number of samples: {len(sel)}")
            pred,res_pred  = self.Get_XGBoost_Pred(feat_rft.reshape(-1, feat_rft.shape[-1]), pred_sel16.reshape(-1), Y.reshape(-1), level, mode, g_name = g_name)
            #a = np.where(res_pred<-self.eps[f'{level}'])[0]
            #b = np.where(res_pred>self.eps[f'{level}'])[0]
            #samp = np.concatenate([a,b])
            #g = self.energy_split(level, X0, pred.reshape(N,h,w), 'te' )
            #samp = g[1]
            #feat_tr = feat[:, self.rft_all[f'level_{level}']['round_2']]
            #pred,res_pred = self.Get_XGBoost_Pred(feat_tr.reshape(-1, feat_tr.shape[-1]), prev_pred.reshape(-1), Y.reshape(-1), level, mode, g_name = 'grp_1')
            #pred = pred.reshape(N, h,w)
            feat_rft = None
            gc.collect()
        return pred

    
    

    def transform_images_onetype(self, samples, aug, gt, samp_0 = None):
        '''
        samples, [n, h, w]
        aug_type_list: e.g. ['flipud', 'rot_anti_cw_270'] for successive operations
            optional list element: 'rot_anti_cw_90', # i.e. 'rot_cw_270'
                                'rot_anti_cw_180', # i.e. 'rot_cw_180'
                                'rot_anti_cw_270', # i.e. 'rot_cw_90'
                                'flipud',
                                'fliplr',
                                'None'
                                anything else generates original input samples as output
        return samples_transformed, np.ndarray, the same shape as the input samples
        '''
        # samples_shape = samples.shape
        # assert len(samples_shape) == 3 or len(samples_shape) == 4
        
        samples_transformed = samples.copy()
        #gt_transformed = gt.copy()
        if (samp_0 is not None):
            samp_0_transformed = samp_0.copy()
       
            # print('aug type', aug)
        if aug == 1:
            
            samples_transformed = np.rot90(samples_transformed,
                                            k=1, axes=(-2, -1)).copy()
        elif aug == 2:
            samples_transformed = np.rot90(samples_transformed,
                                            k=2, axes=(-2, -1)).copy()
        elif aug == 3:
           
            samples_transformed = np.rot90(samples_transformed,
                                            k=1, axes=(-1, -2)).copy()
        elif aug == 4:
            samples_transformed = np.flip(samples_transformed, axis=-2).copy()
        elif aug == 5:
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(gaussian_blur(samples_transformed[n]))
            #samples_transformed = np.asarray(shifted)
            samples_transformed = np.flip(samples_transformed, axis=-1).copy()
        elif aug == 6:
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(random_brightness_shift(samples_transformed[n],0.2))
            #samples_transformed = np.asarray(shifted)
            samples_transformed = np.flip(samples_transformed, axis=-2)
            samples_transformed = np.rot90(samples_transformed,
                                            k=1, axes=(-2, -1)).copy()
        elif aug == 7: 
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(random_contrast(samples_transformed[n]))
            #samples_transformed = np.asarray(shifted)
            samples_transformed = np.flip(samples_transformed, axis=-2)
            samples_transformed = np.rot90(samples_transformed,
                                            k=1, axes=(-1, -2)).copy()
        elif aug == 8:
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(random_brightness_shift(samples_transformed[n],0.2))
            samples_transformed = np.asarray(shifted)

        elif aug == 9:
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(random_gamma(samples_transformed[n]))
            samples_transformed = np.asarray(shifted)

        elif aug == 10:
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(random_contrast(samples_transformed[n]))
            samples_transformed = np.asarray(shifted)
        
        elif aug == 11:
            shifted = []
            for n in range(len(samples_transformed)):
                shifted.append(gaussian_blur(samples_transformed[n]))
            samples_transformed = np.asarray(shifted)

        return samples_transformed



    def train_init(self, level, X0, X, Y, pred_b=None):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        Y_grad = []
        for im in range(len(Y)):
            Y_grad.append(utils.get_edges(Y[im]))
        Y_grad = np.asarray(Y_grad)
        Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
        
        N,h,w = Y.shape

        
        self.pixelhop_prev[f'prevlevel_{level}'] = dict()
        self.pixelhop_all[f'inplevel_{level}'] = dict()
        
        self.Train_PixelHop(level, X0, X0 ,X0, 'init', 3, 5, group= None, g_name= None, thresh = None, ud_idx = None, rot_idx = None)
        feat_inp = self.Get_PixelHopFeat(level, X0, 'init', 0,5, g_name= None, ud_idx = None, rot_idx = None)
        feat_inp = self.Add_SpatialFeat(feat_inp, X0, level,5,'inp' )
        feat_laws = self.Add_LawsFeat(level, X0, 'tr')
        feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
        feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])
        print(f"Shape of input features: {feat_inp.shape}")

        """
        self.Train_PixelHop(0, np.expand_dims(X_K,-1), X0 ,X0, 'prevb', 3, 5, group= None, g_name= None, thresh = None, ud_idx = None, rot_idx = None)
        feat_k = self.Get_PixelHopFeat(0, np.expand_dims(X_K,-1), 'prevb', 0,5, g_name= None, ud_idx = None, rot_idx = None)
        feat_k = self.Add_SpatialFeat(feat_k, np.expand_dims(X_K,-1), level,5,'prev' )
        feat_k = feat_k.reshape(-1, feat_k.shape[-1])
        print(f"Shape of input features: {feat_k.shape}")
        """

        if(pred_b is not None):
            self.Train_PixelHop(0, np.expand_dims(pred_b,-1),X0,np.expand_dims(pred_b,-1), 'prev', 3,  5, group= None, g_name= None, thresh = None, ud_idx = None, rot_idx = None)
            feat_prev = self.Get_PixelHopFeat(0, np.expand_dims(pred_b,-1), 'prev', 0,5, g_name= None, ud_idx = None, rot_idx = None)
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred_b,-1), level,5,'prev')
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])
            print(f"Shape of prev pred features: {feat_prev.shape}")
            feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_k.reshape(-1, feat_k.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1])], axis = -1)
        else: 
            #feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1])], axis = -1)
            feat = feat_inp.reshape(-1, feat_inp.shape[-1])
        feat_idx = np.arange(0, feat.shape[-1], 1)
        g_name = 'init'
        #RFT
        self.rft_all[f'level_init'] = dict()
        self.lnt_all[f'level_init'] = dict()
        #self.rft_feat_selection('init', feat, Y.reshape(-1),'before_lnt', feat_idx, self.args['img_root'], 1, f'initlevel_{level}')
        
        X_train, X_val, y_train, y_val = train_test_split(feat, Y.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
        rft = FeatureTest('rmse')
        rft.fit(X_train, y_train, n_bins=16, outliers=True)
        rft.plot(path=os.path.join(self.args['img_root'], f"train_rft_{level}_{g_name}.png"))

                #logger.info(f'Val RFT, shape: {X_val.shape}')
        rft_val = FeatureTest('rmse')
        rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
        rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rft_{level}_{g_name}.png"))

        plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"joint_{level}_{g_name}.png"))
        rft.n_selected = int(len(feat_idx))
        self.rft_all[f'level_init']['before_lnt'] = rft
                #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                #feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
        feat_tr = rft.transform(feat, n_selected=rft.n_selected)
        
        #feat_tr = feat[:, self.rft_all[f'level_init']['before_lnt']]
        Y = Y.reshape(-1)
        if(self.args['lnt'][level-1]):
            #subsets = [0, 50, 100, 150,200,250  ]
            feat_trl = np.copy(feat_tr)
            
            self.Train_LNT(feat_tr, Y.reshape(-1), 'init', g_name)
            feat_lnt = self.Get_LNT(feat_tr, 'init', g_name)
            feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
            #feat_trl = feat_lnt
            #plot(feat_tr, Y.reshape(-1), feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
            feat_idx = np.arange(0, feat_trl.shape[-1], 1)
            self.rft_feat_selection('init', feat_trl, Y.reshape(-1),f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}LNT_{g_name}')
            
            X_train, X_val, y_train, y_val = train_test_split(feat_trl, Y.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
            rft = FeatureTest('rmse')
            rft.fit(X_train, y_train, n_bins=16, outliers=True)
            rft.plot(path=os.path.join(self.args['img_root'], f"train_rftlnt_{level}_{g_name}.png"))

                    #logger.info(f'Val RFT, shape: {X_val.shape}')
            rft_val = FeatureTest('rmse')
            rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
            rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rftlnt_{level}_{g_name}.png"))

            plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"jointlnt_{level}_{g_name}.png"))
            rft.n_selected = int(1*len(feat_idx))
            self.rft_all[f'level_init'][f'after_lnt_{g_name}'] = rft
            feat_tr = rft.transform(feat_trl, n_selected=rft.n_selected)
                
            #feat_tr = feat_trl[:, self.rft_all[f'level_init'][f'after_lnt_{g_name}']]
        
        self.feat_mean[f'init_{level}'] = np.mean(feat_tr,axis=0,keepdims=True)
        self.feat_std[f'init_{level}'] = np.std(feat_tr,axis=0,keepdims=True)
        feat_tr = (feat_tr - self.feat_mean[f'init_{level}'])/self.feat_std[f'init_{level}'] 
        print(f"Shape of features after RFT {feat_tr.shape}")

        feat_inp = None
        gc.collect()

        feat_trl = None
        gc.collect()
        feat_lnt = None
        gc.collect()
        feat_laws = None
        gc.collect()
        
        self.xgb_all[f'initlevel_{level}'] = []
        X_train, X_val, y_train, y_val = train_test_split(feat_tr, Y.reshape(-1), test_size=0.2,
                                                          random_state=42)
        
        pred_tr = np.zeros(len(y_train)); pred_val = np.zeros(len(y_val))
        xgbr = train_iter_init(self.args, level, X_train, y_train, X_val, y_val, g_name, plot = True, savepath = os.path.join(args['img_root'], f'xgboost_level{level}_init.png'))
        self.xgb_all[f'initlevel_{level}'].append(xgbr)
        rounds=int(len(X_train)/1000000)+1
        respred_tr = []
        for r in range(rounds):
            respred_tr.extend(xgbr.predict(X_train[r*1000000:(r+1)*1000000]))
        #respred_tr = xgbr.predict(X_train)
        respred_val = xgbr.predict(X_val)
        pred_tr = truncation(respred_tr+pred_tr)
        pred_val = truncation(respred_val+pred_val)
        
        mse_tr = calculate_mse(pred_tr, y_train)
        mse_val = calculate_mse(pred_val, y_val)
        print(f'Iter {0}: MSE Train = {mse_tr}, MSE Val = {mse_val}')

        pred_all = xgbr.predict(feat_tr)

        abs_res = np.abs(Y.reshape(-1)- pred_all)
        res_tr = Y.reshape(-1)- pred_all
        sort_absrestr = np.argsort(abs_res)
        
        sel_tr_all = sort_absrestr[int(len(sort_absrestr)/2):]
        self.thresh['init'] = np.min(abs_res[sel_tr_all])
        sel_pos = np.where(res_tr>self.thresh['init'])[0]
        sel_neg = np.where(res_tr<-self.thresh['init'])[0]
        sel_easy = sort_absrestr[:int(len(sort_absrestr)/2)]
        sel_tr = np.concatenate([sel_pos, sel_neg])
        #sel_id = [i for i in range(len(res_tr))]
        a = np.where(pred_all>0.25)[0]
        b = np.where(pred_all<0.75)[0]
        sel_id = np.intersect1d(a,b)
        sel_id = np.unique(sel_id).tolist()
        #sel_id = np.where(Y.reshape(-1)>0.9)[0]
        #place_holder = np.zeros((len(X0), 32,32)).reshape(-1)
        #place_holder[sel_hard] = 1
        #place_holder = place_holder.reshape((len(X0), 32,32))
        Y_all = list(Y)
        for aug in range(1,12):
            place_holder = np.zeros((len(X0), 32,32)).reshape(-1)
            if(aug%2==0):
                place_holder[sel_tr] = 1
            else: 
                place_holder[sel_tr] = 1
            place_holder = place_holder.reshape((len(X0), 32,32))           
            if(aug<8):
                place_holder_aug = self.transform_images_onetype(np.squeeze(place_holder), aug, Y_all)
                #sel_aug = np.where(place_holder_aug.reshape(-1)==1)[0]
                place_holder_windows = view_as_windows(np.pad(place_holder_aug, ((0,0), (1,1),(1,1))), (1,3,3), (1,1,1))
                sel_aug = np.where(np.sum(place_holder_windows.reshape(-1,9), axis = -1)>2)[0]
            else: 
                #sel_aug = random.sample(sel_easy.tolist(), int(0.5*len(sel_easy)))
                #sel_aug = random.sample(sel_id, int(0.75*len(sel_id)))
                sel_aug= sel_easy.copy()
            X0_aug = self.transform_images_onetype(np.squeeze(X0), aug, Y_all)
            if(aug< 8):
                Y_aug = self.transform_images_onetype(np.squeeze(Y.reshape(place_holder.shape)), aug, Y_all)
            else: 
                Y_aug = Y.reshape(place_holder.shape).copy()
            feat_inp = self.Get_PixelHopFeat(level, np.expand_dims(X0_aug,-1), 'init', 0,5, g_name= None, ud_idx = None, rot_idx = None)
            feat_inp = self.Add_SpatialFeat(feat_inp,  np.expand_dims(X0_aug,-1), level,5,'inp' )
            feat_laws = self.Add_LawsFeat(level,  np.expand_dims(X0_aug,-1), 'te')
            feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
            feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])[sel_aug]
            feat = np.concatenate([feat, feat_inp], axis = 0)
            Y_all.extend(Y_aug.reshape(-1)[sel_aug])

        Y = np.asarray(Y_all)
            
        self.rft_all[f'level_init'] = dict()
        self.lnt_all[f'level_init'] = dict()
        #self.rft_feat_selection('init', feat, Y.reshape(-1),'before_lnt', feat_idx, self.args['img_root'], 1, f'initlevel_{level}')
        
        X_train, X_val, y_train, y_val = train_test_split(feat, Y.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
        rft = FeatureTest('rmse')
        rft.fit(X_train, y_train, n_bins=16, outliers=True)
        rft.plot(path=os.path.join(self.args['img_root'], f"train_rft_{level}_{g_name}.png"))

                #logger.info(f'Val RFT, shape: {X_val.shape}')
        rft_val = FeatureTest('rmse')
        rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
        rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rft_{level}_{g_name}.png"))

        plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"joint_{level}_{g_name}.png"))
        rft.n_selected = int(len(feat_idx))
        self.rft_all[f'level_init']['before_lnt'] = rft
                #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                #feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
        feat_tr = rft.transform(feat, n_selected=rft.n_selected)
        
        #feat_tr = feat[:, self.rft_all[f'level_init']['before_lnt']]
        #Y = Y.reshape(-1)
        if(self.args['lnt'][level-1]):
            #subsets = [0, 50, 100, 150,200,250  ]
            feat_trl = np.copy(feat_tr)
            
            self.Train_LNT(feat_tr, Y.reshape(-1), 'init', g_name)
            feat_lnt = self.Get_LNT(feat_tr, 'init', g_name)
            feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
            #feat_trl = feat_lnt
            plot(feat_tr, Y.reshape(-1), feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
            feat_idx = np.arange(0, feat_trl.shape[-1], 1)
            self.rft_feat_selection('init', feat_trl, Y.reshape(-1),f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 1, f'level_{level}LNT_{g_name}')
            
            X_train, X_val, y_train, y_val = train_test_split(feat_trl, Y.reshape(-1), test_size=0.2,
                                                          random_state=42)
                #RFT 
            rft = FeatureTest('rmse')
            rft.fit(X_train, y_train, n_bins=16, outliers=True)
            rft.plot(path=os.path.join(self.args['img_root'], f"train_rftlnt_{level}_{g_name}.png"))

                    #logger.info(f'Val RFT, shape: {X_val.shape}')
            rft_val = FeatureTest('rmse')
            rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
            rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rftlnt_{level}_{g_name}.png"))

            plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"jointlnt_{level}_{g_name}.png"))
            rft.n_selected = int(len(feat_idx))
            self.rft_all[f'level_init'][f'after_lnt_{g_name}'] = rft
            feat_tr = rft.transform(feat_trl, n_selected=rft.n_selected)
        
            #feat_tr = feat_trl[:, self.rft_all[f'level_init'][f'after_lnt_{g_name}']]
        
        self.feat_mean[f'init_{level}'] = np.mean(feat_tr,axis=0,keepdims=True)
        self.feat_std[f'init_{level}'] = np.std(feat_tr,axis=0,keepdims=True)
        feat_tr = (feat_tr - self.feat_mean[f'init_{level}'])/self.feat_std[f'init_{level}'] 
        print(f"Shape of features after RFT {feat_tr.shape}")

        feat_inp = None
        gc.collect()

        X_train, X_val, y_train, y_val = train_test_split(feat_tr, Y, test_size=0.2,
                                                          random_state=42)
        
    
        #abs_res_tr = np.abs(y_train - pred_tr)
        #abs_res_val = np.abs(y_val - pred_val)
        
        
        #sel_tr = np.where(abs_res_tr>=self.thresh['init'])[0]
        
        #sel_val = np.where(abs_res_val>=self.thresh['init'])[0]
        
        g_name = 'init_r2'
        
        #self.xgb_all[f'initlevel_{level}_{g_name}'] = []
        
        #X_train, X_val, y_train, y_val = train_test_split(feat_tr, Y.reshape(-1), test_size=0.2,
        #                                                  random_state=42)
        
        pred_tr = np.zeros(len(y_train)); pred_val = np.zeros(len(y_val))
        xgbr = train_iter_init(self.args, level, X_train, y_train, X_val, y_val, g_name,sel_tr = None, res = None, plot = True, savepath = os.path.join(args['img_root'], f'xgboost_level{level}_initr2.png'))
        self.xgb_all[f'initlevel_{level}'].append(xgbr)
        rounds=int(len(X_train)/1000000)+1
        respred_tr = []
        for r in range(rounds):
            respred_tr.extend(xgbr.predict(X_train[r*1000000:(r+1)*1000000]))
        #respred_tr = xgbr.predict(X_train)
        respred_val = xgbr.predict(X_val)
        pred_tr = truncation(respred_tr+pred_tr)
        pred_val = truncation(respred_val+pred_val)
        

        mse_tr = calculate_mse(pred_tr, y_train)
        mse_val = calculate_mse(pred_val, y_val)
        print(f'Iter {1}: MSE Train = {mse_tr}, MSE Val = {mse_val}')

    def get_init(self, level, X0, X, Y, mode, pred_b = None):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        Y_grad = []
        for im in range(len(Y)):
            Y_grad.append(utils.get_edges(Y[im]))
        Y_grad = np.asarray(Y_grad)
        Y = block_reduce(Y, (1, 2**(level-1), 2**(level-1)), np.mean)
        """
        if(level!=1):
            for l in range(level-1):
                X0 =self.lp(X0)
                Y = np.squeeze(self.lp(Y))
        """
        N,h,w = Y.shape

       
        g_name = 'init'
        feat_inp = self.Get_PixelHopFeat(level, X0, 'init', 0,5, g_name= None, ud_idx = None, rot_idx = None)
        feat_inp = self.Add_SpatialFeat(feat_inp, X0, level,5,'inp' )
        feat_laws = self.Add_LawsFeat(level, X0, 'te')
        feat_inp = np.concatenate([feat_inp, feat_laws], axis = -1)
        feat_inp = feat_inp.reshape(-1, feat_inp.shape[-1])
        print(f"Shape of input features: {feat_inp.shape}")
       
        
        if(pred_b is not None):
            

            feat_prev = self.Get_PixelHopFeat(0, np.expand_dims(pred_b,-1), 'prev', 0,5, g_name= None, ud_idx = None, rot_idx = None)
            feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred_b,-1), level,5,'prev')
            feat_prev = feat_prev.reshape(-1, feat_prev.shape[-1])
            print(f"Shape of prev pred features: {feat_prev.shape}")
            feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_k.reshape(-1, feat_k.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1])], axis = -1)
        else: 
            #feat = np.concatenate([feat_inp.reshape(-1, feat_inp.shape[-1]), feat_k.reshape(-1, feat_k.shape[-1])], axis = -1)
            feat = feat_inp.reshape(-1, feat_inp.shape[-1])
        #feat_tr = feat[:, self.rft_all[f'level_init']['before_lnt']]
        rft = self.rft_all[f'level_init']['before_lnt']
        feat_tr = rft.transform(feat, n_selected = rft.n_selected)
        if(self.args['lnt'][level-1]):
            
            feat_trl = np.copy(feat_tr)
           
            feat_lnt = self.Get_LNT(feat_tr, 'init', g_name)
            feat_trl = np.concatenate([feat_trl, feat_lnt], axis = -1)
            #feat_rft = feat_lnt
            print(feat_trl.shape)
            rft = self.rft_all[f'level_init'][f'after_lnt_{g_name}']
            feat_tr = rft.transform(feat_trl, n_selected = rft.n_selected)
            print(feat_tr.shape)
            #feat_rft = feat_rft[:, self.rft_all[f'level_init'][f'after_lnt_{g_name}']]
        feat_tr = (feat_tr - self.feat_mean[f'init_{level}'])/self.feat_std[f'init_{level}'] 
        feat_rft = None
        gc.collect()
        feat_lnt = None
        feat_trl = None
        feat_inp = None
        gc.collect()

        pred = np.zeros(len(feat_tr))
        rounds=int(len(feat_tr)/1000000)+1
        respred = []
        for r in range(rounds):
            respred.extend(self.xgb_all[f'initlevel_{level}'][1].predict(feat_tr[r*1000000:(r+1)*1000000]))

        pred= truncation(respred+pred)
        
        res = Y.reshape(-1) - pred 

        
        plot_residue([res], f"Residue Histogram XGBoost Init {level} {mode}", self.args['img_root'])
        plot_residue([Y.reshape(-1),pred], f"GT vs Prediction Histogram Init {level} {mode}", self.args['img_root'])
        plot_scatter(Y.reshape(-1), pred,self.args['img_root'], f"GT vs Prediction Scatter Plot Init {level} {mode}" )
        mse_te = calculate_mse(pred, Y.reshape(-1))

        if(mode == 'te'):
            print(f'Iter {0}: Test MSE = {mse_te}')
        elif(mode == 'tr'):
            print(f'Iter {0}: Train MSE = {mse_te}')

        feat_tr = None
        gc.collect()
        
        return pred.reshape(N,h,w)
    
    def augment_patch(self, patch,gt, mode):

        if(mode == 'r1'):
            aug_patch = np.rot90(patch,1)
            aug_gt = np.rot90(gt, 1)
        elif(mode == 'r2'):
            aug_patch = np.rot90(patch,2)
            aug_gt = np.rot90(gt, 2)
        elif(mode == 'r3'):
            aug_patch = np.rot90(patch,3)
            aug_gt = np.rot90(gt, 3)
        elif(mode == 'lr'):
            aug_patch = np.flip(patch,1)
            aug_gt = np.flip(gt, 1)
        elif(mode == 'ud'):
            aug_patch = np.flip(patch,0)
            aug_gt = np.flip(gt, 0)
            
            
        return aug_patch, aug_gt
        
    def tf_equalize(self, patches):
        patch255 = (patches*255).astype('uint8')
        patch_eq = patch255.copy()
        for ind in range(len(patch255)):
            p = patch255[ind]
            unique, counts = np.unique(p, return_counts=True)
            freq = np.zeros(256)
            freq[unique] = counts
            freq = freq/(p.shape[0]*p.shape[1])
            cdf = np.asarray(list(itertools.accumulate(freq)))
            cdf = (cdf*255).astype('uint8')
            a = np.where(cdf<=0.3*255)[0]
            #for i in range(np.min(p), np.max(p)+1):
            #   patch_eq[ind][patch_eq[ind]==i] = cdf[i]
            patch_eq[ind][patch_eq[ind]<np.max(a)] = 1
            patch_eq[ind][patch_eq[ind]>=np.max(a)] = 0

        return patch_eq/255
    
    def load_patches(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['win'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 14
        
        X = []; Y= []; L = []
        if(level>0):
            pad_width = int(win//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loader_test(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
           
            #augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.2, sigma2=0.2)
            #augmentor.fit(image)
            #augmented_images = []
            #for _ in range(2):
            #    augmented_image = augmentor.pop()
            #    augmented_images.append(augmented_image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            
            YUV = color.rgb2lab(H)
            #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
            patches_r = im_to_blocks_padte(patch_size, H,0)
            X.extend(patches_r)
            #L.extend(im_to_blocks_te(patch_size, (YUV))
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte(patch_size, gt))
            
            if(aug == True):
                _, H_s, E = normalizeStaining_aug(img = image)
                if(ind%5 ==0):  
                    H_aug, gt_aug = self.augment_patch(H_s,gt,'r1')
                elif(ind%5 ==1):  
                    H_aug, gt_aug = self.augment_patch(H_s,gt,'r2')
                elif(ind%5 ==2):   
                    H_aug, gt_aug = self.augment_patch(H_s,gt,'r3')
                elif(ind%5 ==3):   
                    H_aug, gt_aug = self.augment_patch(H_s,gt,'lr')
                elif(ind%5 ==4):   
                    H_aug, gt_aug = self.augment_patch(H_s,gt,'ud')
                
                #patch_r=cv2.resize(H_aug, dim, interpolation=cv2.INTER_LANCZOS4)
                YUV = color.rgb2lab(H_s)
                #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                patches_r = im_to_blocks_padte(patch_size, YUV,0)
                X.extend(patches_r)
                #L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #patches_r = im_to_blocks_padte(patch_size, H_aug,0)
                #X.extend(patches_r)
                #gt_r=cv2.resize(gt_aug, dim, interpolation=cv2.INTER_LANCZOS4)
            
                Y.extend(im_to_blocks_gtte(patch_size, gt))
                
        L = np.asarray(L)
        X = np.asarray(X)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra"):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model, mm=channels(X_s)    
            save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
            save_model(mm, os.path.join(save_path, f"mmX_{level}.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
        else:
            sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
            mm = load_model(os.path.join(save_path, f"mmX_{level}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model,mm)
        
        
        X_P=np.asarray(X_trainP)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return np.clip(1-X_P, 0, 1), Y

    def load_im(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['fov'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 14
        
        X = []; Y= []; L = []; X0 = []
        pad_width = 0
        #pad_width = 12
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loader_test(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            YUV = (color.rgb2yuv(H)*255).astype('uint8')
            #_, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            images.append(H)
            Y.append(gt)
            
        images = np.asarray(images); Y = np.asarray(Y)
        return images, Y

    def load_patches_te(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['fov'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 14
        
        X = []; Y= []; L = []; X0 = []
        if(level>0):
            pad_width = int((win)//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loader_test(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            YUV = color.rgb2lab(image)
            #_, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            YUV = color.rgb2lab(H)
            #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
                
            patches_r = im_to_blocks_padte(patch_size, H, pad_width)
            patches_r0 = im_to_blocks_padte(patch_size, H, 0)
            X.extend(patches_r)
            X0.extend(patches_r0)
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte(patch_size, gt))

            if(aug == True):

                if(ind%5 ==0):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'lr')
                elif(ind%5 ==1):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'ud')
                elif(ind%5 ==2):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r1')
                elif(ind%5 ==3):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r2')
                elif(ind%5 ==4):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r3')
                YUV = color.rgb2lab(H_aug)
                #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                patches_r = im_to_blocks_padte(patch_size, YUV, pad_width)
                patches_r0 = im_to_blocks_padte(patch_size, YUV, 0)
                #images.append(H_s)
                X.extend(patches_r)
                X0.extend(patches_r0)
                YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
                YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                #L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
                
                Y.extend(im_to_blocks_gtte(patch_size, gt_aug))
                
               
        X = np.asarray(X); X0 = np.asarray(X0)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra" ):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model=channels(X_s)  
            #X_trainP = channels_test(X_s, bias_term,color_pca_model)  
            save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
            #save_model(mm, os.path.join(save_path, f"mmX_{level}.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        else:
            sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{level}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model)
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        
        X_P=np.asarray(X_trainP); #X0_P=np.asarray(X_train0P)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return np.clip(1-X_P, 0, 1), Y

    def load_patches_tecpm(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['fov'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 32
        
        X = []; Y= []; L = []; X0 = []
        if(level>0):
            pad_width = int((win)//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        stride = patch_size
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loadercpm(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            YUV = color.rgb2lab(image)
            #_, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            YUV = color.rgb2lab(H)
            #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
                
            patches_r = im_to_blocks_padte(patch_size, H, pad_width, stride)
            patches_r0 = im_to_blocks_padte(patch_size, H, 0, stride)
            X.extend(patches_r)
            X0.extend(patches_r0)
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte(patch_size, gt, stride))

            if(aug == True):

                if(ind%5 ==0):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'lr')
                elif(ind%5 ==1):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'ud')
                elif(ind%5 ==2):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r1')
                elif(ind%5 ==3):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r2')
                elif(ind%5 ==4):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r3')
                YUV = color.rgb2lab(H_aug)
                #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                patches_r = im_to_blocks_padte(patch_size, YUV, pad_width)
                patches_r0 = im_to_blocks_padte(patch_size, YUV, 0)
                #images.append(H_s)
                X.extend(patches_r)
                X0.extend(patches_r0)
                YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
                YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                #L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
                
                Y.extend(im_to_blocks_gtte(patch_size, gt_aug))
                
               
        X = np.asarray(X); X0 = np.asarray(X0)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra" ):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model,mm=channels(X_s)  
            #X_trainP = channels_test(X_s, bias_term,color_pca_model)  
            save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
            save_model(mm, os.path.join(save_path, f"mmX_{level}.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        else:
            sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{level}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model)
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        
        X_P=np.asarray(X_trainP); #X0_P=np.asarray(X_train0P)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return np.clip(1-X_P, 0, 1), Y

    def load_patches_tetnbc(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['fov'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 50
        
        X = []; Y= []; L = []; X0 = []
        if(level>0):
            pad_width = int((win)//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        stride = patch_size
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loadertnbc(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            YUV = color.rgb2lab(image)
            #_, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            YUV = color.rgb2lab(H)
            #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
                
            patches_r = im_to_blocks_padte(patch_size, H, pad_width, stride)
            patches_r0 = im_to_blocks_padte(patch_size, H, 0, stride)
            X.extend(patches_r)
            X0.extend(patches_r0)
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte(patch_size, gt, stride))

            if(aug == True):

                if(ind%5 ==0):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'lr')
                elif(ind%5 ==1):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'ud')
                elif(ind%5 ==2):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r1')
                elif(ind%5 ==3):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r2')
                elif(ind%5 ==4):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r3')
                YUV = color.rgb2lab(H_aug)
                #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                patches_r = im_to_blocks_padte(patch_size, YUV, pad_width)
                patches_r0 = im_to_blocks_padte(patch_size, YUV, 0)
                #images.append(H_s)
                X.extend(patches_r)
                X0.extend(patches_r0)
                YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
                YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                #L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
                
                Y.extend(im_to_blocks_gtte(patch_size, gt_aug))
                
        X = np.asarray(X); X0 = np.asarray(X0)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra" ):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model,mm=channels(X_s)  
            #X_trainP = channels_test(X_s, bias_term,color_pca_model)  
            save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
            save_model(mm, os.path.join(save_path, f"mmX_{level}.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        else:
            sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{level}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model)
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        
        X_P=np.asarray(X_trainP); #X0_P=np.asarray(X_train0P)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return np.clip(1-X_P, 0, 1), Y

    def load_patches_te16(self,mode, aug, level, bias_term = None):
        patch_size = 16
        save_path = self.args['modelroot']
        win = self.args['fov'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 14
        
        X = []; Y= []; L = []; X0 = []
        if(level>0):
            pad_width = int((win)//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loader_test(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            #_, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            YUV = color.rgb2lab(H)
            #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
                
            patches_r = im_to_blocks_padte16(patch_size, H, pad_width)
            patches_r0 = im_to_blocks_padte16(patch_size, H, 0)
            X.extend(patches_r)
            X0.extend(patches_r0)
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte16(patch_size, gt))

            if(aug == True):

                if(ind%5 ==0):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'lr')
                elif(ind%5 ==1):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'ud')
                elif(ind%5 ==2):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r1')
                elif(ind%5 ==3):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r2')
                elif(ind%5 ==4):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r3')
                YUV = color.rgb2lab(H_aug)
                #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                patches_r = im_to_blocks_padte(patch_size, YUV, pad_width)
                patches_r0 = im_to_blocks_padte(patch_size, YUV, 0)
                #images.append(H_s)
                X.extend(patches_r)
                X0.extend(patches_r0)
                YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
                YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                #L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
                
                Y.extend(im_to_blocks_gtte(patch_size, gt_aug))
                
            
        X = np.asarray(X); X0 = np.asarray(X0)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra" ):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model,mm=channels(X_s)  
            #X_trainP = channels_test(X_s, bias_term,color_pca_model)  
            save_model(sc, os.path.join(save_path, f"scX_{level}_.pkl"))
            save_model(mm, os.path.join(save_path, f"mmX_{level}_.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}_.pkl"))
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        else:
            sc = load_model(os.path.join(save_path, f"scX_{level}_.pkl"))
            mm = load_model(os.path.join(save_path, f"mmX_{level}_.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}_.pkl"))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model,mm)
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        
        X_P=np.asarray(X_trainP); #X0_P=np.asarray(X_train0P)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return np.clip(1-X_P, 0, 1), Y
    
    def load_patches_tecons(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['win'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 14
        
        X = []; Y= []; L = []; X0 = []
        if(level>0):
            pad_width = int((win-2)//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loadercons(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            _, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            patches_r = im_to_blocks_padte(patch_size, H, pad_width)
            patches_r0 = im_to_blocks_padte(patch_size, H, 0)
            images.append(H)
            X.extend(patches_r)
            X0.extend(patches_r0)
            YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
            YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
            L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte(patch_size, gt))

            if(aug == True):
                patches_r = im_to_blocks_padte(patch_size, H_s, pad_width)
                patches_r0 = im_to_blocks_padte(patch_size, H_s, 0)
                images.append(H_s)
                X.extend(patches_r)
                X0.extend(patches_r0)
                YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
                YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
                
                Y.extend(im_to_blocks_gtte(patch_size, gt))
            
        X = np.asarray(X); X0 = np.asarray(X0)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra" and os.path.exists(os.path.join(save_path, f"scX_{level}.pkl")) is False):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model=channels(X_s)  
            #X_trainP = channels_test(X_s, bias_term,color_pca_model)  
            save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
            sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            X_train0P = channels_test(X0_s, 0,color_pca_model)
        else:
            #sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
            #X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model)
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            sc = StandardScaler().fit(X0.reshape(len(X0), (patch_size)*(patch_size)*3))
            X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            X_train0P = channels_test(X0_s, 0,color_pca_model)
        
        
        X_P=np.asarray(X_trainP); X0_P=np.asarray(X_train0P)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return X_P, Y, X0_P, images

    def load_patches_tec(self,mode, aug, level, bias_term = None):
        patch_size = self.args['patch_size']
        save_path = self.args['modelroot']
        win = self.args['fov'][level-1]
        if(mode == 'tr' or mode =='tra'):
            num = 30            
        elif(mode == 'te'):
            num = 30
        
        X = []; Y= []; L = []; X0 = []
        if(level>0):
            pad_width = int((win)//2)*(2**(level-1))
        else:
            pad_width = 0
        #pad_width = 12
        stride = 256
        images = []
        for ind in range(num):
            #print('------------------------') 
            #print(ind)  
            # load image
            if(mode == 'tr' or mode =='tra'):
                [image, gt, basename] = image_patch_loader(ind)
            elif(mode == 'te'):
                [image, gt, basename] = image_patch_loaderc(ind)
            #print(basename)
            #image =  exposure.equalize_adapthist(image.astype('uint8'))*255
            _, H, E = normalizeStaining(img = image)
            YUV = color.rgb2lab(image)
            #_, H_s, E = normalizeStaining_aug(img = image)
            #enhanced_image = exposure.equalize_adapthist(H.astype('uint8'))
            #H = (exposure.equalize_adapthist(H.astype("uint8"))*255).astype('uint8')
            gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
            
            y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
            #r = 1024.0 / 1000
            #dim = (1024, int(1000 * r))    
            #patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
            
            #Padding input and GT to 1024, 1024
            #H = np.pad(H, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
            #gt = np.pad(gt, ((12,12), (12,12),(0,0) ), mode='reflect')
            YUV = color.rgb2lab(H)
            #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            YUV = minmax_normalize(YUV)
                
            patches_r = im_to_blocks_padte(patch_size, H, pad_width, stride)
            patches_r0 = im_to_blocks_padte(patch_size, H, 0, stride)
            X.extend(patches_r)
            X0.extend(patches_r0)
            #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
            
            Y.extend(im_to_blocks_gtte(patch_size, gt, stride))

            if(aug == True):

                if(ind%5 ==0):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'lr')
                elif(ind%5 ==1):  
                    H_aug, gt_aug = self.augment_patch(H,gt,'ud')
                elif(ind%5 ==2):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r1')
                elif(ind%5 ==3):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r2')
                elif(ind%5 ==4):   
                    H_aug, gt_aug = self.augment_patch(H,gt,'r3')
                YUV = color.rgb2lab(H_aug)
                #YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                patches_r = im_to_blocks_padte(patch_size, YUV, pad_width)
                patches_r0 = im_to_blocks_padte(patch_size, YUV, 0)
                #images.append(H_s)
                X.extend(patches_r)
                X0.extend(patches_r0)
                YUV = cv2.cvtColor(H, cv2.COLOR_RGB2YUV)[:,:,0]
                YUV = cv2.Sobel(src=YUV, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
                YUV = minmax_normalize(YUV)
                #L.extend(im_to_blocks_te(patch_size, np.expand_dims(YUV,-1)))
                #gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
                
                Y.extend(im_to_blocks_gtte(patch_size, gt_aug))
                
               
        X = np.asarray(X); X0 = np.asarray(X0)
        Y = np.squeeze(np.asarray(Y).astype('uint8'))
        if(mode == "tra" ):
            sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #self.patch_mean = np.mean(X, axis = 0)
            #X_s = X - self.patch_mean
            X_trainP, color_pca_model,mm=channels(X_s)  
            #X_trainP = channels_test(X_s, bias_term,color_pca_model)  
            save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
            save_model(mm, os.path.join(save_path, f"mmX_{level}.pkl"))
            save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        else:
            sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{level}.pkl"))
            color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
            X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
            #X_s = X - self.patch_mean
            X_trainP = channels_test(X_s, 0,color_pca_model)
            #sc = load_model(os.path.join(save_path, f"scX_{0}.pkl"))
            #mm = load_model(os.path.join(save_path, f"mmX_{0}.pkl"))
            #color_pca_model = load_model(os.path.join(save_path, f"pcaX_{0}.pkl"))
            #X0_s = sc.transform(X0.reshape(len(X0), (patch_size)*(patch_size)*3)).reshape(len(X0), patch_size,patch_size,3)
            #X_train0P = channels_test(X0_s, 0,color_pca_model,mm)
        
        X_P=np.asarray(X_trainP); #X0_P=np.asarray(X_train0P)
        #X_P = self.tf_equalize(X_P)
        #X_P =  exposure.equalize_adapthist(X_P)
        print(f'Loaded {num} images')
        
        return np.clip(1-X_P, 0, 1), Y
    
    def Train_KMeans_InitPred(self, patches, win_size, num_clusters = 1024):
        if(len(patches.shape)>3):
            patches = np.squeeze(patches)
        patches = block_reduce(np.squeeze(patches), (1,8,8), np.mean)
        feat = view_as_windows(patches, (1, win_size, win_size), (1,1,1))
        
        #feat = sc.fit_transform(feat.reshape(len(feat),-1)).reshape(len(feat), 32,32)
        #test_feat = sc.fit_transform(test_feat.reshape(len(test_feat),-1)).reshape(len(test_feat), 32,32)
        mse_tr = []; mse_val = []
       
        self.K_Means =  MiniBatchKMeans(
                n_clusters= num_clusters,
                random_state=42,
                batch_size=1000000,
                verbose=0
            ).fit((feat.reshape(-1, win_size*win_size)))
        
        print("Trained K-Means")
           
    def Get_KMeans_InitPred(self, patches, win_size ):
        if(len(patches.shape)>3):
            patches = np.squeeze(patches)
        patches = block_reduce(np.squeeze(patches), (1,8,8), np.mean)
        feat = view_as_windows(patches, (1, win_size, win_size), (1,win_size,win_size))
        pred = self.K_Means.predict(feat.reshape(-1, win_size*win_size))
        cc = self.K_Means.cluster_centers_.reshape(-1, win_size, win_size)
        res = cc[pred]
        init_pred = duplicate_1d(res, win_size, 4)
        print(f'Shape of Initial Predictions {init_pred.shape}')
        return init_pred
    
    def Train_PixelHop(self, level, data_all, X0, prev_pred, mode,  sampling, win,group, g_name, thresh = None, ud_idx = None, rot_idx = None):
        SaabArgs = self.args['SaabArgs']
        concatArg = self.args['concatArg']
        
        
        
        # -----------Data Preprocessing-----------
        #data_all = np.load(dataroot+filename)
        #data_all = data_all[:,:,:,0]
        data_all = data_all[:,np.newaxis,:,:,:]
        
        print('Shape of all data is {}'.format(data_all.shape))
        
        if(mode == 'inp'):
            if(isinstance(level, int)): 
                data_all = block_reduce(data_all, (1,1, 2**(level-1), 2**(level-1) ,1), np.mean)
            print('Shape of all data is {}'.format(data_all.shape))
            if(g_name[:4]!='laws'):
                shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]
            elif(g_name[:4]=='laws' and level == 'clf'): 
                shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': int(13//2), 'pool': 1,'aug':Shrink_patch}]
            else:   
                shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]

        elif(mode == 'prev' or mode =='prevb'):
            if(isinstance(level, int)):
                shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': int(self.args['fov'][level-1]//2), 'pool': 1,'aug':Shrink_patch}]
            else: 
                shrinkArgs = [{'func':Shrink, 'win':self.args['fov'][0]+2, 'stride': 1, 'pad': int(self.args['fov'][0]//2), 'pool': 1,'aug':Shrink_patch}]
        elif( mode == 'patches'):
            shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]
        else: 
            data_all = block_reduce(data_all, (1,1, 2**(level-1), 2**(level-1) ,1), np.mean)
            print('Shape of all data is {}'.format(data_all.shape))
            shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]
        data_tr = data_all[np.array(range(0,len(data_all),sampling))]
        
        if(g_name is not None and g_name[:4]!='laws'):
            if(mode != 'patches'):
                X0 = X0[:,:,:,:,np.newaxis]
                
                prev_pred = prev_pred[:,:,:,:,np.newaxis]
                X0_tr = X0[np.array(range(0,len(X0),sampling))]
                prev_pred_tr = prev_pred[np.array(range(0,len(X0),sampling))]
                if(thresh is not None):
                    group = self.energy_split_test(X0_tr[:,:,:,:,0], prev_pred_tr[:,:,:,:,0],level,thresh, 'orig')
                
                elif(g_name[:3] != 'grp'): 
                    idx = np.arange(0, len(prev_pred_tr.reshape(-1)), 1).tolist()
                    if(int(g_name[-1])==0):
                        group = []
                        group.append(idx)
                    else: 
                        group.append(idx)
                else: 
                    a = np.where(prev_pred_tr.reshape(-1)>0.1)[0]
                    b = np.where(prev_pred_tr.reshape(-1)<0.9)[0]
                    group.append(np.intersect1d(a,b).tolist())


                layer1 = torchSaab(kernel_size=(shrinkArgs[0]['win']-2,shrinkArgs[0]['win']-2), stride=(shrinkArgs[0]['stride'],shrinkArgs[0]['stride']),pad = (shrinkArgs[0]['pad'],shrinkArgs[0]['pad']), channelwise=False, ud_idx = ud_idx, rot_idx = rot_idx, grp = group[int(g_name[-1])])
                model = sslModel([layer1],"saab")
                
            else:
                if(g_name[-2]!='p'):
                    layer1 = torchSaab(kernel_size=(11,11), stride=(1,1), pad = (0,0), channelwise=False, ud_idx = ud_idx, rot_idx = rot_idx, grp = group[int(g_name[-1])])
                else: 
                    layer1 = torchSaab(kernel_size=(16,16), stride=(1,1), pad = (0,0), channelwise=False, ud_idx = ud_idx, rot_idx = rot_idx, grp = group[int(g_name[-1])])

                model = sslModel([layer1],"saab")
                

                                
        else:
            group = np.arange(0, len(data_tr.reshape(-1)),1).tolist()
            layer1 = torchSaab(kernel_size=(shrinkArgs[0]['win']-2,shrinkArgs[0]['win']-2), stride=(shrinkArgs[0]['stride'],shrinkArgs[0]['stride']), pad = (shrinkArgs[0]['pad'],shrinkArgs[0]['pad']),channelwise=False, ud_idx= ud_idx, rot_idx = rot_idx, grp = group,)
            model = sslModel([layer1], "saab")
            

        print('Shape of training data is {}'.format(data_tr.shape))
        
        for channel in range(1):
            # -----------Train Voxelhop-----------
            print(channel)
            start = time.time()
            
                
            
            #p5.fit(data_tr[:,:,:,channel])
            model.fit(data_tr[:,:,:,:,channel])
            #p5.save(modelroot+'Saab_HP' + str(level)+ str(channel) + str(shrinkArgs['win']))
            data_tr = None
            gc.collect()
            if(mode == 'inp'):
                self.pixelhop_all[f'inplevel_{level}'][g_name] = model
            elif(mode == 'prev' or mode =='prevb'):
                self.pixelhop_prev[f'prevlevel_{level}'][f'{g_name}_{mode}'] = model
            elif(mode == 'patches'):
                self.pixelhop_haar[f'{level}'][g_name] = model
            else:
                self.pixelhop_all[f'inplevel_{level}_init'] = model 
            
            
        
        end = time.time()
        print('Pixelhop training or loading time is {}'.format(end - start))


    def Get_PixelHopFeat(self, level, data_all,mode, channel, win, g_name,ud_idx = None, rot_idx= None):
        SaabArgs = self.args['SaabArgs']
        concatArg = self.args['concatArg']
        
        data_all = data_all[:,np.newaxis,:,:,:]
        #print('Shape of all data is {}'.format(data_all.shape))
        
        if(mode == 'inp'):
            if(isinstance(level, int)): 
                data_all = block_reduce(data_all, (1,1, 2**(level-1), 2**(level-1) ,1), np.mean)
            print('Shape of all data is {}'.format(data_all.shape))
            shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]
            if(g_name[:4]=='laws'):
                shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': int(13/2), 'pool': 1,'aug':Shrink_patch}]
        elif(mode == 'prev' or mode =='prevb'):
            shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': int(win//2), 'pool': 1,'aug':Shrink_patch}]

        elif( mode == 'patches'):
            shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]
        else: 
            data_all = block_reduce(data_all, (1,1, 2**(level-1), 2**(level-1) ,1), np.mean)
            print('Shape of all data is {}'.format(data_all.shape))
            shrinkArgs = [{'func':Shrink, 'win':win, 'stride': 1, 'pad': 0, 'pool': 1,'aug':Shrink_patch}]
        
        # -----------Interface Pixelhop-----------
        #start = time.time()
        
        
        if(mode == 'inp'):
            model = self.pixelhop_all[f'inplevel_{level}'][g_name]
        elif(mode =='prev' or mode =='prevb'):
            if(f'{g_name}_{mode}' in  self.pixelhop_prev[f'prevlevel_{level}'].keys()):
                model = self.pixelhop_prev[f'prevlevel_{level}'][f'{g_name}_{mode}']
            else: 
                model = self.pixelhop_prev[f'prevlevel_{level}'][f'{g_name}']
        elif(mode == 'patches'):
            model = self.pixelhop_haar[f'{level}'][g_name]
        else: 
            model = self.pixelhop_all[f'inplevel_{level}_init']
        end = time.time()
    
       
        
        #print('Pixelhop training or loading time is {}'.format(end - start))
        #start = time.time()
        #save_feat(data_all[:,:,:,channel],p5,ch=channel, level = level, BS=0, mode=mode_save, saveroot=savepath, num_layers=1, output_all=True)
        if(ud_idx is not None):
            feat = model(data_all[:,:,:,:,channel], 'tr').transpose(0,2,3,1)
        else: 
            feat = model(data_all[:,:,:,:,channel], 'te').transpose(0,2,3,1)
        #feat = get_feat(data_all[:,:,:,channel], p5, num_layers =1, output_all= True )
        #end = time.time()
        
        #print('Pixelhop interface time is {}'.format(end - start))

        #data_all = None
        #gc.collect()
        print('Done!')
        
        if(mode!='patches'):
            if(g_name is None or g_name[:4]!='laws' ):
                if(level ==4):
                    feat_n = add_neighbor_feat(np.asarray(feat), 3, 1)
                else: 
                    if(isinstance(level, int)):
                        feat_n = add_neighbor_feat(np.asarray(feat), self.args['n_win'][level-1], 2)
                    else: 
                        feat_n = feat    
            else:
                if(level == 4):
                    feat_n = add_neighbor_feat(np.asarray(feat), self.args['laws_fov'][level-1], 1)
                else: 
                    if(isinstance(level, int)):
                        feat_n = add_neighbor_feat(np.asarray(feat), self.args['laws_fov'][level-1], 2)
                    else: 
                        feat_n = add_neighbor_feat(np.asarray(feat), 11, 2)
        if(mode != 'patches'):
            return feat_n
        else: 
            return feat.reshape(len(feat),-1)

    def Add_LawsFeat(self, level, patches, mode):
        l3 = (1/6)*np.asarray([1,2,1])
        e3 = (1/2)*np.asarray([-1,0,1])
        s3 = (1/2)*np.asarray([-1,2,-1])
        laws = [l3, e3, s3]
        laws_feat = []
        """
        if(patches.shape[1]==32 and level!=1):
            patches = block_reduce(np.squeeze(patches), (1, 2**(level-1), 2**(level-1)), np.mean)
        """
        for lar in range(2):
            for lac in range(2):
                if(lar==0 and lac ==0):
                    continue 
                kernel = np.outer(laws[lar],laws[lac])
                laws_feat.append(ndi.convolve(np.squeeze(patches), kernel[np.newaxis, :, :], mode='reflect'))
        
        feat = []
        grp = np.arange(0, len(laws_feat), 1).tolist()
        if(mode =='tr'):
            for lk in range(3):
                self.Train_PixelHop(level, np.expand_dims(laws_feat[lk],-1), laws_feat, laws_feat, 'inp', 3, 5, grp,f'lawsinit_{lk}', None, None, None)
                feat.append(self.Get_PixelHopFeat(level,np.expand_dims(laws_feat[lk],-1), 'inp', 0,5, f'lawsinit_{lk}', None, None ))
        else: 
            for lk in range(3):
                feat.append(self.Get_PixelHopFeat(level, np.expand_dims(laws_feat[lk],-1), 'inp', 0,5, f'lawsinit_{lk}', None, None ))

        return np.stack(feat,-1).reshape(feat[0].shape[0],feat[0].shape[1],feat[0].shape[2],-1)
        print("")
    
    def Add_SpatialFeat(self, feat, patches, level, win, mode, ud_idx = None, rot_idx = None):
        
        if(level!='clf'):
            ld = 2**(level-1)
        else: 
            ld= 1
        s_win = win - 2
        feat_lbp = None
        feat_spi = None
        if(mode == 'inp'):
            
            N,h,w, _ = feat.shape
            if(ud_idx is not None):
                feat_sp = utils.Shrink_patch(patches, ld, win+2, 1, 0)
                feat_spa = self.patch_alignment_test(feat_sp, ud_idx, rot_idx)
                feat_spa = np.squeeze(view_as_windows(np.squeeze(feat_spa), (1,s_win,s_win),(1,1,1)))
                feat_sp = feat_spa.reshape(2*N, int(h/ld),int(w/ld),feat_spa.shape[-3]*feat_spa.shape[-3]*s_win*s_win)
            else:
                
                    #feat_sp,_ = self.lp(patches, patches, patches)
                #patches = grad_mag(patches)
                feat_spi = utils.Shrink_patch(patches, ld, win, 1, 0)
                feat_spi = feat_spi.reshape(N, h,w,feat_spi.shape[-1]*feat_spi.shape[-1])
                patches = local_lbp(patches, ld)
                feat_sp = utils.Shrink_patch(np.expand_dims(patches,-1), 1, win, 1, 0)
                feat_sp = feat_sp.reshape(N, h,w,feat_sp.shape[-1]*feat_sp.shape[-1])

                l5 = [1, 4 , 6, 4, 1]
                e5 = [-1, -2, 0, 2, 1]
                s5 = [-1, 0, 2, 0, -1]
                w5 = [-1, 2, 0, -2, 1]
                r5 = [1, -4 , 6, -4, 1]

                l3 = (1/6)*np.asarray([1,2,1])
                e3 = (1/2)*np.asarray([1,0,-1])
                s3 = (1/2)*np.asarray([-1,2,-1])



                #laws = [l5, e5, s5, w5, r5]
                laws = [l3, e3, s3]
                
                #patches_laws = utils.Shrink_patch(patches, ld, s_win, 1, int((s_win)//2))
                
                #feat_spa = np.squeeze(view_as_windows(np.squeeze(feat_sp), (1,s_win,s_win),(1,1,1)))
                
                
                #feat_grad = np.max(np.abs(np.gradient(feat_spa, edge_order= 2, axis = 0).reshape(len(feat_spa),-1,feat_spa.shape[-1], feat_spa.shape[-1])), axis = 1)
                #feat_grada = feat_grad.reshape(N, int(h),int(w),feat_grad.shape[-1]*feat_grad.shape[-1])
        elif(mode == 'prev'):
            N,h,w, _ = patches.shape
            if(ud_idx is not None):
                feat_sp = utils.Shrink_patch(patches, 1, win+2, 1, int((win+2)//2))
                feat_spa = self.patch_alignment_test(feat_sp, ud_idx, rot_idx)
                feat_spa = np.squeeze(view_as_windows(np.squeeze(feat_spa), (1,s_win,s_win),(1,1,1)))
                feat_sp = feat_spa.reshape(2*N, h,w,feat_spa.shape[-3]*feat_spa.shape[-3]*s_win*s_win)
            else:
                #patches = grad_mag(patches)
                feat_sp = utils.Shrink_patch(patches, 1, win, 1, int((win)//2))
                if(isinstance(level, int) == False):
                    feat_lbp = self.extract_lbp_features(feat_sp[rot_idx] )
                #else: 
                #    feat_lbp = self.extract_lbp_features(feat_sp )
                if(level ==3):
                    feat_spa = np.squeeze(view_as_windows(np.squeeze(feat_sp), (1,s_win,s_win),(1,1,1)))
                else: 
                    feat_spa = np.squeeze(view_as_windows(np.squeeze(feat_sp), (1,s_win,s_win),(1,2,2)))
                
                #feat_spa = np.squeeze(view_as_windows(np.squeeze(feat_sp), (1,s_win,s_win),(1,1,1)))
                feat_sp = feat_sp.reshape(N, h,w,feat_sp.shape[-1]*feat_sp.shape[-1])
                
                #feat_grad = np.max(np.abs(np.gradient(feat_spa, edge_order= 2, axis = 0).reshape(len(feat_spa),-1,feat_spa.shape[-1], feat_spa.shape[-1])), axis = 1)
                #feat_grada = feat_grad.reshape(N, int(h),int(w),feat_grad.shape[-1]*feat_grad.shape[-1])
        elif(mode == 'patches'):
            _, h,w,_ = patches.shape
            if(h!=16):
                patches = patches[:, 5:h-5, 5:w-5]
            feat_sp = patches.reshape(-1, patches.shape[1]*patches.shape[1] )
        #feat_n = add_neighbor_feat(np.asarray(feat_sp), 3)
        feat_spa = None
        #gc.collect()
        

        if(len(feat)>0):
            if(feat_spi is not None):
                feat_all = np.concatenate([feat, feat_sp, feat_spi], axis = -1)
            else: 
                feat_all = np.concatenate([feat, feat_sp], axis = -1)
            if(feat_lbp is not None):
                return feat_all, feat_lbp
            else: 
                return feat_all
        else:
            return np.asarray(feat_sp)
    
    def Train_LNT(self, feat, res_tr, level, g_name):
        # LNT feature generation
        if(len(feat)>500000):
            sel =random.sample(np.arange(0, len(feat),1).tolist(), 500000)
        else: 
            sel = np.arange(0, len(feat),1).tolist()
        y_comb = select_comb(res_tr)
        lda = LinearRegression().fit(feat[sel]-np.mean(feat[sel], axis=0), y_comb[sel]-np.mean(y_comb[sel]))
        T = lda.coef_.transpose()
        svd = TruncatedSVD(n_components=T.shape[1]-1, random_state=42).fit(T)
        lnt_model = svd.transform(T)[:, np.argsort(svd.explained_variance_ratio_)[::-1]]
        
        self.lnt_all[f'level_{level}'][f'lnt_model_{g_name}'] = lnt_model
        
        if(level == 1):
            depth_lnt = [1]
        else: 
            depth_lnt = [1]
        
        for depth in depth_lnt:
            print('LNT depth is {}'.format(depth))
            idx = np.arange(0,len(feat),1)
            if(len(feat)>1000000):
                sel = random.sample(idx.tolist(), 1000000)
            else: 
                sel = idx
            feat_gene = LNT(depth=depth, num_tree=self.args['num_lnt'], mode='gpu', saveroot=os.path.join(self.args['img_root'], 'LNT'))
            feat_gene.fit(feat[sel], res_tr[sel])
            tr_new_feat = feat_gene.transform(feat)
            feat_idx = np.arange(0, tr_new_feat.shape[-1], 1)
            if(len(np.unique(res_tr))==3): 
                self.dft_feat_selection(level, tr_new_feat, res_tr,f'lnt_{depth}_{g_name}', feat_idx, self.args['img_root'], 0.8, f'level_{level}LNT_{depth}_{g_name}')
            else: 
                #self.rft_feat_selection(level, tr_new_feat, res_tr,f'lnt_{depth}_{g_name}', feat_idx, self.args['img_root'], 0.5, f'level_{level}LNT_{depth}_{g_name}')
                
                X_train, X_val, y_train, y_val = train_test_split(tr_new_feat, res_tr, test_size=0.2,
                                                          random_state=42)
                    #RFT 
                if(len(np.unique(res_tr))==2): 
                    loss = 'bce'
                else: 
                    loss = 'rmse'
                rft = FeatureTest(loss)
                rft.fit(X_train, y_train, n_bins=16, outliers=True)
                rft.plot(path=os.path.join(self.args['img_root'], f"train_rft_lnt_{level}_{depth}_{g_name}.png"))

                    #logger.info(f'Val RFT, shape: {X_val.shape}')
                rft_val = FeatureTest(loss)
                rft_val.fit(X_val, y_val, n_bins=16, outliers=True)
                rft_val.plot(path=os.path.join(self.args['img_root'], f"val_rft_lnt_{level}_{depth}_{g_name}.png"))

                plot_train_val_rank(rft, rft_val, path=os.path.join(self.args['img_root'], f"jointlnt_{level}_{depth}_{g_name}.png"))
                rft.n_selected = int(0.75*len(feat_idx))
                
                self.rft_all[f'level_{level}'][f'lnt_{depth}_{g_name}'] = rft
                    #self.rft_feat_selection(level, feat, res,f'before_lnt_{g_name}', feat_idx, self.args['img_root'], self.args['FS'][level-1][g_ind], f'level_{level}_{g_name}')
                    #feat = feat[:, self.rft_all[f'level_{level}'][f'before_lnt_{g_name}']]
                #feat_tr = rft.transform(tr_new_feat, n_selected=rft.n_selected)
           
            #model['lnt'] = feat_gene
            
            self.lnt_all[f'level_{level}'][f'{depth}_{g_name}'] = feat_gene
    
    
    
    def Get_LNT_feat(self, feat, y_train, comb_size, level, g_name):
        all_combs = list(combinations(range(feat.shape[1]), comb_size))
        lnt_feat_test_list = []
        y_tr = y_train.copy()
        y_trr_cls = select_comb(y_tr)
        for i, comb in enumerate(all_combs):
            X_test_sub = feat[:, comb]
            _, lnt_feat_te = lnt_kernel(X_test_sub, y_tr, y_trr_cls, svd=self.lnt_all[f'level_{level}'][f'lnt_model_{g_name}'][i])
            lnt_feat_test_list.append(lnt_feat_te)
        lnt_feat_test = np.concatenate(lnt_feat_test_list, axis=1)
        lnt_feat = lnt_feat_test[:,self.rft_all[f'level_{level}'][f'lnt_{g_name}']]

        return lnt_feat

    def Get_LNT(self, feat, level, g_name):
        feat_lnt = feat @ self.lnt_all[f'level_{level}'][f'lnt_model_{g_name}']
        if(level == 1):
            depth_lnt = [1]
        else: 
            depth_lnt = [1]
        
        for depth in depth_lnt:
            depth_feat = self.lnt_all[f'level_{level}'][f'{depth}_{g_name}'].transform(feat)
            #depth_feat = depth_feat[:,self.rft_all[f'level_{level}'][f'lnt_{depth}_{g_name}']]
            rft = self.rft_all[f'level_{level}'][f'lnt_{depth}_{g_name}']
            depth_feat = rft.transform(depth_feat, n_selected = rft.n_selected)
            feat_lnt = np.concatenate([feat_lnt, depth_feat], axis = -1)
            
        return feat_lnt

    def dft_feat_selection(self, level, x_tr, y_tr, mode,feat_idx, imgroot, FS, name ):
        idx = np.arange(0, len(x_tr), 1)
        if(len(idx)>1000000):
            sel = random.sample(idx.tolist(), 1000000)
        else:
            sel = idx
        X_train, X_val, y_train, y_val = train_test_split(x_tr[sel], y_tr[sel], test_size=0.2,
                                                          random_state=42)
        keep_num = int(X_train.shape[-1]*FS)
        feat_idx_selected_tr, tr_loss = feature_selection(X_train, y_train, FStype='DFT_entropy', thrs=1.0, B=16)
        feat_idx_selected_val, val_loss = feature_selection(X_val, y_val, FStype='DFT_entropy', thrs=1.0, B=16)
        feat_idx_selected_tr = feat_idx_selected_tr[:keep_num]
        feat_idx_selected_val = feat_idx_selected_val[:keep_num]
        feat_idx_selected = []
        for idx in feat_idx_selected_tr:
            if idx in feat_idx_selected_val:
                feat_idx_selected.append(idx)

        plot_loss_curve(tr_loss, f'train_{name}', imgroot)
        plot_loss_curve(val_loss, f'validation_{name}', imgroot)
        plot_loss_scatter(tr_loss, val_loss, name, imgroot)

        tr_loss.sort()
        dft_threshold = tr_loss[keep_num-1]
        print('DFT threshold: {}'.format(dft_threshold))
        
        self.rft_all[f'level_{level}'][mode] = feat_idx_selected
           
    def rft_feat_selection(self,level,x_tr, y_tr, mode, feat_idx, imgroot, FS, name):
        idx = np.arange(0, len(x_tr), 1)
        if(len(idx)>1000000):
            sel = random.sample(idx.tolist(), 1000000)
        else:
            sel = idx
        X_train, X_val, y_train, y_val = train_test_split(x_tr[sel], y_tr[sel], test_size=0.2,
                                                          random_state=42)
        feat_select = RFT(name=name)
        tr_loss = feat_select.fit(X_train, y_train)
        val_loss = feat_select.fit(X_val, y_val)
        #plot_joint_elbow(tr_loss, val_loss, name, imgroot)

        keep_num = int(X_train.shape[-1]*FS)
        feat_idx_selected_tr = feat_select.clf.get_selected_feat_idx_multidim(tr_loss, keep_num)
        feat_idx_selected_val = feat_select.clf.get_selected_feat_idx_multidim(val_loss, keep_num)
        feat_idx_selected = []
        for idx in feat_idx_selected_tr:
            if idx in feat_idx_selected_val:
                feat_idx_selected.append(idx)
    

        #feat_select.plot_loss_hist(tr_loss, feat_idx, imgroot)
        feat_select.plot_loss_curve(tr_loss, 'train', imgroot)
        feat_select.plot_loss_curve(val_loss, 'validation', imgroot)
        feat_select.plot_loss_scatter(tr_loss, val_loss, imgroot)

        tr_loss.sort()
        rft_threshold = tr_loss[keep_num-1]
        print('RFT threshold: {}'.format(rft_threshold))
        
        self.rft_all[f'level_{level}'][mode] = feat_idx_selected
    
    def Get_RFT_feat(self, level, feat, mode):
        return feat[:, self.rft_all[f'level_{level}']][mode]
    
    def Train_XGB_Haar(self, X, gt, level, g_name):
        Y = gt.copy()

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2,
                                                          random_state=42)
        
        self.xgb_all[f'level_{level}'][g_name] = []
        for iter in range(self.args['num_iter'][level-1]):
            
            xgbr = train_iter(self.args, level, X_train, y_train, X_val, y_val,g_name,  plot = True, savepath = os.path.join(args['img_root'], f'xgboost_level{level}_iter{iter}_{g_name}.png'))
            self.xgb_all[f'level_{level}'][g_name].append(xgbr)
            rounds=int(len(X_train)/1000000)+1
            respred_tr = []
            for r in range(rounds):
                respred_tr.extend(xgbr.predict(X_train[r*1000000:(r+1)*1000000]))
            #respred_tr = xgbr.predict(X_train)
            respred_val = xgbr.predict(X_val)
            
            mse_tr = calculate_mse(respred_tr, y_train)
            mse_val = calculate_mse(respred_val, y_val)
            print(f'Iter {iter}: MSE Train = {mse_tr}, MSE Val = {mse_val}')
            
    def Get_XGB_Haar(self, X, gt, level, g_name, mode):
        for iter in range(len(self.xgb_all[f'level_{level}'][g_name])):
            rounds=int(len(X)/1000000)+1
            iter_pred = []
            for r in range(rounds):
                iter_pred.extend(self.xgb_all[f'level_{level}'][g_name][iter].predict(X[r*1000000:(r+1)*1000000]))
            pred = iter_pred
            mse_te = calculate_mse(pred, gt)
            if(mode == 'te'):
                print(f'Iter {iter}: Test MSE = {mse_te}')
            elif(mode == 'tr'):
                print(f'Iter {iter}: Train MSE = {mse_te}')
            
        return pred
    def Train_XGBoost(self, X, prev_pred, gt, level, g_name):
        
        Y = [prev_pred, gt]
        
        X_train, X_val, y0_train, y0_val,y1_train, y1_val = train_test_split(X, Y[0], Y[1], test_size=0.2,
                                                          random_state=42)
        res_tr = y1_train - y0_train
        res_val = y1_val - y0_val
        mse_tr = calculate_mse(y0_train, y1_train)
        mse_val = calculate_mse(y0_val, y1_val)
        print(f'Initial MSE Train = {mse_tr}, MSE Val = {mse_val}')
        
        #pred_tr = y0_train.copy()
        #pred_val = y0_val.copy()
        pred_tr = np.zeros(len(y0_train))
        pred_val = np.zeros(len(y0_val))
        
        self.xgb_all[f'level_{level}'][g_name] = []
        for iter in range(self.args['num_iter'][level-1]):
            #Not reducing training samples
            if(len(X_train)>5000000):
                sel = [i for i in range(len(X_train))]
                idx = random.sample(sel, 5000000)
            else: 
                idx = [i for i in range(len(X_train))]
            xgbr = train_iter(self.args, level, X_train[idx], y1_train[idx], X_val, y1_val,g_name, plot = True, savepath = os.path.join(args['img_root'], f'xgboost_level{level}_iter{iter}_{g_name}.png'))
            self.xgb_all[f'level_{level}'][g_name].append(xgbr)
            rounds=int(len(X_train)/500000)+1
            respred_tr = []
            for r in range(rounds):
                respred_tr.extend(xgbr.predict(X_train[r*500000:(r+1)*500000]))
            #respred_tr = xgbr.predict(X_train)
            respred_val = xgbr.predict(X_val)
            #respred_tr = (np.sign(respred_tr)*(np.exp(np.abs(respred_tr))-1))
            #respred_val = (np.sign(respred_val)*(np.exp(np.abs(respred_val))-1))
            pred_tr = truncation(respred_tr+pred_tr)
            pred_val = truncation(respred_val+pred_val)
            res_tr =y1_train - pred_tr
            res_val = y1_val - pred_val
            
            mse_tr = calculate_mse(pred_tr, y1_train)
            mse_val = calculate_mse(pred_val, y1_val)
            print(f'Iter {iter}: MSE Train = {mse_tr}, MSE Val = {mse_val}')
            
    def Get_XGBoost_Pred(self, X, prev_pred, gt, level, mode,g_name):
        
        #pred = prev_pred.copy()
        #res = gt - pred
        
        mse_te = calculate_mse(prev_pred, gt)
        
        pred = np.zeros(len(prev_pred))
        if(mode == 'te'):
            print(f'Initial Test MSE = {mse_te}')
        elif(mode[:2] == 'tr'):
            print(f'Initial Train MSE = {mse_te}')
        
        for iter in range(len(self.xgb_all[f'level_{level}'][g_name])):
            rounds=int(len(X)/700000)+1
            iter_pred = []
            for r in range(rounds):
                iter_pred.extend(self.xgb_all[f'level_{level}'][g_name][iter].predict(X[r*700000:(r+1)*700000]))
            #iter_pred = (np.sign(iter_pred)*(np.exp(np.abs(iter_pred))-1))
            pred = truncation(iter_pred+pred)
            
            
            plot_residue([gt,iter_pred], f"GT vs Prediction Histogram Level {level} {mode} {g_name}", self.args['img_root'])
            
            mse_te = calculate_mse(pred, gt)
            
            if(mode == 'te'):
                print(f'Iter {iter}: Test MSE = {mse_te}')
            elif(mode[:2] == 'tr'):
                print(f'Iter {iter}: Train MSE = {mse_te}')
            
            
        return pred, iter_pred

    def extract_lbp_features(self, image, P=8, R=1, method="uniform"):
        """
        LBP histogram features.
        P: number of sampling points
        R: radius
        """
        hist_all = []; hist_all2 = [];hara = []; hog_feat = []
        P = 16
        R = 2
        for im in image:
            P = 16
            R = 2
            lbp = local_binary_pattern(np.squeeze(im), P=P, R=R, method=method)
            
            # For 'uniform', number of bins is P + 2
            n_bins = P + 2 if method == "uniform" else int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            hist_all.append(hist)
            P = 8
            R = 1
            lbp = local_binary_pattern(np.squeeze(im), P=P, R=R, method=method)
            
            # For 'uniform', number of bins is P + 2
            n_bins = P + 2 if method == "uniform" else int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            hist_all2.append(hist)
            haralick = mahotas.features.haralick((im[0]/ im.max() * 31).astype('uint8'))
            haralick = haralick.mean(axis = 0)
            hara.append(haralick)

            hog_features, _ = hog(
                im[0],
                orientations=9,
                pixels_per_cell=(11, 11),
                cells_per_block=(1, 1),
                visualize=True,
                block_norm='L2-Hys'
            )
            hog_feat.append(hog_features)
        return np.concatenate([hist_all,hist_all2, hara, hog_feat], axis = -1)

    def Patch_XGBoost_Train(self, X,pred, gt, hard = None):
        
        
        level = 'clf'
        X_sel = X; Y_sel = gt; pred_sel = pred
        idx = np.arange(0, len(X_sel.reshape(-1)), 1)
        self.pixelhop_haar[f'{level}'] = dict()
        #dc, var, sel = self.select_roi_im(X_sel, pred_sel,'clf' ,'tra')
        #grp,self.thresh[f'level_clf'] = self.energy_split_train(np.expand_dims(X_sel, -1),np.expand_dims(pred_sel,-1), 1) 
        #sel = grp[1]
        
        pred_sel1000, _ = rec_img(pred_sel, pred_sel.reshape(len(pred_sel),args['patch_size'], args['patch_size'] ), args['patch_size'], len(pred_sel), 30)

        pred_sel16 = view_as_windows(np.asarray(pred_sel1000), (1,16,16), (1, 12, 12))
        pred_sel16 = pred_sel16.reshape(Y_sel.shape)
        mse_im = []

        for im in range(len(pred_sel16)):
            mse_im.append(calculate_mse(Y_sel[im], pred_sel16[im]))
        
        #mse_av = np.mean(mse_im, axis = -1)
        mse_id = np.argsort(mse_im)[::-1]
        self.thresh[f'level'] = mse_im[mse_id[int(len(mse_im)/2)]]
        y_he = np.zeros(len(mse_id))
        y_he[mse_im>self.thresh[f'level']] = 1
        y_he = y_he.astype('uint8').tolist()

        hardp = np.where(np.asarray(y_he)==1)[0]
        easyp = np.where(np.asarray(y_he)==0)[0]
        X_all=  X_sel.copy().tolist()
        pred_sel_all = pred_sel16.copy().tolist()
        y_aug = y_he.copy()
        for aug in range(1,2):
            X_aug = self.transform_images_onetype(np.squeeze(X_sel), aug, X_sel)
            X_all.extend(np.expand_dims(X_aug,-1))
            pred_aug = self.transform_images_onetype(pred_sel16, aug, pred_sel16)
            pred_sel_all.extend(pred_aug)
            y_aug.extend(y_he)
        

        #X_all = np.concatenate([X_all[0], X_all[1], X_all[2]], axis = 0)
        #pred_sel_all = np.concatenate([pred_sel_all[0], pred_sel_all[1], pred_sel_all[2]], axis = 0)
        X_all = np.asarray(X_all)
        pred_sel_all = np.asarray(pred_sel_all)
     
        y_aug = np.asarray(y_aug)

        self.Train_PixelHop(level, X_all, np.expand_dims(X_all, -1),np.expand_dims(pred_sel_all, -1),'patches', 4, 5,idx , 'bin0', None )
        X_tr = self.Get_PixelHopFeat(level, X_all, 'patches', 0, 5, 'bin0')
        #X_tr = []
        X_tr = self.Add_SpatialFeat(X_tr, X_all, 'clf', 16, 'patches')

        self.Train_PixelHop(level, np.expand_dims(pred_sel_all, -1), np.expand_dims(X_all, -1),np.expand_dims(pred_sel_all, -1),'patches', 4, 5,idx , 'binp0',None )
        X_prev = self.Get_PixelHopFeat(level, np.expand_dims(pred_sel_all, -1), 'patches', 0, 5, 'binp0')
        X_prev = self.Add_SpatialFeat(X_prev, np.expand_dims(pred_sel_all, -1), 'clf', 5, 'patches')
        lbp = self.extract_lbp_features(X_all)
        
        feat = np.concatenate([X_tr.reshape(-1, X_tr.shape[-1]), X_prev.reshape(-1, X_prev.shape[-1]), lbp], axis = -1)
        #feat = X_tr.reshape(-1, X_tr.shape[-1])
        
        if(hard is not None):
            feat = feat[hard]
            Y_sel = Y_sel.reshape(-1)[hard]
        feat_idx = np.arange(0, feat.shape[-1], 1)
        Y_sel = Y_sel.reshape(-1)
        #res = Y_sel - pred.reshape(-1)
        Y_res = y_aug.copy()
        
        if(hard is None):
            g_name = 'clf1'
            self.rft_all['level_clf'] = dict()
            self.lnt_all[f'level_clf'] = dict()
            rem = np.where(Y_sel==0)[0]
            rem_sel = random.sample(rem.tolist(), int(0.5*len(rem)))
            down = np.concatenate([np.where(Y_sel==1)[0], rem_sel])
            self.dft_feat_selection(level, feat, Y_res,'bin', feat_idx, self.args['img_root'], 0.75, f'level_clf1' )
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin']]
            
        else: 
            g_name = 'clf2'
            self.dft_feat_selection(level, feat, Y_res,'bin1', feat_idx, self.args['img_root'], 0.5 , f'level_clf2' )
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin1']]
        print(f"Shape of features after RFT: {feat_1.shape}")
        
        
        if(True):
            
            self.Train_LNT(feat_1, Y_res, level, g_name)
            feat_lnt = self.Get_LNT(feat_1, level, g_name)
            #feat_tr = feat_lnt.copy()
            feat_tr = np.concatenate([feat_1, feat_lnt], axis = -1)
            
            """
            _ = self.generate_lnt_features(feat_1, Y_sel, 2, level, g_name)
            feat_lnt = self.Get_LNT_feat(feat_1, Y_sel, 2, level, g_name)
            """
            plot(feat_1, Y_sel, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
            #feat_tr = np.concatenate([feat_1, feat_lnt], axis = -1)
            feat_idx = np.arange(0, feat_tr.shape[-1], 1)
            #self.dft_feat_selection(level, feat_tr, Y_sel,f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 0.6, f'level_{level}LNT_{g_name}')
            #feat_tr = feat_tr[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            print(f"Shape of features after DFT: {feat_tr.shape}")
        else:
            feat_tr = feat_1
        data_idx = np.arange(0, len(feat_1), 1)
        X_train, X_val, y_train, y_val, data_tr, data_val = train_test_split(feat_tr, Y_res, data_idx, test_size=0.2,
                                                          random_state=42)
        if(hard is None):
            rem = np.where(y_train==0)[0]
            rem_sel = random.sample(rem.tolist(), int(0.5*len(rem)))
            down = np.concatenate([np.where(y_train==1)[0], rem_sel])
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            print(f'pos weight: {scale_pos_weight}')
            self.xgb_clf1 = train_clf(self.args, X_train, y_train, X_val, y_val, 1, pos_weight= scale_pos_weight, plot = True, savepath= os.path.join(args['img_root'], f'xgboost_clfbin.png'))
        else: 
            self.xgb_clf2 = train_clf(self.args, X_train, y_train, X_val, y_val, 2, plot = True, savepath= os.path.join(args['img_root'], f'xgboost_clfbin2.png'))
        if(hard is None):
            rounds=int(len(X_train)/1000000)+1
            pred_tr = []; pred_cs_tr = []
            for r in range(rounds):
                pred_tr.extend(self.xgb_clf1.predict(X_train[r*1000000:(r+1)*1000000]))
                pred_cs_tr.extend(self.xgb_clf1.predict_proba(X_train[r*1000000:(r+1)*1000000]))
            #pred_tr = self.xgb_clf.predict(X_train)
            rounds=int(len(X_val)/1000000)+1
            pred_val = []; pred_cs_val = []
            for r in range(rounds):
                pred_val.extend(self.xgb_clf1.predict(X_val[r*1000000:(r+1)*1000000]))
                pred_cs_val.extend(self.xgb_clf1.predict_proba(X_val[r*1000000:(r+1)*1000000]))
            #pred_val = self.xgb_clf.predict(X_val)
            
            pred_tr = np.asarray(pred_tr); pred_val = np.asarray(pred_val)
            pred_cs_tr = np.asarray(pred_cs_tr);pred_cs_val = np.asarray(pred_cs_val)
            acc_tr = accuracy_score(y_train, pred_tr)
            acc_val = accuracy_score(y_val, pred_val)
            print(f'Classifier Train Accuracy Round 1 : {acc_tr}, Val Accuracy : {acc_val}')

            #a = np.where(pred_cs[:,1]>0.2)[0]
            #b = np.where(pred_cs[:,1]<0.8)[0]
            #sel = np.intersect1d(a,b)
            
        else: 
            rounds=int(len(X_train)/1000000)+1
            pred_tr = []
            for r in range(rounds):
                pred_tr.extend(self.xgb_clf2.predict(X_train[r*1000000:(r+1)*1000000]))
            #pred_tr = self.xgb_clf.predict(X_train)
            rounds=int(len(X_val)/1000000)+1
            pred_val = []
            for r in range(rounds):
                pred_val.extend(self.xgb_clf2.predict(X_val[r*1000000:(r+1)*1000000]))
            acc_tr = accuracy_score(y_train, pred_tr)
            acc_val = accuracy_score(y_val, pred_val)
            print(f'Classifier Train Accuracy Round 2: {acc_tr}, Val Accuracy : {acc_val}')

        if(hard is None):
            return data_val[sel_val]

    def Patch_XGBoost_Pred(self, X,pred, gt, mode, hard = None):
        level = 'clf'
        #idx = np.arange(0, len(X.reshape(-1)), 1)
        #dc, var, sel = self.select_roi_im(X, pred,'clf' ,mode)
        #a = np.where(X.reshape(-1)>0.2)[0]
        #b = np.where(X.reshape(-1)<0.8)[0]
        #sel = np.intersect1d(a,b)
        #grp = self.energy_split_test(np.expand_dims(X, -1),np.expand_dims(pred,-1), 'clf',self.thresh[f'level_clf'] ,'orig') 
        #g = grp[1]

        if(mode == 'tr'):
            num_im = 30
        else: 
            num_im = 14
        pred_sel1000, _ = rec_img(pred, pred.reshape(len(pred),args['patch_size'], args['patch_size'] ), args['patch_size'], len(pred), num_im)

        pred_sel16 = view_as_windows(np.asarray(pred_sel1000), (1,16,16), (1, 12, 12))
        pred_sel16 = pred_sel16.reshape(gt.shape)
        mse_im = []

        for im in range(len(pred_sel16)):
            mse_im.append(calculate_mse(gt[im], pred_sel16[im]))
        
        mse_id = np.argsort(mse_im)[::-1]
        #self.thresh[f'level'] = mse_im[mse_id[int(len(mse_im)/2)]]
        y_he = np.zeros(len(mse_id))
        y_he[mse_im>self.thresh[f'level']] = 1
        y_he = y_he.astype('uint8').tolist()

        feat_tr = self.Get_PixelHopFeat(level, X,  'patches', 0,5, 'bin0')
        feat_tr = self.Add_SpatialFeat(feat_tr,  X, 'clf', 5, 'patches' )
        feat_prev = self.Get_PixelHopFeat('clf',  np.expand_dims(pred_sel16, -1), 'patches', 0,5,'binp0')
        feat_prev = self.Add_SpatialFeat(feat_prev, np.expand_dims(pred_sel16, -1), 'clf', 5, 'patches' )
        lbp = self.extract_lbp_features(X)
        feat = np.concatenate([feat_tr.reshape(-1, feat_tr.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1]), lbp], axis = -1)
        #feat = feat.reshape(-1, feat.shape[-1])
        #feat = feat_tr.reshape(-1, feat_tr.shape[-1])
        
        
        xgb_pred = []
        #p = pred.reshape(-1).copy()
        #res = gt.reshape(-1) - p
        Y_res = y_he.copy()
        
        #g_0 = grp[0]
        #p_g = p[g_0]
        #p_g[p_g<0.36] = 0; p_g[p_g>0.36] =1
        #p[g_0] = p_g
        #p[p<=0.2] = 0; p[p>=0.8] = 1
        #p[np.intersect1d(vr, dc_b)] = 0; p[np.intersect1d(vr, dc_n)] = 1
        if(hard is None):
            g_name = 'clf1'
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin']]
            if(True):
                #feat_lnt = self.Get_LNT_feat(feat_1, gt.reshape(-1), 2, level, g_name )
                feat_lnt = self.Get_LNT(feat_1, level, g_name)
                #feat_rft = feat_lnt.copy()
                feat_rft = np.concatenate([feat_1, feat_lnt], axis = -1)
                #feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            else: 
                feat_rft = feat_1
            rounds=int(len(feat_rft)/1000000)+1
            for r in range(rounds):
            
                xgb_pred.extend(self.xgb_clf1.predict(feat_rft[r*1000000:(r+1)*1000000]))
            p = np.asarray(xgb_pred)
        else: 
            g_name = 'clf2'
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin1']][hard]
            if(False):
                feat_lnt = self.Get_LNT_feat(feat_1, gt.reshape(-1), 2, level, g_name )
                #feat_lnt = self.Get_LNT(feat_1, level, g_name)
                #feat_rft = feat_lnt.copy()
                feat_rft = np.concatenate([feat_1, feat_lnt], axis = -1)
                feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            else: 
                feat_rft = feat_1
            rounds=int(len(feat_rft)/2000000)+1
            for r in range(rounds):
            
                xgb_pred.extend(self.xgb_clf2.predict(feat_rft[r*2000000:(r+1)*2000000]))
            p[hard] = np.asarray(xgb_pred)
        #pred = self.xgb_clf.predict(feat.reshape(-1, feat.shape[-1]))
        
      
        Y_res = np.asarray(Y_res)
        acc_te = accuracy_score(Y_res.reshape(-1), p)
        recall = recall_score(Y_res.reshape(-1), p)
        prec = precision_score(Y_res.reshape(-1), p)
        if(mode == 'te'):
            print(f'Test Classifier Accuracy : {acc_te}, {recall}, {prec}')
        else:
            print(f'Train Classifier Accuracy : {acc_te}, {recall}, {prec}')
        
        #pred = p.reshape(X.shape)
        if(hard is None):
            return np.asarray(np.where(p==1)[0])
        else: 
            return np.asarray(pred)
    
    def Bin_XGBoost_Train(self, X,pred, gt, hard = None):
        
        
        level = 'clf'
        X_sel = X; Y_sel = gt; pred_sel = pred
        idx = np.arange(0, len(X_sel.reshape(-1)), 1)
        self.pixelhop_prev['prevlevel_clf'] = dict()
        self.pixelhop_all[f'inplevel_clf'] = dict()
        a = np.where(pred.reshape(-1)>0.4)[0]
        b = np.where(pred.reshape(-1)<0.6)[0]
        c = np.intersect1d(a,b)
        #dc, var, sel = self.select_roi_im(X_sel, pred_sel,'clf' ,'tra')
        #grp,self.thresh[f'level_clf'] = self.energy_split_train(np.expand_dims(X_sel, -1),np.expand_dims(pred_sel,-1), 1) 
        #sel = grp[1]
        #self.Train_PixelHop(level, X_sel, np.expand_dims(X_sel, -1),np.expand_dims(pred_sel, -1),'inp', 4, 5,idx , 'bin0', None )
        #X_tr = self.Get_PixelHopFeat(level, X_sel, 'inp', 0, 5, 'bin0')
        #X_tr = []
        #X_tr = self.Add_SpatialFeat(X_tr, X_sel, 'clf', 7, 'inp')
        self.Train_PixelHop(level, X_sel, np.expand_dims(X_sel, -1),np.expand_dims(pred_sel, -1),'prev', 4, 11,idx , 'binp0',None )
        X_prev = self.Get_PixelHopFeat(level, X_sel, 'prev', 0, 11, 'binp0')
        X_prev, lbp = self.Add_SpatialFeat(X_prev, X_sel, 'clf', 11, 'prev', rot_idx = c)
        feat_laws = self.Add_LawsFeat(level, X_sel, 'tr')
        #level = 'init'
        #feat = np.concatenate([X_tr.reshape(-1, X_tr.shape[-1]), X_prev.reshape(-1, X_prev.shape[-1]), feat_laws.reshape(-1, feat_laws.shape[-1])], axis = -1)
        feat = np.concatenate([X_prev.reshape(-1, X_prev.shape[-1])[c], feat_laws.reshape(-1, feat_laws.shape[-1])[c],lbp ], axis = -1)
        #feat = X_prev.reshape(-1, X_prev.shape[-1])
        
        if(hard is not None):
            feat = feat[hard]
            Y_sel = Y_sel.reshape(-1)[hard]
        kernel = np.ones((5, 5), np.uint8)
        bdry = []
        for n in range(len(Y_sel)):
            bdry.append(cv2.morphologyEx(Y_sel[n], cv2.MORPH_GRADIENT, kernel))
        bdry = np.asarray(bdry)        #bdry = bdry.reshape(-1)
        bdry[bdry==1] = 2
        bdry[bdry==0] = 1
        feat_idx = np.arange(0, feat.shape[-1], 1)
        Y_sel = Y_sel.reshape(-1)
        res = Y_sel - pred.reshape(-1)
        Y_res = np.zeros(len(res))
        
        Y_res[abs(res)<=0.05] = 0
        Y_res[abs(res)>0.05] = 1

        if(hard is None):
            g_name = 'clf1'
            self.rft_all['level_clf'] = dict()
            self.lnt_all[f'level_clf'] = dict()
            rem = np.where(Y_sel==0)[0]
            rem_sel = random.sample(rem.tolist(), int(0.5*len(rem)))
            down = np.concatenate([np.where(Y_sel==1)[0], rem_sel])
            self.dft_feat_selection(level, feat, Y_sel[c],'bin', feat_idx, self.args['img_root'], 0.4, f'level_clf1' )
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin']]
            
        else: 
            g_name = 'clf2'
            self.dft_feat_selection(level, feat, Y_sel,'bin1', feat_idx, self.args['img_root'], 0.5 , f'level_clf2' )
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin1']]
        print(f"Shape of features after RFT: {feat_1.shape}")
        
        
        if(True):
            
            self.Train_LNT(feat_1, Y_sel[c], level, g_name)
            feat_lnt = self.Get_LNT(feat_1, level, g_name)
            #feat_tr = feat_lnt.copy()
            feat_tr = np.concatenate([feat_1, feat_lnt], axis = -1)
            
            
            plot(feat_1, Y_sel, feat_lnt,f'Level_{level}',  os.path.join(self.args['img_root'], 'LNT') )
            #feat_tr = np.concatenate([feat_1, feat_lnt], axis = -1)
            feat_idx = np.arange(0, feat_tr.shape[-1], 1)
            self.dft_feat_selection(level, feat_tr, Y_sel[c],f'after_lnt_{g_name}', feat_idx, self.args['img_root'], 0.75, f'level_{level}LNT_{g_name}')
            feat_tr = feat_tr[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            print(f"Shape of features after DFT: {feat_tr.shape}")
        else:
            feat_tr = feat_1
        data_idx = np.arange(0, len(feat_1), 1)
        X_train, X_val, y_train, y_val, data_tr, data_val = train_test_split(feat_tr, Y_sel[c], data_idx, test_size=0.2,
                                                          random_state=42)
        if(hard is None):
            rem = np.where(y_train==0)[0]
            rem_sel = random.sample(rem.tolist(), int(0.5*len(rem)))
            down = np.concatenate([np.where(y_train==1)[0], rem_sel])
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            print(f'pos weight: {scale_pos_weight}')
            self.xgb_clf1 = train_clf(self.args, X_train, y_train, X_val, y_val, 1, pos_weight = scale_pos_weight ,plot = True, savepath= os.path.join(args['img_root'], f'xgboost_clfbin.png'))
        else: 
            self.xgb_clf2 = train_clf(self.args, X_train, y_train, X_val, y_val, 2, plot = True, savepath= os.path.join(args['img_root'], f'xgboost_clfbin2.png'))
        if(hard is None):
            rounds=int(len(X_train)/500000)+1
            pred_tr = []; pred_cs_tr = []
            for r in range(rounds):
                pred_tr.extend(self.xgb_clf1.predict(X_train[r*500000:(r+1)*500000]))
                pred_cs_tr.extend(self.xgb_clf1.predict_proba(X_train[r*500000:(r+1)*500000]))
            #pred_tr = self.xgb_clf.predict(X_train)
            rounds=int(len(X_val)/500000)+1
            pred_val = []; pred_cs_val = []
            for r in range(rounds):
                pred_val.extend(self.xgb_clf1.predict(X_val[r*500000:(r+1)*500000]))
                pred_cs_val.extend(self.xgb_clf1.predict_proba(X_val[r*500000:(r+1)*500000]))
            #pred_val = self.xgb_clf.predict(X_val)
            
            pred_tr = np.asarray(pred_tr); pred_val = np.asarray(pred_val)
            pred_cs_tr = np.asarray(pred_cs_tr);pred_cs_val = np.asarray(pred_cs_val)
            acc_tr = accuracy_score(y_train, pred_tr)
            acc_val = accuracy_score(y_val, pred_val)
            print(f'Classifier Train Accuracy Round 1 : {acc_tr}, Val Accuracy : {acc_val}')

            #a = np.where(pred_cs[:,1]>0.2)[0]
            #b = np.where(pred_cs[:,1]<0.8)[0]
            #sel = np.intersect1d(a,b)
            
        else: 
            rounds=int(len(X_train)/1000000)+1
            pred_tr = []
            for r in range(rounds):
                pred_tr.extend(self.xgb_clf2.predict(X_train[r*1000000:(r+1)*1000000]))
            #pred_tr = self.xgb_clf.predict(X_train)
            rounds=int(len(X_val)/1000000)+1
            pred_val = []
            for r in range(rounds):
                pred_val.extend(self.xgb_clf2.predict(X_val[r*1000000:(r+1)*1000000]))
            acc_tr = accuracy_score(y_train, pred_tr)
            acc_val = accuracy_score(y_val, pred_val)
            print(f'Classifier Train Accuracy Round 2: {acc_tr}, Val Accuracy : {acc_val}')

       
        if(hard is None):
            return data_val[sel_val]

    def Bin_XGBoost_Pred(self, X,pred, mode, hard = None):
        level = 'clf'
        p = pred.reshape(-1).copy()
        a = np.where(p>0.4)[0]
        b = np.where(p<0.6)[0]
        c = np.intersect1d(a,b)
        #idx = np.arange(0, len(X.reshape(-1)), 1)
        #dc, var, sel = self.select_roi_im(X, pred,'clf' ,mode)
        #a = np.where(X.reshape(-1)>0.2)[0]
        #b = np.where(X.reshape(-1)<0.8)[0]
        #sel = np.intersect1d(a,b)
        #grp = self.energy_split_test(np.expand_dims(X, -1),np.expand_dims(pred,-1), 'clf',self.thresh[f'level_clf'] ,'orig') 
        #g = grp[1]
        #feat_tr = self.Get_PixelHopFeat(level, X,  'inp', 0,5, 'bin0')
        #feat_tr = self.Add_SpatialFeat(feat_tr,  X, 'clf', 7, 'inp' )
        feat_prev = self.Get_PixelHopFeat('clf',  X, 'prev', 0,11,'binp0')
        feat_prev, lbp = self.Add_SpatialFeat(feat_prev, X, 'clf', 11, 'prev',rot_idx=c )
        feat_laws = self.Add_LawsFeat('clf', X, 'te')

        #feat = np.concatenate([feat_tr.reshape(-1, feat_tr.shape[-1]), feat_prev.reshape(-1, feat_prev.shape[-1]),feat_laws.reshape(-1, feat_laws.shape[-1])], axis = -1)
        #feat = feat.reshape(-1, feat.shape[-1])
        #feat = feat_prev.reshape(-1, feat_prev.shape[-1])
        feat = np.concatenate([feat_prev.reshape(-1, feat_prev.shape[-1])[c], feat_laws.reshape(-1, feat_laws.shape[-1])[c], lbp], axis = -1)
        
        xgb_pred = []
        
        
        
        #g_0 = grp[0]
        #p_g = p[g_0]
        #p_g[p_g<0.36] = 0; p_g[p_g>0.36] =1
        #p[g_0] = p_g
        #p[p<=0.2] = 0; p[p>=0.8] = 1
        #p[np.intersect1d(vr, dc_b)] = 0; p[np.intersect1d(vr, dc_n)] = 1
        if(hard is None):
            g_name = 'clf1'
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin']]
            if(True):
                #feat_lnt = self.Get_LNT_feat(feat_1, gt.reshape(-1), 2, level, g_name )
                feat_lnt = self.Get_LNT(feat_1, level, g_name)
                #feat_rft = feat_lnt.copy()
                feat_rft = np.concatenate([feat_1, feat_lnt], axis = -1)
                feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            else: 
                feat_rft = feat_1
            rounds=int(len(feat_rft)/500000)+1
            for r in range(rounds):
            
                xgb_pred.extend(self.xgb_clf1.predict(feat_rft[r*500000:(r+1)*500000]))
            p[c] = np.asarray(xgb_pred)
        else: 
            g_name = 'clf2'
            feat_1 = feat[:, self.rft_all[f'level_{level}']['bin1']][hard]
            if(True):
                #feat_lnt = self.Get_LNT_feat(feat_1, gt.reshape(-1), 2, level, g_name )
                feat_lnt = self.Get_LNT(feat_1, level, g_name)
                #feat_rft = feat_lnt.copy()
                feat_rft = np.concatenate([feat_1, feat_lnt], axis = -1)
                feat_rft = feat_rft[:, self.rft_all[f'level_{level}'][f'after_lnt_{g_name}']]
            else: 
                feat_rft = feat_1
            rounds=int(len(feat_rft)/2000000)+1
            for r in range(rounds):
            
                xgb_pred.extend(self.xgb_clf2.predict(feat_rft[r*2000000:(r+1)*2000000]))
            p[hard] = np.asarray(xgb_pred)
        #pred = self.xgb_clf.predict(feat.reshape(-1, feat.shape[-1]))
        
        
        
        pred_te = p.reshape(pred.shape)
        if(hard is None):
            return np.asarray(pred_te)
        else: 
            return np.asarray(pred_te)
    
   
    def load_pred(self, level, mode):
        X0, Y = model.load_patches_te(mode, False, 4)

        if(level == 3):
            pred = self.get_init(4, X0,X0, Y,  mode)
        else: 
            pred = self.get_init(4, X0,X0, Y,  mode)
            for l in range(3, level-1, -1): 
                X_l, Y = model.load_patches_te(mode, False, l)
                pred = model.test_level(l, X_l, X_l, Y,  'tr' ,pred)
            pred_te = model.Bin_XGBoost_Pred(X_l[:,5:266-5,5:266-5], pred, mode)
        return pred

def local_lbp(im, ld):
    lbp_all = []
    for i in range(len(im)):
        lbp_all.append(local_binary_pattern(np.squeeze(block_reduce(im[i], (ld,ld,1), np.mean)), P=8, R=1, method='uniform'))

    return np.asarray(lbp_all)

def compute_metrics(pred_te, Y_test):
    
    gt_te, l1_te = rec_img(pred_te.reshape(len(Y_test),args['patch_size'], args['patch_size'] ), Y_test.reshape(len(Y_test),args['patch_size'], args['patch_size'] ), args['patch_size'], len(Y_test), 14)
    gt_te = np.asarray(gt_te)
    l1_te = np.asarray(l1_te)

    l1_b = binarize(l1_te, 0.5).astype('uint8')
    
    fin_aji = find_aji32(gt_te,l1_b)
    print(fin_aji)
    
if __name__ == "__main__":
    
    
    args = {'patch_size' : 256,
            'SaabArgs' : [{'num_AC_kernels':-1, 'needBias':False, 'cw': False}], 
            'concatArg' : {'func':Concat}, 
            'win' : [5,5,5,5], 
            'n_win': [9,7,5,3],
            'fov' : [11,9,7,5],
            'laws_fov': [9,7,5,3],
            'num_iter' : [1, 1, 1, 1], 
            'downsample_patches' : [1, 1, 1, 1], 
            'FS' : [[0.8,0.8, 0.4,0.8 ],[0.8,0.8, 0.6 ],[0.8, 0.8, 0.8],[0.8, 0.8, 0.4]],
            'modelroot' : #insert path to Models folder,
            'img_root' : #insert path to Img folder, 
            'guslmodel' : #insert path to GUSL_Models folder,
            'K_means' : False, 
            'lnt' : [True, True, True, True],
            'num_lnt' : 700,
            'eval_metric': 'rmse', 
            'clf_eval_metric' : 'logloss',
            'max_depth' : [4,3,5],
            'clf_max_depth': [2,2],
            'iclf_max_depth': [3,3], 
            'estimators' : [[300, 1000,500,800, 800], [600, 1500, 700], [500, 1500, 800], [500, 500, 600, 1500, 1000]],
            'estimators_2' : [400, 1700, 1600, 600],
            'clf_estimators' : [4000,500], 
            'iclf_estimators' : [3000,500],
            'lr' : 0.1,
            'clf_lr': 0.007,
            'iclf_lr': 0.01
            }
    
    model = Model(args)
    
    for l in range(3,-1,-1):
        model_paths = os.path.join(args['guslmodel'], f'level_{l}.pkl')
        if(os.path.exists(model_paths)):
            model.load_level(model_paths, l)
            print(f'Level {l} loaded successfully')
    model_paths = os.path.join(args['guslmodel'], f'level_clf.pkl')
    model.load_level_clf(model_paths, 'clf')
    print(f'Level clf loaded successfully')
    #if mode == 'tr' loads MoNuSeg training results
    #if mode == 'te' loads MoNuSeg testing results
    
    #Load data
    X_test3, Y_test= model.load_patches_te("te", False, 1)
    pred_final =  model.load_pred(1, 'te')
    compute_metrics(pred_final,Y_test )
    print("")