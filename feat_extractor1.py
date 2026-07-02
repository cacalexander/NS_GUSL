import os
import numpy as np 
import cv2

from HE_Norm import normalizeStaining
from skimage.measure import block_reduce
from data_patch_loader import image_patch_loader, generate_patches, image_patch_loader_test, generate_patches_new
from skimage import exposure
from lib_data_transformation import pqr_decomposition, minmax_normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.util import view_as_windows, view_as_blocks
from skimage.measure import block_reduce
import pickle
import gc
import time

from utils import load_model, save_model

def im_to_blocks_pad(patch_size, ROI,pad_width ):
    
    #pad_width+=12
    ROI=np.pad(ROI, ((pad_width+12,pad_width+12), (pad_width+12,pad_width+12),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    start_i = pad_width; start_j = pad_width
    per_row=int(h/patch_size)
    per_col=int(w/patch_size)
    """
    blocks_arr=np.zeros((per_row*per_col, patch_size + (2*pad_width), patch_size + (2*pad_width), c))
    for i in range(per_row):
        for j in range(per_col):
            
            blocks_arr[(i*per_row)+(j)]=ROI[((start_i)-pad_width):(start_i+patch_size+pad_width), ((start_j)-pad_width):start_j+patch_size +pad_width]
            start_j += patch_size
        start_i+= patch_size
        start_j= pad_width
    """
    blocks_arr = view_as_windows(ROI, (patch_size + (2*pad_width), patch_size + (2*pad_width), c), patch_size)
    blocks_arr = blocks_arr.reshape(-1, patch_size + (2*pad_width), patch_size + (2*pad_width), c)
    blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr

def im_to_blocks_padte(patch_size, ROI,pad_width, stride = None ):
    
    #pad_width+=12
    ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    start_i = pad_width; start_j = pad_width
    per_row=int(h/patch_size)
    per_col=int(w/patch_size)
    """
    blocks_arr=np.zeros((per_row*per_col, patch_size + (2*pad_width), patch_size + (2*pad_width), c))
    for i in range(per_row):
        for j in range(per_col):
            
            blocks_arr[(i*per_row)+(j)]=ROI[((start_i)-pad_width):(start_i+patch_size+pad_width), ((start_j)-pad_width):start_j+patch_size +pad_width]
            start_j += patch_size
        start_i+= patch_size
        start_j= pad_width
    """
    if(stride == None):
        stride = patch_size - 8
    
    blocks_arr = view_as_windows(ROI, (patch_size + (2*pad_width), patch_size + (2*pad_width), c),stride)
    blocks_arr = blocks_arr.reshape(-1, patch_size + (2*pad_width), patch_size + (2*pad_width), c)
    if(ROI.dtype== 'uint8'):
        blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr

def im_to_blocks_gtte(patch_size, ROI, stride = None):
    
    pad_width=0
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    per_row=int(h/patch_size)
    per_col=int(w/patch_size)
    blocks_arr=np.zeros((per_row*per_col, patch_size, patch_size, c))
    """
    for i in range(per_row):
        for j in range(per_col):
            
            blocks_arr[(i*per_row)+j]=ROI[(i*patch_size):(i+1)*patch_size, (j*patch_size):(j+1)*patch_size]
    """

    if(stride == None):
        stride = patch_size - 8
    blocks_arr = view_as_windows(ROI, (patch_size + (2*pad_width), patch_size + (2*pad_width), c), stride)
    blocks_arr = blocks_arr.reshape(-1, patch_size + (2*pad_width), patch_size + (2*pad_width), c)
    blocks_arr=blocks_arr.astype('uint8')
    
    #blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr

def im_to_blocks_padte16(patch_size, ROI,pad_width, stride = None ):
    
    #pad_width+=12
    ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    start_i = pad_width; start_j = pad_width
    per_row=int(h/patch_size)
    per_col=int(w/patch_size)
    """
    blocks_arr=np.zeros((per_row*per_col, patch_size + (2*pad_width), patch_size + (2*pad_width), c))
    for i in range(per_row):
        for j in range(per_col):
            
            blocks_arr[(i*per_row)+(j)]=ROI[((start_i)-pad_width):(start_i+patch_size+pad_width), ((start_j)-pad_width):start_j+patch_size +pad_width]
            start_j += patch_size
        start_i+= patch_size
        start_j= pad_width
    """
    if(stride == None):
        stride = patch_size - 4
    
    blocks_arr = view_as_windows(ROI, (patch_size + (2*pad_width), patch_size + (2*pad_width), c),stride)
    blocks_arr = blocks_arr.reshape(-1, patch_size + (2*pad_width), patch_size + (2*pad_width), c)
    if(ROI.dtype== 'uint8'):
        blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr

def im_to_blocks_gtte16(patch_size, ROI, stride = None):
    
    pad_width=0
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    per_row=int(h/patch_size)
    per_col=int(w/patch_size)
    blocks_arr=np.zeros((per_row*per_col, patch_size, patch_size, c))
    """
    for i in range(per_row):
        for j in range(per_col):
            
            blocks_arr[(i*per_row)+j]=ROI[(i*patch_size):(i+1)*patch_size, (j*patch_size):(j+1)*patch_size]
    """

    if(stride == None):
        stride = patch_size - 4
    blocks_arr = view_as_windows(ROI, (patch_size + (2*pad_width), patch_size + (2*pad_width), c), stride)
    blocks_arr = blocks_arr.reshape(-1, patch_size + (2*pad_width), patch_size + (2*pad_width), c)
    blocks_arr=blocks_arr.astype('uint8')
    
    #blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr

def im_to_blocks_te(patch_size, ROI ):
    
    pad_width=12
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    per_row=int(h/patch_size)
    per_col=int(w/patch_size)
    blocks_arr=np.zeros((per_row*per_col, patch_size, patch_size, c))
    for i in range(per_row):
        for j in range(per_col):
            
            blocks_arr[(i*per_row)+j]=ROI[(i*patch_size):(i+1)*patch_size, (j*patch_size):(j+1)*patch_size]
    
    
    #blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr


def im_to_blocks(patch_size, ROI):
    
    stride=64
    #pad_width = int(stride/2)
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width),(0,0) ), mode='reflect')
    #ROI=np.pad(ROI, ((pad_width,pad_width), (pad_width,pad_width)) , mode='symmetric')
    h,w,c=ROI.shape
    nrow=int(h/stride)
    ncol=int(w/stride)
    #blocks_arr=np.zeros((per_row*per_col, patch_size, patch_size, c))
    blocks_arr = []
    for i in range(nrow-int(patch_size/stride)+1):
        for j in range(ncol-int(patch_size/stride)+1):
            h_start = i *stride
            w_start = j * stride
            blocks_arr.append(ROI[(h_start):(h_start+patch_size), (w_start):(w_start + patch_size)])
    
    blocks_arr = np.asarray(blocks_arr)
    blocks_arr=blocks_arr.astype('uint8')
    return blocks_arr

def channels(ROI):

    pqr_input=ROI
    
    enhanced_image_pqr, color_pca_model, bias_term = pqr_decomposition(pqr_input, forma='P', color_pca=None, bias=0)
    norm_enhanced = minmax_normalize(enhanced_image_pqr, single=True)
    #sc = MinMaxScaler().fit(enhanced_image_pqr.reshape(len(enhanced_image_pqr), -1))
    #norm_enhanced = sc.transform(enhanced_image_pqr.reshape(len(enhanced_image_pqr), -1)).reshape(len(enhanced_image_pqr), enhanced_image_pqr.shape[1],enhanced_image_pqr.shape[2],1)
    #norm_enhanced = enhanced_image_pqr.copy()
    return (norm_enhanced), color_pca_model

def channels_test(ROI, bias_term, color_pca_model, sc = None):
    
    pqr_input=ROI
    new_data, _, _ = pqr_decomposition(
        pqr_input, forma='P', color_pca=color_pca_model, bias=0)
    norm_enhanced = minmax_normalize(new_data, single=True)
    #norm_enhanced = sc.transform(new_data.reshape(len(new_data), -1)).reshape(len(new_data), new_data.shape[1],new_data.shape[2],1)

    #norm_enhanced = new_data.copy()
    return (norm_enhanced)

def Shrink(X, shrinkArg):
    #---- max pooling----
    pool = shrinkArg['pool']
    # TODO: fill in the rest of max pooling
    X = block_reduce(X, (1, pool, pool, 1), np.max)
    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # TODO: fill in the rest of neighborhood construction
    # numpy padding
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    X = view_as_windows(X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1]))

    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
# Example callback function for how to concate features from different hops

def Shrink(X, shrinkArg):
    #---- max pooling----
    pool = shrinkArg['pool']
    # TODO: fill in the rest of max pooling
    X = block_reduce(X, (1, pool, pool, 1), np.max)
    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # TODO: fill in the rest of neighborhood construction
    # numpy padding
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    X = view_as_windows(X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1]))

    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
# Example callback function for how to concate features from different hops

def Shrink_patch(X, shrinkArg):
    #---- max pooling----
    pool = shrinkArg['pool']
    # TODO: fill in the rest of max pooling
    X = block_reduce(X, (1, pool, pool, 1), np.max)
    #---- neighborhood construction
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # TODO: fill in the rest of neighborhood construction
    # numpy padding
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    X = view_as_windows(X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1]))
    X = np.squeeze(X)
    X = X.reshape(-1,1,win,win)
    return X
# Example callback function for how to concate features from different hops

def Concat(X, concatArg):
    return X

def get_feat(X, p, num_layers=4, output_all=False):
    if output_all == True:
        output = []
        output.append(p.transform_singleHop(X, layer=0))
    else:
        output = p.transform_singleHop(X, layer=0)

    if num_layers > 1:
        for i in range(num_layers - 1):
            if output_all == True:
                tmp = p.transform_singleHop(output[-1], layer=i+1)
                output.append(tmp)
            else:
                output = p.transform_singleHop(output, layer=i+1)

    return output

def save_feat(X, model, ch, level, BS=10, mode='tr', saveroot='./', num_layers=4, output_all=True):
    N_Full = X.shape[0]
    if BS>0:
        for i in range(int(np.ceil(N_Full/BS))):
            tmp_output = get_feat(X[i*BS:(i+1)*BS], model, num_layers=num_layers, output_all=output_all)
            NUM_HOPS = len(tmp_output)
            for k in range(NUM_HOPS):
                with open(saveroot + mode + str(ch)+'_output_stage'+str(k+1)+'_'+str(i)+'.npy', 'wb') as f:
                    np.save(f, tmp_output[k])
            tmp_output = None
            gc.collect()

        for k in range(num_layers):
            output = []
            for i in range(int(np.ceil(N_Full/BS))):
                with open(saveroot + mode + '_output_stage'+str(k+1)+'_'+str(i)+'.npy', 'rb') as f:
                    output.append(np.load(f))
                with open(saveroot + mode + '_output_stage'+str(k+1)+'_'+str(i)+'.npy', 'wb') as f:
                    np.save(f, np.array([0]))
            output = np.concatenate(output,axis=0)
            print('Shape of features for layer {}: {}'.format(k+1, output.shape))
            with open(saveroot + mode + str(ch)+'_feature_Hop'+str(k+1)+'.npy', 'wb') as f:
                np.save(f, output)
            output = None
            gc.collect()
    else:
        output = get_feat(X, model,num_layers=num_layers, output_all=output_all)
        NUM_HOPS = len(output)
        for k in range(NUM_HOPS):
            print('Shape of features for layer {}: {}'.format(k+1, output[k].shape))
            with open(saveroot + mode + str(ch)+ '_feature_Hop'+str(level)+'.npy', 'wb') as f:
                np.save(f, output[k])

def add_neighbor_feat(feature, window, stride):
    res = feature
    if window != 1:
        N, H, W, C = feature.shape
        padding = window // 2
        #res = np.pad(feature, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='reflect')
        res = view_as_windows(res, (1, window, window, 1), step=(1, 1, 1, 1))
        if(stride!=1):
            res = res[:,:,:,:,:,[0,int((window-1)/2),window-1],:,:]
            res = res[:,:,:,:,:,:,[0,int((window-1)/2),window-1],:]
        res = res.reshape(N, res.shape[1], res.shape[2], -1)
    return res

def neighbor_patches(feature, win):
    res = feature
    N,h,w = feature.shape
    

def load_patches(mode, aug, level, filter_size, patch_size, save_path, bias_term = None):
    if(mode == 'tr'):
        num = 30
    elif(mode == 'te'):
        num = 14
    
    X = []; Y= []
    if(level>0):
        pad_width = 2**(level)
    else:
        pad_width = 0
    
    
    for ind in range(num):
        print('------------------------') 
        print(ind)  
        # load image
        [image, gt, basename] = image_patch_loader(ind)
        print(basename)
        
        _, H, E = normalizeStaining(img = image)
        
        gt = gt.astype('uint8') ; gt[gt == 255] = 1 ;
        y=np.asarray(gt).reshape(gt.shape[0]*gt.shape[1])
        r = 1024.0 / 1000
        dim = (1024, int(1000 * r))    
        patch_r=cv2.resize(H, dim, interpolation=cv2.INTER_LANCZOS4)
        
        patches_r = im_to_blocks_pad(256, patch_r,pad_width)
        X.extend(patches_r)
       
        gt_r=cv2.resize(gt, dim, interpolation=cv2.INTER_LANCZOS4)
        
        Y.extend(im_to_blocks_te(patch_size, np.expand_dims(gt_r,-1)))
    
    X = np.asarray(X)
    Y = np.squeeze(np.asarray(Y).astype('uint8'))
    if(mode == "tr"):
        sc = StandardScaler().fit(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3))
        X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
        X_trainP, color_pca_model, bias_term=channels(X_s)    
        save_model(sc, os.path.join(save_path, f"scX_{level}.pkl"))
        save_model(color_pca_model, os.path.join(save_path, f"pcaX_{level}.pkl"))
    elif(mode == "te"):
        sc = load_model(os.path.join(save_path, f"scX_{level}.pkl"))
        color_pca_model = load_model(os.path.join(save_path, f"pcaX_{level}.pkl"))
        X_s = sc.transform(X.reshape(len(X), (patch_size + (2*pad_width))*(patch_size + (2*pad_width))*3)).reshape(len(X), patch_size + (2*pad_width),patch_size + (2*pad_width),3)
        X_trainP = channels_test(X_s, bias_term,color_pca_model)
    X_P=np.asarray(X_trainP)
    
    
    return X_P, Y, bias_term


    
    
    
    