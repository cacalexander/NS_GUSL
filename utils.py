import os
import pickle
import math
import numpy as np
import cv2
from skimage.util import view_as_windows
from skimage.measure import block_reduce
from scipy.ndimage import zoom
from skimage import segmentation, filters
from skimage.measure import label, regionprops
from skimage import morphology
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from AJI import AJI, get_fast_aji
from F1Score import f1_score
from mean_shift_post_processing_v2 import recover_img_patch, recover_img_patcho, remove_noise, hole_fill, postProc, postProc2
from matplotlib.ticker import MaxNLocator
#from rft import RFT
from models.research.deeplab.evaluation import panoptic_quality
from models.research import slim
from scipy import ndimage
import tensorflow as tf
import random
from skimage.filters import sobel


from skimage.measure import regionprops, label

import mahotas

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_model(model, path):
    with open(path, 'wb') as file:
        # Dump the dictionary into the file
        pickle.dump(model, file)


def load_model(path):
    with open(path, 'rb') as file:
        # Load the dictionary back from the pickle file.
        model = pickle.load(file)
    return model

def Shrink(X, pool,win, stride, pad):
    # TODO: fill in the rest of max pooling
    X = block_reduce(X, (1, pool, pool, 1), np.max)
    #---- neighborhood construction
    
    
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    X = view_as_windows(X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1]))

    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def Shrink_patch(X, pool,win, stride, pad):
    # TODO: fill in the rest of max pooling
    if(pool != 1):
        X = block_reduce(X, (1, pool, pool, 1), np.max)
    #---- neighborhood construction
    
    
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    X = view_as_windows(X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1]))
    X = np.squeeze(X)
    X = X.reshape(-1,1,win,win)
    return X

def duplicate_1d(pred, win_size, level):
    n_row = int(2**(8-level+1)/win_size)
    n = int(len(pred)/(n_row*n_row))
    pred_2d = []
    offset = np.ones((win_size,win_size))
    idx = 0
    for i in range(n):
        dup_pred = np.zeros((2**(8-level+1),2**(8-level+1)))
        for j in range(n_row):
            for k in range(n_row):
                dup_pred[j*win_size:(j+1)*win_size, k*win_size:(k+1)*win_size] = offset - pred[idx]
                idx+=1
        pred_2d.append(dup_pred)
    
    pred_2d = np.asarray(pred_2d)
    return pred_2d

def add_gaussian_noise(img, mean=0, std=0.5):
    noise = np.random.normal(mean, std, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def add_to_hue(images, range=(-15,15)):
    """Perturbe the hue of input images."""
    img = images.copy()  # aleju input batch as default (always=1 in our case)
    hue = np.random.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return np.clip(ret,0,1)

def add_to_saturation(images, range=(-0.2, 0.2)):
    """Perturbe the saturation of input images."""
    img = images.copy() # aleju input batch as default (always=1 in our case)
    value = 1 + np.random.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, range=(-26, 26)):
    """Perturbe the contrast of input images."""
    img = images.copy()  # aleju input batch as default (always=1 in our case)
    value = np.random.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, range=(0.75, 1.25)):
    """Perturbe the brightness of input images."""
    img = images.copy()  # aleju input batch as default (always=1 in our case)
    value = np.random.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]

def gaussian_blur(images, max_ksize=3):
    """Apply Gaussian blur to input images."""
    img = images.copy()  # aleju input batch as default (always=1 in our case)
    ksize = np.random.randint(1, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    #ret = ret.astype(np.uint8)
    return ret
    return np.clip(ret,0,1)


####
def median_blur(images, max_ksize=3):
    """Apply median blur to input images."""
    img = images.copy()  # aleju input batch as default (always=1 in our case)
    ksize = np.random.randint(1, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    #ret = ret.astype(np.uint8)
    return np.clip(ret,0,1)

def duplicate(pred):
    N, h, w = pred.shape
    dup_pred_all = []
    for n in range(N):
        dup_pred = np.zeros((h*2,w*2))
        for i in range(h):
            for j in range(w):
                dup_pred[i*2:(i+1)*2, j*2:(j+1)*2] = pred[n,i,j]
        dup_pred_all.append(dup_pred)        
    
    dup_pred_all = np.asarray(dup_pred_all)        
    return dup_pred_all

def binarize(data, threshold=0.5):
    shape = data.shape
    res = data.copy()
    res = res.reshape(shape[0], shape[1], shape[2])
    res[res>=threshold] = 1
    res[res<threshold] = 0
    return res

def thresh(data, low, high):
    
    a = np.where(data<low)[0]
    b = np.where(data>high)[0]
    sel = np.concatenate([a,b])

    return sel

def truncation(data):
    data[data<0] = 0
    data[data>1] = 1
    return data

def calculate_mse(X, Y):
    mse = np.sum(np.square(np.abs(Y - X))) / len(X.reshape(-1))
    return mse

def compute_full_img_aji(pred_masks, full_gt):
    pred_masks = np.array(pred_masks)

    #aji = AJI(pred_masks, full_gt)
    aji = get_fast_aji(pred_masks, full_gt)

    return aji

def sigmoid(x):
    # Use clipping to prevent overflow in exp
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def semantic_to_instance(semantic_map, void_label=0):
    """
    Converts a semantic map to an instance map using connected components.

    Args:
        semantic_map: HxW numpy array of class IDs.
        void_label: class ID to ignore (optional).

    Returns:
        instance_map: HxW numpy array of instance IDs (starting from 1 per class).
    """
    instance_map = np.zeros_like(semantic_map, dtype=np.int32)
    next_instance_id = 1

    classes = np.unique(semantic_map)
    for c in classes:
        if c == void_label:
            continue

        # binary mask for this class
        mask = (semantic_map == c)
        labeled, num_features = ndimage.label(mask)
        
        # assign global instance IDs
        labeled[labeled > 0] += next_instance_id - 1
        instance_map[mask] = labeled[mask]
        next_instance_id += num_features

    return instance_map

def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def comp_pq(gt, pred):
    
    pq = panoptic_quality.PanopticQuality(
        num_categories=2,
        ignored_label=0,
        max_instances_per_category=2000,
        offset= 10000)
    pq.compare_and_accumulate(gt, label(gt), pred,
                              label(pred))
    
    pqs = pq.detailed_results()['All']['pq']
    sq = pq.detailed_results()['All']['sq']
    rq = pq.detailed_results()['All']['rq']
    print("")

    return pqs, sq, rq

def find_aji(gt, pred,patch_size, num_patches, num_im):
    
    img_arr=[];gt_arr=[]
    blocks_per_im=int(num_patches/num_im)
    if(num_im ==14):
        im_size = 1000
    else: 
        im_size = 512
    for img in range(num_im):
        if(im_size == 1000):
            image=recover_img_patcho(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-8, img_size=im_size, isRGB=False)
            gt_img=recover_img_patcho(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-8, img_size=im_size, isRGB=False)
        else: 
            image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=im_size, isRGB=False)
            gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=im_size, isRGB=False)
        #image_r = image[12:1012, 12:1012]
        #gt_r = gt_img[12:1012, 12:1012]
        #image_r=cv2.resize(image.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #gt_r=cv2.resize(gt_img.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        image[image>0] = 1
        img_arr.append(image)
        gt_arr.append(gt_img)
    aji_pred = []; f1_pred=[];dice_pred = []; pq_pred = []; precision = []; recall = []
    #temp_gt = binarize(np.asarray(gt_arr), 0.5)
    #temp_pred=binarize(np.asarray(img_arr), thre)
        
    for i in range(len(gt_arr)):
        aji = compute_full_img_aji(np.squeeze(img_arr[i].astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
           
        f1 = f1_score(np.squeeze(img_arr[i].astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
        dice = dice_coef(np.squeeze(gt_arr[i].astype('uint8')),np.squeeze(img_arr[i].astype('uint8')) )
        pq = comp_pq(np.squeeze(gt_arr[i].astype('uint8')),np.squeeze(img_arr[i].astype('uint8')) )
        aji_pred.append(aji)
        f1_pred.append(f1[0])
        precision.append(f1[1])
        recall.append(f1[2])
        dice_pred.append(dice)
        pq_pred.append(pq)

    print(aji_pred)
    fin_aji=np.mean(aji_pred)
    fin_f1 = np.mean(f1_pred)
    fin_prec = np.mean(precision)
    fin_rec = np.mean(recall)
    fin_dice = np.mean(dice_pred)
    fin_pq = np.mean(pq_pred)
    

    return fin_aji, fin_f1, fin_prec, fin_rec, fin_dice, fin_pq

def find_aji32(gt_arr, img_arr):
    
    
    aji_pred = []; f1_pred=[];dice_pred = []; pq_pred = []; precision = []; recall = []; rq_pred = []; sq_pred = []
    #temp_gt = binarize(np.asarray(gt_arr), 0.5)
    #temp_pred=binarize(np.asarray(img_arr), thre)
        
    for i in range(len(gt_arr)):
        aji = compute_full_img_aji(label(np.squeeze(img_arr[i].astype('uint8'))), label(np.squeeze(gt_arr[i].astype('uint8'))))
           
        f1 = f1_score(np.squeeze(img_arr[i].astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
        dice = dice_coef(np.squeeze(gt_arr[i].astype('uint8')),np.squeeze(img_arr[i].astype('uint8')) )
        pq, sq, rq = comp_pq((np.squeeze(gt_arr[i].astype('uint8'))),(np.squeeze(img_arr[i].astype('uint8'))) )
        aji_pred.append(aji)
        f1_pred.append(f1[0])
        precision.append(f1[1])
        recall.append(f1[2])
        dice_pred.append(dice)
        pq_pred.append(pq)
        rq_pred.append(rq)
        sq_pred.append(sq)

    print(aji_pred)
    fin_aji=np.mean(aji_pred)
    fin_f1 = np.mean(f1_pred)
    fin_prec = np.mean(precision)
    fin_rec = np.mean(recall)
    fin_dice = np.mean(dice_pred)
    fin_pq = np.mean(pq_pred)
    fin_rq = np.mean(rq_pred)
    fin_sq = np.mean(sq_pred)
    

    return fin_aji, fin_f1, fin_prec, fin_rec, fin_dice, fin_pq, fin_rq, fin_sq

def augment(mask):

    aug = np.flip(mask, 1)

    return aug

def extract_patch(image, centroid, patch_size):
    h, w = patch_size
    
    cy = centroid[0]+7
    cx = centroid[1]+7
    x1 = int(cx - w // 2)
    y1 = int(cy - h // 2)
    x2 = x1 + w
    y2 = y1 + h

    # Handle borders (clip to image bounds)
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)
    x2_clipped = min(image.shape[1], x2)
    y2_clipped = min(image.shape[0], y2)

    patch = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    return patch

def recover_fn(pred, mask, orig, mode): 
    pc = pred.copy()
    if(mode=='tr'):
        pc[pc>0.75] = 0
        pc[pc<0.25] = 0
        sel_p = pc!=0
    else: 
        pc[pc>0.75] = 0
        pc[pc<0.25] = 0
        sel_p = pc!=0
    boundary_mask = morphology.binary_opening(sel_p, morphology.disk(1))
    boundary_mask = morphology.binary_closing(boundary_mask, morphology.disk(2))
    connected = label(boundary_mask, connectivity = 2)
    regions = regionprops(connected)
    cand_area = [];
    cand_convarea = [];
    cand_centroids = [];
    cand_solidity = [];
    cand_labels = [];
    for r in regions:
        cand_area.append(r.area)
        cand_convarea.append(r.convex_area)
        cand_centroids.append(r.centroid)
        cand_solidity.append(r.solidity)
        cand_labels.append(r.label)
    cand_area = np.asarray(cand_area)
    cand_convarea = np.asarray(cand_convarea)
    cand_centroids = np.asarray(cand_centroids)
    cand_solidity = np.asarray(cand_solidity)
    cand_labels = np.asarray(cand_labels)

    #solidity = np.asarray(cand_area)/np.asarray(cand_convarea)
    saab_ps = []
    sel = np.where(cand_solidity>0.5)[0]
    #print("Len of sel", len(sel))
    sel_inst = np.where(cand_area[sel]>100)[0]
    #print(cand_area[sel[sel_inst]])
    #print("Len of sel_inst", len(sel_inst))
    sel_i = sel[sel_inst]
    val = np.max(connected)
    sel_r = []
    
    for i in sel:
        if(cand_area[i]<1200 and  cand_area[i]>100):
            #connected[connected==cand_labels[i]] = val+1
            sel_r.append(i+1)
    #connected[connected!=val+1] = 0
    #connected[connected==val+1] = 1
    
    patches = []
    reg_features = []
    y = []
    image_padded = np.pad(orig, ((7, 7), (7, 7), (0,0)), mode='reflect')
    for r in regions:
        if r.label in sel_r:
            
            minr, minc, maxr, maxc = r.bbox

            

            saab_patch = extract_patch(image_padded, r.centroid, (11,11))
            saab_ps.append(saab_patch)
            if(minr-5>=0 and minc-5>=0 and maxr+5<=999 and maxc+5<=999):
                patch = orig[minr-5:maxr+5, minc-5:maxc+5]
                pc_n = pred[minr-5:maxr+5, minc-5:maxc+5]
            else: 
                patch = orig[minr:maxr, minc:maxc]
                pc_n = pred[minr:maxr, minc:maxc]
            
            #patch = orig[minr:maxr, minc:maxc]
            pc_n = pred[minr:maxr, minc:maxc]
            patch_s = [np.mean(patch[:,:,0]),np.mean(patch[:,:,1]), np.mean(patch[:,:,2]), np.max(patch[:,:,0]),np.max(patch[:,:,1]),np.max(patch[:,:,2])  ]
            pc_s = [np.mean(pc_n), np.max(pc_n) ]
            mask_n = mask[minr:maxr, minc:maxc]
            mask_n = mask_n.astype('uint8')
            
            conn = connected[minr:maxr, minc:maxc].copy()
            conn[conn == r.label] = 1
            mask_n[conn != 1] = 0
            grad_mag = (sobel(patch)*255).astype('uint8')
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            #gray[conn!=1] = 0
            grad_mag_gray = cv2.cvtColor(grad_mag, cv2.COLOR_RGB2GRAY)
            #patch[conn!=1] = 0
            feat = []
            
            try:
                feat.append(mahotas.features.haralick((patch[:,:,0] / patch[:,:,0].max() * 31).astype('uint8')))
                feat.append(mahotas.features.haralick((patch[:,:,1] / patch[:,:,1].max() * 31).astype('uint8')))
                feat.append(mahotas.features.haralick((patch[:,:,2] / patch[:,:,2].max() * 31).astype('uint8')))
                feat.append(mahotas.features.haralick((grad_mag_gray/ grad_mag_gray.max() * 31).astype('uint8')))
                
                
            except: 
                print("Error")
            feat[0] = feat[0].mean(axis = 0)
            feat[1] = feat[1].mean(axis = 0)
            feat[2] = feat[2].mean(axis = 0)
            feat[3] = feat[3].mean(axis = 0)
            
            lbp = mahotas.features.lbp(gray, 2, 8)
            n_bins = 10
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            hist_gray, _ = np.histogram(gray.ravel(), bins=n_bins,range=(0, 255), density=True)
            feat = np.concatenate([feat[0],feat[1], feat[2], feat[3], hist,hist_gray, patch_s, pc_s])
            patches.append([r, patch, mask_n])

            reg_features.append(feat)
            y.append(np.max(mask_n))

            """
            if(np.max(mask_n) == 0):
                patch_aug = augment(patch)
                gray_aug = augment(gray)
                grad_img_gray_aug = augment(grad_mag_gray)
                feat = []
                feat.append(mahotas.features.haralick((patch_aug[:,:,0] / patch_aug[:,:,0].max() * 31).astype('uint8')))
                feat.append(mahotas.features.haralick((patch_aug[:,:,1] / patch_aug[:,:,1].max() * 31).astype('uint8')))
                feat.append(mahotas.features.haralick((patch_aug[:,:,2] / patch_aug[:,:,2].max() * 31).astype('uint8')))
                feat.append(mahotas.features.haralick((grad_img_gray_aug/ grad_img_gray_aug.max() * 31).astype('uint8')))
                feat[0] = feat[0].mean(axis = 0)
                feat[1] = feat[1].mean(axis = 0)
                feat[2] = feat[2].mean(axis = 0)
                feat[3] = feat[3].mean(axis = 0)
                lbp = mahotas.features.lbp(gray_aug, 2, 8)
                n_bins = 10
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                feat = np.concatenate([feat[0],feat[1], feat[2], feat[3], hist, patch_s, pc_s])
                patches.append([r, patch_aug, mask_n])

                reg_features.append(feat)
                y.append(np.max(mask_n))
            """


    return patches, np.asarray(reg_features), np.asarray(y), connected, saab_ps

def update_pred(pred, orig, mask):
    im = []
    r_all = []
    feat_all = []
    y_all = []
    cc = []
    imgs = []
    if(len(pred)==30):
        mode = 'tr'
    else: 
        mode = 'te'
    for i in range(len(pred)):
        n = pred[i].copy()
        r, feat, y, connected, patches = recover_fn(n, orig[i], mask[i], mode)
        r_all.append(r)
        feat_all.extend(feat)
        y_all.extend(y)
        cc.append(connected)
        imgs.extend(patches)
    return r_all, np.asarray(feat_all), np.asarray(y_all), np.asarray(cc), np.asarray(imgs)

def find_aji_m(gt, pred,patch_size, num_patches, num_im):
    
    img_arr=[];gt_arr=[]
    blocks_per_im=int(num_patches/num_im)
    
    for img in range(num_im):
        image=recover_img_patcho(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-8, img_size=1000, isRGB=False)
        gt_img=recover_img_patcho(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-8, img_size=1000, isRGB=False)
        #image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #image_r = image[12:1012, 12:1012]
        #gt_r = gt_img[12:1012, 12:1012]
        #image_r=cv2.resize(image.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #gt_r=cv2.resize(gt_img.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #image = remove_noise(image)
        #image = postProc2(image.astype('uint8'), gt_img.astype('uint8'))
        image = hole_fill(image*255)
        image[image==255] = 1
        img_arr.append(image)
        gt_arr.append(gt_img)
    aji_pred = []; f1_pred=[]
    #temp_gt = binarize(np.asarray(gt_arr), 0.5)
    #temp_pred=binarize(np.asarray(img_arr), thre)
        
    for i in range(len(gt_arr)):
        aji = compute_full_img_aji(np.squeeze(img_arr[i].astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
           
        f1 = f1_score(np.squeeze(img_arr[i].astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
        aji_pred.append(aji)
        f1_pred.append(f1)
    print(aji_pred)
    fin_aji=np.mean(aji_pred)
    fin_f1 = np.mean(f1_pred)
    
    return fin_aji, fin_f1

def find_aji_w(gt_arr, img_arr,patch_size, num_patches, num_im):
    
    
    aji_pred = []; f1_pred=[]
    #temp_gt = binarize(np.asarray(gt_arr), 0.5)
    #temp_pred=binarize(np.asarray(img_arr), thre)
        
    for i in range(len(gt_arr)):
        image = np.asarray(img_arr[i])
        image = remove_noise(image)
        aji = compute_full_img_aji(np.squeeze(image.astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
           
        f1 = f1_score(np.squeeze(image.astype('uint8')), np.squeeze(gt_arr[i].astype('uint8')))
        aji_pred.append(aji)
        f1_pred.append(f1)

    print(aji_pred)
    fin_aji=np.mean(aji_pred)
    fin_f1 = np.mean(f1_pred)
    
    return fin_aji, fin_f1

# 2021.01.07
# evaluation metrics

def rec_img(gt, pred,patch_size, num_patches, num_im):
    img_arr=[];gt_arr=[]
    blocks_per_im=int(num_patches/num_im)
    for img in range(num_im):
        image=recover_img_patcho(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-8, img_size=1000, isRGB=False)
        gt_img=recover_img_patcho(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-8, img_size=1000, isRGB=False)
        #image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #image_r = image[12:1012, 12:1012]
        #gt_r = gt_img[12:1012, 12:1012]
        #image_r=cv2.resize(image.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #gt_r=cv2.resize(gt_img.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        img_arr.append(image)
        gt_arr.append(gt_img)
        
    return img_arr, gt_arr

def rec_img512(gt, pred,patch_size, num_patches, num_im):
    img_arr=[];gt_arr=[]
    blocks_per_im=int(num_patches/num_im)
    for img in range(num_im):
        image=recover_img_patcho(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size, img_size=512, isRGB=False)
        gt_img=recover_img_patcho(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size, img_size=512, isRGB=False)
        #image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #image_r = image[12:1012, 12:1012]
        #gt_r = gt_img[12:1012, 12:1012]
        #image_r=cv2.resize(image.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #gt_r=cv2.resize(gt_img.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        img_arr.append(image)
        gt_arr.append(gt_img)
        
    return img_arr, gt_arr

def rec_img32(gt, pred,patch_size, num_patches, num_im, im_size ):
    img_arr=[];gt_arr=[]
    blocks_per_im=int(num_patches/num_im)
    for img in range(num_im):
        image=recover_img_patcho(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-4, img_size=im_size, isRGB=False)
        gt_img=recover_img_patcho(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, stride = patch_size-4, img_size=im_size, isRGB=False)
        #image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #image_r = image[12:1012, 12:1012]
        #gt_r = gt_img[12:1012, 12:1012]
        #image_r=cv2.resize(image.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #gt_r=cv2.resize(gt_img.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        img_arr.append(image)
        gt_arr.append(gt_img)
        
    return img_arr, gt_arr

def rec_imgo(gt, pred,patch_size, num_patches, num_im):
    img_arr=[];gt_arr=[]
    blocks_per_im=int(num_patches/num_im)
    for img in range(num_im):
        image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size, img_size=1000, isRGB=False)
        gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=1000, isRGB=False)
        #image=recover_img_patch(pred[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #gt_img=recover_img_patch(gt[(img*blocks_per_im):((img+1)*blocks_per_im)], patch_size,  img_size=int(math.sqrt(blocks_per_im))*patch_size, isRGB=False)
        #image_r = image[12:1012, 12:1012]
        #gt_r = gt_img[12:1012, 12:1012]
        #image_r=cv2.resize(image.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        #gt_r=cv2.resize(gt_img.astype('uint8'), (1000,1000), interpolation=cv2.INTER_LINEAR)
        img_arr.append(image)
        gt_arr.append(gt_img)
        
    return img_arr, gt_arr
    

def meanIoU(pred, gt, num_cls = 3, smooth=0.001):
    iou = np.zeros((pred.shape[0],num_cls))
    for c in range(num_cls):
        # for n in range(miou.size):
        pred_tmp = (pred==c)
        gt_tmp = (gt==c)
        intersection = np.sum(np.logical_and(pred_tmp, gt_tmp),axis=(1,2))
        union = np.sum(np.logical_or(pred_tmp, gt_tmp),axis=(1,2))
        iou[:,c] = (intersection + smooth) / (union + smooth)
    miou = np.mean(iou)
    return miou

def meanIoU_cls(pred, gt, num_cls = 3, smooth=0.001):
    iou = np.zeros((pred.shape[0],num_cls))
    for c in range(num_cls):
        # for n in range(miou.size):
        pred_tmp = (pred==c)
        gt_tmp = (gt==c)
        intersection = np.sum(np.logical_and(pred_tmp, gt_tmp),axis=(1,2))
        union = np.sum(np.logical_or(pred_tmp, gt_tmp),axis=(1,2))
        iou[:,c] = (intersection + smooth) / (union + smooth)
    miou = np.mean(iou,axis=0)
    return miou



def dice_coef(y_true, y_pred):
    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')
    intersection = np.dot(y_true.reshape(1,-1),y_pred.reshape(-1,1)).squeeze()
    smooth = 1e-9
    return np.round((2. * intersection.squeeze()) / (np.sum(y_true) + np.sum(y_pred) + smooth),decimals=3)

def dice_multi(y_true, y_pred, num_cls=3):
    dice=0
    for c in range(num_cls):
        dice += dice_coef(y_true==c, y_pred==c)
    return dice/num_cls # taking average

    
def dice_cls(y_true, y_pred, num_cls=3):
    dice = []
    for c in range(num_cls):
        dice.append(dice_coef(y_true==c, y_pred==c))
    return dice 

def dice_cls_2(y_true, y_pred, num_cls=3):
    dice = []
    for c in range(num_cls):
        if c==0: # background --> change to foreground
            dice.append(dice_coef(y_true>c, y_pred>c))
        else:
            dice.append(dice_coef(y_true==c, y_pred==c))
    return dice 

def dice_cls_3(y_true, y_pred, num_cls=2):
    dice = []
    dice.append(dice_coef(y_true==1, y_pred==1))
    return dice

def dice_cls_2D(y_true, y_pred, num_cls=3):
    dice = []
    for c in range(num_cls):
        if c==0: # background --> change to foreground
            dice.append(dice_coef(y_true>c, y_pred>c))
        else:
            dice.append(dice_coef(y_true==c, y_pred==c))
    return np.array(dice)

def plot_train_val_rank(dft, val_dft, path=None):
    training_rank = [dft.dim_rank[dim] for dim in range(dft.dim)]
    validation_rank = [val_dft.dim_rank[dim] for dim in range(val_dft.dim)]
    plt.figure(figsize=(8, 6))
    plt.scatter(training_rank, validation_rank, color='blue', alpha=1)

    plt.title('Feature Rank in Training vs. Validation')
    plt.xlabel('Training Rank')
    plt.ylabel('Validation Rank')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()
    
def lnt_kernel(x_train, y_train, y_tr, svd=None, num=600, if_plot=False):
    if svd is None:
        lda = LinearRegression().fit(x_train, y_tr)
        T = lda.coef_.transpose()
        #print(T.shape)
        svd = TruncatedSVD(n_components=T.shape[1]-1, random_state=42).fit(T)
        # print(np.argsort(svd.explained_variance_ratio_)[::-1])
        svd = svd.transform(T)[:, np.argsort(svd.explained_variance_ratio_)[::-1]]
        #print(T.shape)

    # Test
    kernel = svd[:, :num]
    x_tr_lda = x_train @ kernel
    #print("done Test", x_tr_lda.shape)

    """
    if if_plot:
        # DFT check
        x_tr = np.concatenate((x_tr_lda, x_train), axis=-1)
        print(x_tr.shape)
        _, dft_loss, _ = FEAT.feature_selection(x_tr, y_train, FStype='DFT_entropy', thrs=1.0, B=16)
        idx = np.argsort(dft_loss)

        # plot
        fig = plt.figure(figsize=(4, 10))

        # whole feat
        # plt.subplot(321)
        plt.plot([i for i in range(len(dft_loss))], np.sort(dft_loss))
        pick = [i for i in range(x_tr_lda.shape[1])]
        ind = np.where(np.isin(idx, pick))[0]
        plt.scatter(ind, dft_loss[idx[ind]], label='new', marker='.', color='orange')
        pick = [i for i in range(x_tr_lda.shape[1], len(dft_loss))]
        ind = np.where(np.isin(idx, pick))[0]
        plt.scatter(ind, dft_loss[idx[ind]], label='original', marker='.', color='green')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    """
    return svd[:, :num], x_tr_lda

def select_comb(gt):
    #For LNT
    num = len(gt)
    gt_comb = np.zeros((num, 2))
    gt_comb[:, 0] = 1 - gt
    gt_comb[:, 1] = gt
    return gt_comb

def random_contrast(image, min_factor=0.8, max_factor=0.99):
    factor = np.random.uniform(min_factor, max_factor)
    mean = np.mean(image)
    image = (image - mean) * factor + mean
    return image
    return np.clip(image, 0, 1)

def adjust_saturation_random(image, min_factor=0.5, max_factor=1.5):
    factor = np.random.uniform(min_factor, max_factor)

    img = image.astype(np.float32)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)  # <-- FIX

    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(out, 0, 1)

def apply_clahe_gray(image):
    clip = np.random.uniform(1.0, 4.0)
    grid = random.choice([(4,4), (8,8), (16,16)])
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return np.clip(clahe.apply(image),0,1)


def random_brightness_shift(image, max_shift=0.2):
    shift = np.random.uniform(-max_shift, max_shift)
    image = image + shift
    #image = np.clip(image, 0, 1)
    return image

def random_gamma(image, gamma_range=(0.8,1.2)):
    gamma = np.random.uniform(*gamma_range)
    return image ** gamma

def upscale_pred(pred, target_size, order="mean"):
    N, H, W = pred.shape
    ratio = target_size[0] / H
    #logger.info(f'Start upscale pred to {target_size} using order {order}')

    if order == "mean":
        repeat = int(ratio)
        resized = np.repeat(np.repeat(pred, repeat, axis=1), repeat, axis=2)

    elif order == "lanczos":
        scale = (1, ratio, ratio)
        resized = zoom(pred, scale, order=3)

    #logger.info('End Resize')
    return resized

def upsample(pred, target_size):
    N, H, W = pred.shape
    kernel = [[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]
    kernel = np.asarray(kernel)
    out = np.zeros((N, target_size+2, target_size+2))
    for n in range(N):
        startX = 0
        startY = 0
        for row in range(0, H):
            for col in range(0, W):
                out[n, startX:startX+3, startY:startY+3] += (kernel * pred[n, row, col])
                startY = startY + 2 
            startY = 0
            startX = startX + 2


    return out[:,1:target_size+1,1:target_size+1]


def plot(x, y, pred, name, saveroot):
        # RFT
        x_all = np.concatenate((pred, x), axis=-1)
        print('Shape of all features {}'.format(x_all.shape))
        feat_select = RFT(name='LNT')
        rft_loss = feat_select.fit(x_all, y)
        idx = np.argsort(rft_loss)

        # plot
        plt.plot([i for i in range(len(rft_loss))], np.sort(rft_loss))
        pick = [i for i in range(pred.shape[1])]
        ind = np.where(np.isin(idx, pick))[0]
        plt.scatter(ind, rft_loss[idx[ind]], label='new', marker='.', color='orange')
        pick = [i for i in range(pred.shape[1], len(rft_loss))]
        ind = np.where(np.isin(idx, pick))[0]
        plt.scatter(ind, rft_loss[idx[ind]], label='original', marker='.', color='green')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xlabel('Feature index')
        plt.ylabel('Training loss')
        plt.title('RFT loss curve')
        plt.tight_layout() 
        if name:
            plt.savefig(saveroot + 'RFT_curve_LNT_' + str(name) + '.png')
        else:
            plt.savefig(saveroot + 'RFT_curve_LNT.png')
        plt.close()

def plot_residue(res, name, saveroot):
    #res is an array of gt and residue predictions [gt, pred]
    import seaborn as sns
    plt.cla();plt.clf();
    for p in range(len(res)):
        if(p==0):
            sns.kdeplot(res[p], label = 'GT')
        else: 
            sns.kdeplot(res[p], label = 'Prediction')
    
    plt.title(str(name))
    plt.legend()
    plt.savefig(saveroot + str(name) + '.png')
    plt.cla();plt.clf();

    return

def plot_scatter(label, pred, saveroot, name):
    plt.cla();plt.clf();
    plt.scatter(label, pred)
    plt.title("GT vs Predictions")
    plt.savefig(saveroot + str(name) +'.png')

def pca_cal(X: np.ndarray):
    cov = X.transpose() @ X
    eva, eve = np.linalg.eigh(cov)
    inds = eva.argsort()[::-1]
    eva = eva[inds]
    kernels = eve.transpose()[inds]
    return kernels, eva / (X.shape[0] - 1)

def energy_cal(X):
    assert (len(X.shape) == 2), "Input must be a 2D array!"
    #energy = []
    #for s in range(len(X)):
    #    energy.append(np.sum(np.square(X[s])))
    energy = np.sum(np.square(X), axis = 1)
    energy = (energy - np.min(energy))/(np.max(energy) - np.min(energy))
    return energy

def entropy_from_probs(probs):
    entropy = 0.0
    for p in probs:
        if p > 0:  # avoid log(0)
            entropy -= p * math.log2(p)
    return entropy

def im_mask(im, gt):

    colored_mask = np.zeros_like(im)
    colored_mask[:, :, 0] = 255  # Red channel

    # Blend image and mask
    alpha = 0.5
    mask_bool = gt > 0
    blended = im.copy()
    #mask_3d = np.stack([mask_bool] * 3, axis=-1)

    #blended[mask_3d] = (alpha * colored_mask[mask_3d] + (1 - alpha) * im[mask_3d]).astype(np.uint8)

    blended = im.copy()
    blended[mask_bool] = cv2.addWeighted(im[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0)

    return blended

def overlay_prediction_on_label(label, pred, alpha=0.5):
    """
    Overlays a 2D prediction mask over a 2D ground truth label.
    
    Parameters:
        label (np.ndarray): Ground truth mask (2D array).
        pred (np.ndarray): Predicted mask (2D array).
        alpha (float): Opacity of the prediction overlay.
    """
    assert label.shape == pred.shape, "Label and prediction must be the same shape"
    
    # Create RGB images
    overlay = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.float32)

    # Ground truth in green
    overlay[label == 1] = [0, 1, 0]

    # Prediction in red
    overlay[pred == 1] = [1, 0, 0]

    # Where both match (true positives), set to yellow
    overlay[(label == 1) & (pred == 1)] = [1, 1, 0]

    plt.imshow(overlay)
    plt.title("Overlay: Green=False Negative, Red=False Positive, Yellow=TP")
    plt.axis('off')
    plt.savefig("Overlay")
    
def get_edges(mask):
    """
    Extracts edges from a binary nuclei mask using OpenCV.
    
    Parameters:
        mask (np.ndarray): 2D binary mask (0: background, 1: nuclei)

    Returns:
        edge_mask (np.ndarray): Binary edge mask
    """
    # Ensure binary mask is uint8
    #mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    #contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on an empty mask
    #edge_mask = np.zeros_like(mask_uint8)
    #cv2.drawContours(edge_mask, contours, -1, color=1, thickness=1)
    
    edge_mask = segmentation.find_boundaries(mask, mode='inner').astype(np.uint8)
    
    return edge_mask


def get_edge_morph(binary_mask):
    mask = (binary_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    return gradient * 255

def error_map(pred, gt):
    """
    pred: predicted binary mask (0/1)
    gt: ground truth binary mask (0/1)

    returns: RGB image highlighting errors
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # True Positive (correct foreground) → Green
    out[(pred == 1) & (gt == 1)] = [0, 255, 0]

    # True Negative (correct background) → Black (optional)
    out[(pred == 0) & (gt == 0)] = [0, 0, 0]

    # False Positive → Red
    out[(pred == 1) & (gt == 0)] = [255, 0, 0]

    # False Negative → Blue
    out[(pred == 0) & (gt == 1)] = [0, 0, 255]

    return out