import numpy as np
import cv2


def f1_score(pred, gt, show=False):
    pred_num_label, pred_label_map = cv2.connectedComponents(pred)
    true_label_num, gt_label_map = cv2.connectedComponents(gt)
    gt_num_label = np.unique(gt_label_map)
    pred_labels = np.unique(pred_label_map)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(gt_num_label)):
        if gt_num_label[i] == 0:
            continue
        gt_coord = np.where(gt_label_map == gt_num_label[i])
        pred_region = pred[gt_coord]
        num_exist = np.count_nonzero(pred_region)
        num_true_label = len(gt_coord[0])
        if num_exist >= 0.5 * num_true_label:
            true_positive += 1
        else:
            false_negative += 1
    for i in range(len(pred_labels)):
        if pred_labels[i] == 0:
            continue
        pred_coord = np.where(pred_label_map == pred_labels[i])
        gt_region = gt[pred_coord]
        num_exist = np.count_nonzero(gt_region)
        if num_exist < 0.5 * len(pred_coord[0]):
            false_positive += 1
    if(true_positive + false_positive!=0):
        precision = true_positive / (true_positive + false_positive)
    else:
        precision=0
    if(true_positive + false_negative!=0):
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0 
    if(precision+recall!=0):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1=0
    if(show==True):
        print("Number of false positives: {}".format(false_positive))
        print("Number of false negatives: {}".format(false_negative))
    return f1, precision, recall

# one nulcei in the ground, multiple nulcei in the pred mask at corresponding region
# if true positive: sum of nulcei pixel in the pred mask / nulcei pixel region in the ground > 0.5
#
