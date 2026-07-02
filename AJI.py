

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def AJI(pred_mask, gt_mask):
    # print('AJI Evaluation starts:')
    #print("Pred mask size:", pred_mask.shape)
    num_labels, pred_labels = cv.connectedComponents(
        pred_mask.astype(np.uint8))
    num_labels, gt_labels = cv.connectedComponents(
        gt_mask.astype(np.uint8))
    # pred_labels = pred_mask
    # gt_labels = gt_mask
    unique_pred_labels = np.unique(pred_labels)
    if (len(unique_pred_labels) == 1 and unique_pred_labels[0] == 1):
        pred_list = unique_pred_labels
    else:
        pred_list = np.unique(pred_labels)[1:]
    gt_list = np.unique(gt_labels)[1:]
    # print(pred_list)
    pred_list = [pred_list, np.zeros(len(pred_list))]
    pred_list = np.array(pred_list).transpose()
    # print(gt_list)
    i = gt_list.shape[0]

    overall_correct_count = 0
    union_pixel_count = 0
    for nuc_idx in gt_list:
        gt_corrd = np.where(gt_labels == nuc_idx)
        gt_corrd = np.array(gt_corrd).transpose()

        gt = np.zeros(gt_mask.shape)
        for x, y in gt_corrd:
            gt[x][y] = nuc_idx

        pred_match = gt*pred_labels

        if np.count_nonzero(pred_match) == 0:
            union_pixel_count += np.count_nonzero(gt)
        else:
            pred_nuc_idx = np.unique(pred_match)
            pred_nuc_idx = pred_nuc_idx[1:]
            JI = 0
            best_match = 0
            for j in range(len(pred_nuc_idx)):
                matched = np.zeros(pred_labels.shape)
                matched_coord = np.where(
                    pred_labels == pred_nuc_idx[j]/nuc_idx)
                matched_coord = np.array(matched_coord).transpose()
                for x, y in matched_coord:
                    matched[x][y] = pred_labels[x][y]

                NJI = np.count_nonzero(gt*matched) / \
                    np.count_nonzero(gt+matched)
                if NJI > JI:
                    best_match = pred_nuc_idx[j]/nuc_idx
                    JI = NJI

            pred_nuclei = np.zeros(pred_labels.shape)
            pred_nuclei_coord = np.where(pred_labels == best_match)
            pred_nuclei_coord = np.array(pred_nuclei_coord).transpose()
            for x, y in pred_nuclei_coord:
                pred_nuclei[x][y] = best_match
            # plt.subplot(2, 1, 1)
            # plt.imshow(pred_nuclei, 'gray')
            # plt.subplot(2, 1, 2)
            # plt.imshow(gt, 'gray')
            # plt.show()
            overall_correct_count += np.count_nonzero((gt*pred_nuclei))
            union_pixel_count += np.count_nonzero(gt+pred_nuclei)

            # print(pred_nuclei)
            index = np.where(pred_list == best_match)
            index = np.array(index).transpose()
            # print(pred_list)
            idx = index[0][0]

            pred_list[idx][1] = pred_list[idx][1] + 1

    unused_nuclei_index = np.where(pred_list[:, 1] == 0)
    unused_nuclei_index = np.array(unused_nuclei_index)
    for i in range(unused_nuclei_index.shape[0]):
        if len(pred_list[unused_nuclei_index[i]]) > 0:
            unused_nuclei_coord = np.where(
                pred_labels == pred_list[unused_nuclei_index[i][0]][0])
            unused_nuclei_coord = np.array(unused_nuclei_coord).transpose()
            unused_nuclei = np.zeros(pred_labels.shape)
            for x, y in unused_nuclei_coord:
                unused_nuclei[x][y] = pred_list[unused_nuclei_index[i][0]][0]

            union_pixel_count += np.count_nonzero(unused_nuclei)

    return overall_correct_count/union_pixel_count

def get_fast_aji(pred, true):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    #true_id_list = [0 , 1]
    #pred_id_list = [0 , 1]
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
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
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
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union

    print(overall_inter, overall_union)
    return aji_score


# # pred_mask = Image.open("pred_mask_unet.jpg")
# # gt_mask = Image.open("gt_mask_unet.png")
# pred_mask = Image.open("pred_mask.png")
# gt_mask = Image.open("GT_mask.png")
# pred_mask = np.array(pred_mask)
# gt_mask = np.array(gt_mask)
# # pred_mask = np.array([[0, 0, 1], [2, 0, 1], [0, 3, 0]])
# # gt_mask = np.array([[0, 0, 1], [2, 0, 1], [0, 3, 0]])

# aji = AJI(pred_mask, gt_mask)
# print(aji)
# # [0, 0, 1],
# # [2, 0, 1],
# # [0, 3, 0]
# https://github.com/ruchikaverma-iitg/MoNuSeg/blob/master/Aggregated_Jaccard_Index_v1_0.m
