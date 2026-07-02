from numpy.core.fromnumeric import shape
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from skimage.util.shape import view_as_windows
from skimage import color
from skimage.morphology import convex_hull_image, remove_small_objects
from scipy.spatial import ConvexHull
from operator import add
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.python.keras.backend import zeros
#from data_loader import data_loader
from AJI import AJI
from skimage import exposure
#from Code.lib_data_transformation import pqr_decomposition
from sklearn.cluster import MeanShift, estimate_bandwidth
# https: // www.programmersought.com/article/7581250465/
# image = Image.open("./img_patches/pred_mask/img1_0_0.png")
# image = np.array(image)
import cv2

from sklearn.cluster import KMeans
from data_patch_loader import generate_patches


def generate_pqr(image):
    # image: N, H, W, C
    tr_pqr, color_pqr_model, bias_term = pqr_decomposition(
        image, forma="PQR", color_pca=None, bias=None)
    # tr_pqr = cv2.normalize(tr_pqr, None, 0, 255,
    #                        cv2.NORM_MINMAX).astype(np.uint8)

    return tr_pqr


def generate_lab(image):
    lab = color.rgb2lab(
        image, illuminant='D65', observer='2')
    L, A, B = cv2.split(lab)
    L = cv2.normalize(
        L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    A = cv2.normalize(
        A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    B = cv2.normalize(
        B, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return [L, A, B]


def create_L_channel_patch_set(image, patch_size=64):
    X_train = []
    new_imgs = view_as_windows(
        image, (patch_size, patch_size, 3), (patch_size//2, patch_size//2, 3))
    # create patches for original images.
    for patch in new_imgs:
        X_train.append(patch)
    enhanced_patch = []
    X_train = np.array(X_train).reshape(-1, patch_size, patch_size, 3)
    for pidx in range(X_train.shape[0]):
        enhanced_patch.append(exposure.equalize_adapthist(X_train[pidx]))
    enhanced_patch = np.array(enhanced_patch)

    L_channel_set = []
    for pidx in range(enhanced_patch.shape[0]):
        lab = color.rgb2lab(
            enhanced_patch[pidx], illuminant='D65', observer='2')
        L, A, B = cv2.split(lab)
        L = cv2.normalize(
            L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        A = cv2.normalize(
            A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        B = cv2.normalize(
            B, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        L_channel_set.append(L)
    L_channel_set = np.array(L_channel_set).astype(np.uint8)
    return L_channel_set


def otsu(img):
    # Otsu's thresholding
    img_shape = img.shape
    img = img.squeeze()
    ret2, th2 = cv2.threshold(
        (img), 0, 255, cv2.THRESH_OTSU)
    # print(X_train.shape)
    # plt.subplot(1, 4, 1)

    # plt.subplot(1, 4, 2)
    # plt.imshow(img, "gray")
    # plt.title('P channel patch')
    # plt.show()
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    # plt.subplot(1, 4, 3)
    # plt.imshow(blur, "gray")
    # plt.title('Patch after Gaussian filter')

    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plt.subplot(1, 4, 4)
    # plt.imshow(th3, "gray")
    # plt.title('Binary map using Otsu')
    # plt.show()
    th3 = th3.reshape(img_shape).astype(np.uint8)

    # th2 = th2.reshape(img_shape)

    return th3


def create_otsu_set_for_patches(L_channel_set):
    otsu_set = []
    for pidx in range(L_channel_set.shape[0]):
        otsu_set.append(otsu(L_channel_set[pidx]))
    otsu_set = np.array(otsu_set)
    otsu_set = np.expand_dims(otsu_set, axis=-1)
    return otsu_set

def remove_noise(img_arr):
    img_nr = []

    for c in range(len(img_arr)):
        im = remove_small_objects(np.asarray(img_arr[c], bool), min_size = 5, connectivity = 2)
        img_nr.append(im.astype('uint8'))

    return np.asarray(img_nr)


def recover_img_patch(tiles, patch_size, img_size=1000, isRGB=False):
    patch_width = patch_size

    if (isRGB):
        img = np.zeros((img_size, img_size, 3))
    else:
        img = np.zeros((img_size, img_size))

    # print(img.shape)
    row = np.sqrt(len(tiles)).astype(np.int32)
    col = row

    count = 0
    for x in range(0, row):
        for y in range(0, col):
            if x == 0 and y == 0:
                startX = 0
                endX = patch_width
                startY = 0
                endY = patch_width
            else:
                # overlapping area replace previous area
                startX = x*(patch_width)
                endX = startX + (patch_width)
                startY = y*(patch_width)
                endY = startY + (patch_width)
            # print(startX, startY)
            # print(endX, endY)
            # print(tiles[count].shape)
            # print(img.shape)
            if (isRGB): 
                img[startX:endX, startY:endY, :] = tiles[count]
            else:
                img[startX:endX, startY:endY] = tiles[count]
                
            count = count + 1
    #print(count)
    return img

def recover_img_patcho(tiles, patch_size, stride, img_size=1000, isRGB=False):
    patch_width = patch_size

    if (isRGB):
        img = np.zeros((img_size, img_size, 3))
        cov = np.zeros((img_size, img_size, 3))
    else:
        img = np.zeros((img_size, img_size))
        cov = np.zeros((img_size, img_size))

    # print(img.shape)
    row = np.sqrt(len(tiles)).astype(np.int32)
    col = row

    count = 0
    for x in range(0, row):
        for y in range(0, col):
            if x == 0 and y == 0:
                startX = 0
                endX = patch_width
                startY = 0
                endY = patch_width
            else:
                # overlapping area replace previous area
                startX = x*(stride)
                endX = startX + (patch_width)
                startY = y*(stride)
                endY = startY + (patch_width)
            # print(startX, startY)
            # print(endX, endY)
            # print(tiles[count].shape)
            # print(img.shape)
            if (isRGB): 
                region_mask = cov[startX:endX, startY:endY]
                if not region_mask.any():
                    img[startX:endX, startY:endY, :] = tiles[count]
                else:
                    # Elementwise max
                    np.mean(img[startX:endX, startY:endY, :], tiles[count], out=img[startX:endX, startY:endY, :])
                region_mask[...] = True
            else:
                """
                region_mask = cov[startX:endX, startY:endY]
                if not region_mask.any():
                    img[startX:endX, startY:endY] = tiles[count]
                else:
                    # Elementwise max
                    np.mean(img[startX:endX, startY:endY], tiles[count], out=img[startX:endX, startY:endY])
                region_mask[...] = True
                """
                tile = tiles[count].astype(np.float32)
                img[startX:endX, startY:endY] += tile
                cov[startX:endX, startY:endY] += 1
                
                
            count = count + 1
    dtype = tiles[0].dtype
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(img, cov)
        result[np.isnan(result)] = 0

    # Cast back to original dtype if needed
    if np.issubdtype(dtype, np.integer):
        result = np.clip(result, 0, np.iinfo(dtype).max)
        result[result>0.5] =1
        result[result<=0.5] = 0
        result = result.astype(dtype)
    else:
        result = result.astype(dtype)
    #print(count)
    return result


def morph_operations(binary_map_patch):
    # open white holes for predicted mask
    kernel = cv2.getStructuringElement(1, (3, 3))
    morphed_patches = np.zeros(binary_map_patch.shape)
    for patch_map_idx in range(binary_map_patch.shape[0]):
        opening = cv2.morphologyEx(
            binary_map_patch[patch_map_idx], cv2.MORPH_OPEN, kernel)
        # opening = np.expand_dims(opening, axis=-1)
        morphed_patches[patch_map_idx] = opening
    return morphed_patches


def hole_fill(pred_mask):
    shape = pred_mask.shape
    pred_mask_pad = np.zeros((shape[0]+2, shape[1]+2)).astype(np.uint8)
    coord = np.where(pred_mask == 255)
    # pred_mask_pad[coord+1] = 255
    for i in range(len(coord)):
        for j in range(len(coord[i])):
            coord[i][j] += 1
    pred_mask_pad[coord] = 255
    #
    pred_mask = pred_mask_pad
    # Copy the thresholded image.
    im_floodfill = pred_mask.copy()
    # Mask used to flood filling.
    h, w = pred_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = pred_mask | im_floodfill_inv
    # # Display images.
    # cv2.imshow("Thresholded Image", pred_mask)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    coord = np.where(im_out == 255)
    for i in range(len(coord)):
        for j in range(len(coord[i])):
            coord[i][j] -= 1
    mask_wo_hole = np.zeros(shape).astype(np.uint8)
    mask_wo_hole[coord] = 255
    return mask_wo_hole


def mean_shift_to_binary(image, gt, mean_large_patch_luminance):
    # patch size =64 x 64
    patch = image
    originShape = np.uint32([image.shape[0], image.shape[1]])
    print(originShape)
    # 512 x 512 images resize to 64 x 64 to do meanshift

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg = np.reshape(patch, [-1, 3])

    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.105, n_samples=70)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # Performing meanshift on flatImg
    ms.fit(flatImg)

    flatImg = np.reshape(patch, [-1, 3])
    labels = ms.predict(flatImg)

    n_clusters_ = len(ms.cluster_centers_)
    print("num of clusters:", n_clusters_)
    # (r,g,b) vectors corresponding to the different clusters after meanshift
    # labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    luminance = 0.2126*cluster_centers[:, 0] + 0.7152 * \
        cluster_centers[:, 1] + 0.0722 * cluster_centers[:, 2]
    # mean_lum = np.mean(luminance)
    binary_map = np.zeros(labels.shape)
    for i in range(len(labels)):
        if luminance[labels[i]] < mean_large_patch_luminance and luminance[labels[i]] < mean_large_patch_luminance - 0.1:
            binary_map[i] = 255
        else:
            binary_map[i] = 0
    # pred = generate_pred_mask(luminance, labels)

    # Finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    # Displaying segmented image by resizing image back to 512 x 512
    segmentedImg = np.reshape(
        binary_map, (patch.shape[0], patch.shape[1])).astype(np.uint8)
    segmentedImg[segmentedImg >= 128] = 255
    segmentedImg[segmentedImg < 128] = 0

    labels = np.reshape(labels, (patch.shape[0], patch.shape[1]))

    # add a connected component check largest nuclei size to decide enlarge or not.
    nLabels, segmented_labels = cv2.connectedComponents(segmentedImg)
    largestNuclei = 0
    if (nLabels != 1):
        for i in range(1, nLabels):
            largestNuclei = max(largestNuclei, len(
                np.where(segmented_labels == i)[0]))

    print("largest nuclei size:", largestNuclei,
          " total:", segmented_labels.shape[0]**2)

    # check if some nuclei is large enough to better look for details
    is_zoom_in = False
    if largestNuclei / (segmented_labels.shape[0]**2) > 0.08:
        is_zoom_in = True

    return segmentedImg, ms, is_zoom_in


def zoom_in_patches(patch, gt, ms, patch_width, pred_mask, mean_large_patch_luminance):
    """
        patch: rgb color patch with size of 128 x 128
        gt: ground truth
        ms: Mean shift model
        patch_width: desired output size

        small_patch: 64 x 64 patch size, size 9 in totoal 
        new_img: recovered 128 x 128 -> 64 x 64 patch pred mask
    """
    # add a connected component check largest nuclei size to decide enlarge or not.
    new_pred = []
    # zoom in original patch
    # 128 -> 64
    small_patches, gt_small_patch = generate_patches(
        patch, gt, patch_width=patch_width // 2, patch_height=patch_width // 2, if_rgb=True)
    small_patches = np.array(small_patches)
    new_small_patches = []
    for i in range(small_patches.shape[0]):
        #
        flat_zoom_in_patch = np.reshape(small_patches[i], [-1, 3])
        zoom_in_labels = ms.predict(flat_zoom_in_patch)
        num_label = np.unique(zoom_in_labels)
        unique_labels = len(num_label)
        print("num of clusters:", unique_labels)
        zoom_in_cluster_centers = ms.cluster_centers_
        luminance = 0.2126*zoom_in_cluster_centers[:, 0] + 0.7152 * \
            zoom_in_cluster_centers[:, 1] + 0.0722 * \
            zoom_in_cluster_centers[:, 2]

        print("luminance:", luminance, mean_large_patch_luminance)
        zoom_in_pred = np.zeros(zoom_in_labels.shape)
        for i in range(len(zoom_in_labels)):

            if luminance[zoom_in_labels[i]] < mean_large_patch_luminance and luminance[zoom_in_labels[i]] < mean_large_patch_luminance - 0.1:
                zoom_in_pred[i] = 255
            else:
                zoom_in_pred[i] = 0

        zoom_in_pred = np.array(
            zoom_in_pred).reshape(patch_width // 2, patch_width // 2).astype(np.uint8)
        new_pred.append(zoom_in_pred)

    new_pred = np.array(new_pred)
    new_img = recover_img_patch(
        new_pred, patch_width // 2, img_size=patch.shape[0])
    # filter_new_img = cv2.medianBlur(new_img.astype(np.uint8), 3)

    kernel = cv2.getStructuringElement(1, (5, 5))
    filter_new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)

    filter_new_img = cv2.resize(
        filter_new_img, (patch_width, patch_width), interpolation=cv2.INTER_LANCZOS4)

    filter_new_img[filter_new_img >= 128] = 255
    filter_new_img[filter_new_img < 128] = 0

    # plt.subplot(4, 1, 1)
    # plt.imshow(patch)
    # plt.title("original image patch")
    # plt.subplot(4, 1, 2)
    # plt.imshow(pred_mask, 'gray')
    # plt.title("Zoom out pred mask")
    # plt.subplot(4, 1, 3)
    # plt.imshow(new_img, 'gray')
    # plt.title("Zoom in pred mask")
    # plt.subplot(4, 1, 4)
    # plt.imshow(filter_new_img, 'gray')
    # plt.title("Resized to 64 pred mask")

    # plt.show()

    return filter_new_img

# main method call from upper level

def postProc(pred, GT):

    connectBaysWidth = 2
    connectivity = 4

    #(numLabels_GT, _, _, _) = cv2.connectedComponentsWithStats(GT.astype('uint8'), connectivity, cv2.CV_32S)

    #imshow(pred, cmap='gray') ; plt.show()

    # 1 - Refinement
    filtered = cv2.medianBlur(pred,3) ; filtered = cv2.medianBlur(filtered,3) ; filtered = cv2.medianBlur(filtered,3)

    kernel = np.ones((5,5),np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

    # 2 - Outliers Removal 
    

    for i in range(5):

        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(filtered.astype('uint8'), connectivity, cv2.CV_32S)
        
        # 2A - Remove Small Area Instances
        idx = np.where(stats[:,cv2.CC_STAT_AREA] < 50)[0]   # 30

        for id in idx:
            filtered[labels == id] = 0

        # MAKE IT ITERATIVE HERE
        """
        # 2B - Split large pieces
        idx = np.where(stats[:,cv2.CC_STAT_AREA] > 300)[0]  #450

        for id in idx[1:]:

            instance = np.zeros((1000,1000),dtype='uint8') ; instance[labels == id] = 1
            contours, hierachy = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hull = cv2.convexHull(contours[0], returnPoints = False)
            defects = cv2.convexityDefects(contours[0], hull)

            # Find the large bay areas (Filter out using a threshold)
            #if(defects == None):
                #return filtered

            try:
                depths = [defects[i][...,-1][0] for i in range(len(defects))]
            except:
                return filtered

            bays_idx = np.argsort(depths)[::-1]

            # Consider only steep bays
            if(defects[bays_idx][0][0][3] < 1500):  # 1800
                continue

            try:
                bayA_idx = contours[0][defects[bays_idx][0][0][2]] ; bayB_idx = contours[0][defects[bays_idx][1][0][2]]
            except:
                continue

            # Connect here the two bays
            cv2.line(filtered, (bayA_idx[0][0], bayA_idx[0][1]), (bayB_idx[0][0], bayB_idx[0][1]), 0, thickness = connectBaysWidth)

    """
    #imshow(filtered, cmap='gray') ; plt.show()

    filtered = hole_fill(filtered)

    #print(numLabels)

    return filtered


def postProc2(pred, GT):

    connectBaysWidth = 2
    connectivity = 4

    #(numLabels_GT, _, _, _) = cv2.connectedComponentsWithStats(GT.astype('uint8'), connectivity, cv2.CV_32S)

    #imshow(pred, cmap='gray') ; plt.show()

    # 1 - Refinement
    filtered=hole_fill(pred)
    
    filtered = cv2.medianBlur(filtered,3) ; filtered = cv2.medianBlur(filtered,3) ; filtered = cv2.medianBlur(filtered,3)
    footprint=[[0, 1, 0], [1, 1, 1], [0,1,0]]
    filtered=cv2.dilate(filtered, np.asarray(footprint).astype('uint8'), iterations=1)
    kernel = np.ones((3,3),np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    

    # 2 - Outliers Removal 
    

    for i in range(2):

        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(filtered.astype('uint8'), connectivity, cv2.CV_32S)
        
        # 2A - Remove Small Area Instances
        idx = np.where(stats[:,cv2.CC_STAT_AREA] < 0)[0]   # 30

        for id in idx:
            filtered[labels == id] = 0

        # MAKE IT ITERATIVE HERE

        # 2B - Split large pieces
        idx = np.where(stats[:,cv2.CC_STAT_AREA] > 400)[0]  #450
        
        for id in idx[1:]:

            instance = np.zeros((1000,1000),dtype='uint8') ; instance[labels == id] = 1
            contours, hierachy = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hull = cv2.convexHull(contours[0], returnPoints = False)
            hull[::-1].sort(axis=0)
            defects = cv2.convexityDefects(contours[0], hull)

            # Find the large bay areas (Filter out using a threshold)
            #if(defects == None):
                #return filtered

            try:
                depths = [defects[i][...,-1][0] for i in range(len(defects))]
            except:
                return filtered

            bays_idx = np.argsort(depths)[::-1]

            # Consider only steep bays
            if(defects[bays_idx][0][0][3] < 1500):  # 1800
                continue

            try:
                bayA_idx = contours[0][defects[bays_idx][0][0][2]] ; bayB_idx = contours[0][defects[bays_idx][1][0][2]]
            except:
                continue

            # Connect here the two bays
            cv2.line(filtered, (bayA_idx[0][0], bayA_idx[0][1]), (bayB_idx[0][0], bayB_idx[0][1]), 0, thickness = connectBaysWidth)

    
    #imshow(filtered, cmap='gray') ; plt.show()

    filtered = hole_fill(filtered)

    #print(numLabels)

    return filtered

def patch_segmentation(image):
    # image_gt = Image.open("./img_patches/GroundTruth/img1_0_0_GT.png")
    # image_gt = np.array(image_gt).astype(np.uint8)
    # image = Image.open("./img_patches/training/img1_0_0.png")
    # image = np.array(image)

    print(image.shape)

    # pred_mask = mean_shift_to_binary(image)
    pred_mask = image
    expanded_pred_mask = np.expand_dims(pred_mask, axis=0).astype(np.uint8)
    pred_mask = morph_operations(expanded_pred_mask)
    pred_mask = pred_mask.squeeze().astype(np.uint8)
    # cv2.imshow("morphed pred_mask_patch", pred_mask)

    num_labels, pred_mask_labels = cv2.connectedComponents(
        pred_mask.astype(np.uint8))

    # hole filling for each nuclei
    patch_no_hole = np.zeros(pred_mask.shape).astype(np.uint8)
    pred_mask_no_hole = np.zeros(pred_mask.shape).astype(np.uint8)
    for i in range(1, num_labels):
        label_coord = np.where(pred_mask_labels == i)
        patch_no_hole[label_coord] = 255
        patch_no_hole = hole_fill(patch_no_hole)
        coord = np.where(patch_no_hole == 255)
        pred_mask_no_hole[coord] = 255

    pred_mask_wo_holes = hole_fill(pred_mask)
    pred_mask_wo_holes = pred_mask_no_hole

    old_pred_mask = pred_mask.copy()

    num_labels, pred_labels = cv2.connectedComponents(
        pred_mask_wo_holes.astype(np.uint8))

    # aji = AJI(pred_mask_wo_holes.squeeze(), image_gt.squeeze())
    # print("AJI (before post-proceesing):", aji)

    # compute nuclei sizes

    def nuclei_sizes(num_labels, pred_labels):
        # count frequency of nuclei labels
        nuclei_sizes = [np.count_nonzero(pred_labels[pred_labels == i])
                        for i in range(1, num_labels)]
        nuclei_sizes = np.array(nuclei_sizes)
        return nuclei_sizes

    # ------------------------------------------------------------------------------------
    # find largest nuclei
    nuclei_size = nuclei_sizes(num_labels, pred_labels)
    # for i in range(nuclei_size.shape[0]):
    #     if (nuclei_size[i] < 10):
    #         pred_mask_wo_holes[pred_labels == i] = 0

    top_5_large = nuclei_size.argsort()[::-1]
    for max_idx in top_5_large:
        # max_idx = np.argmax(nuclei_size)
        large_nuclei = np.zeros(pred_labels.shape).astype(np.uint8)
        large_nuclei[pred_labels == max_idx+1] = 255
        # plt.imshow(large_nuclei, 'gray')
        # plt.show()

        # # orginal large nuclei image
        # enhanced_large_nulcei = enhanced * large_nuclei
        # enhanced_large_nulcei = enhanced_large_nulcei.astype(np.uint8)
        # enhanced_large_nulcei = exposure.equalize_adapthist(
        #     enhanced_large_nulcei)*255

        # ----------------------------------------------------------------------------------
        # process one large nuclei
        # 1. find bay areas
        # 2. find corner points on all bays
        # 3. compute normal vectors given corner points on all bays (remove unwanted corner points by searching locally)
        # 4. remove unwanted tunnels based on normal vectors and
        # 5. replace old large nulcei with separated nuclei

        def find_convexed_nuclei_and_its_bay_areas(large_nuclei):
            chull = convex_hull_image(large_nuclei)
            chull_coord = np.where(chull == 1)
            convexed_large = np.zeros(large_nuclei.shape)
            convexed_large[chull_coord] = 255
            convexed_large = convexed_large.astype(np.uint8)
            # plt.subplot(3, 1, 1)
            # plt.imshow(convexed_large)
            # plt.subplot(3, 1, 2)
            # plt.imshow(large_nuclei)

            bay_diff = convexed_large - large_nuclei
            bay_diff = np.array(bay_diff).astype(np.uint8)
            bay_diff = cv2.medianBlur(bay_diff, 3)
            # plt.subplot(3, 1, 3)
            # plt.imshow(bay_diff, 'gray')
            # plt.show()
            return [convexed_large, bay_diff]

# TODO: need to add priority to select the corner points for tunnel creation
        def find_all_corner_points_in_a_group_of_nuclei(large_nuclei, large_nuclei_bay_areas):
            # TODO: Add curvatures
            #
            #
            #
            #
            # plt.subplot(3, 1, 1)
            # plt.imshow(large_nuclei, 'gray')
            # #
            # plt.subplot(3, 1, 2)
            # plt.imshow(large_nuclei_bay_areas, 'gray')

            small_size_delete = 15
            #
            num_labels, large_nuclei_labels = cv2.connectedComponents(
                large_nuclei_bay_areas.astype(np.uint8))
            bay_sizes = nuclei_sizes(num_labels, large_nuclei_labels)
            num_bay_less_amount = np.count_nonzero(
                bay_sizes[bay_sizes < small_size_delete])
            num_qualified_bays = len(bay_sizes)-num_bay_less_amount

            corner_points_all_bay = []
            for idx in range(1, num_labels):
                # remove small bay < 10 pixels
                if bay_sizes[idx-1] > small_size_delete and num_qualified_bays > 1:
                    bay = np.zeros(large_nuclei_labels.shape)
                    bay[large_nuclei_labels == idx] = 1
                    point = np.array(np.where(bay == 1)).transpose()
                    if point.shape[0] > 7:
                        hull = ConvexHull(point)
                        corner_points_all_bay.append(point[hull.vertices])

            all_corner_show = np.zeros(large_nuclei_labels.shape)
            for i in range(len(corner_points_all_bay)):
                for p in corner_points_all_bay[i]:
                    all_corner_show[p[0]][p[1]] = 1
            #
            # plt.subplot(3, 1, 3)
            # plt.imshow(all_corner_show, 'gray')
            # plt.show()
            #
            return corner_points_all_bay

        def check_two_sides_are_land(large_nuclei, x, y, ratio=0.5):
            # at each corner point, check its black white ratio in a box
            land = 0
            side = 3
            for i in range(x-side, x+side):
                for j in range(y-side, y+side):
                    if i >= large_nuclei.shape[0] or j >= large_nuclei.shape[1]:
                        continue
                    elif large_nuclei[i][j] != 0:
                        land += 1
            # print(land/((2*side)**2))
            return 1 if land / (2*side*2*side) > ratio else 0

        def draw_box(large_nuclei_bay_areas, x, y):
            new_area = np.array(large_nuclei_bay_areas)
            land = 0
            side = 3
            for i in range(x-side, x+side):
                for j in range(y-side, y+side):
                    new_area[i][j] = 128
            return new_area

        # normal vector could be useless

        def compute_normal_vector(corner_points_all_bay):
            # compute normal vector for all bays
            normal_vectors = []

            for i in range(len(corner_points_all_bay)):

                one_bay = corner_points_all_bay[i]
                # one_bay = np.array(one_bay).transpose()
                x_t = np.gradient(one_bay[:, 0])
                y_t = np.gradient(one_bay[:, 1])
                xx_t = np.gradient(x_t)
                yy_t = np.gradient(y_t)

                normal = np.array([[xx_t[i], yy_t[i]]
                                   for i in range(len(one_bay))])
                normal_vectors.append(normal)
            normal_vectors = np.array(normal_vectors)
            # print('normal vectors:', normal_vectors)
            return normal_vectors

        def find_tunnels_for_all_bay_areas(corner_points_all_bay, normal_vectors):
            line_list = []
            for pidx in range(len(corner_points_all_bay)):
                # min_dist = float('inf')
                # every corner point in a bay
                one_bay = []
                idx = 0
                for vx, vy in corner_points_all_bay[pidx]:
                    bay_corner_lines = []
                    if check_two_sides_are_land(large_nuclei, vx, vy):
                        # show = draw_box(large_nuclei, vx, vy)
                        # plt.imshow(show, 'gray')
                        # plt.show()

                        # other bays
                        # every point in other bay
                        min_dist = float('inf')
                        min_dist_corners = ((), ())
                        for next_point_idx in range(len(corner_points_all_bay)):
                            next_idx = 0
                            if (next_point_idx != pidx):
                                # every point in other bay
                                for nx, ny in corner_points_all_bay[next_point_idx]:

                                    if check_two_sides_are_land(large_nuclei, nx, ny):
                                        # show = draw_box(large_nuclei, nx, ny)
                                        # plt.imshow(show, 'gray')
                                        # plt.show()
                                        dist = np.linalg.norm([vx-nx, vy-ny])
                                        if (min_dist > dist):
                                            min_dist = dist
                                            min_dist_corners = (([vy, vx], [ny, nx]), ([normal_vectors[pidx][idx]], [
                                                normal_vectors[next_point_idx][next_idx]]))
                                    next_idx += 1
                        bay_corner_lines.append(min_dist_corners)
                    idx += 1
                    one_bay.append(bay_corner_lines)
                line_list.append(one_bay)
            # print(line_list)
            line_list = np.array(line_list)
            # print('normal:', normal_vectors.shape)
            # print('line_list:', line_list.shape)
            return line_list

        def create_tunnels_and_remove_unwanted_tunnels(line_list, large_nuclei_bay_areas, large_nuclei):
            current_nuclei = large_nuclei.copy()
            for i in range(len(line_list)):
                for j in range(len(line_list[i])):
                    for line in range(len(line_list[i][j])):
                        if len(line_list[i][j][line][0]) != 0 and len(line_list[i][j][line][1]) != 0:
                            # print(line_list[i][j][line][1][0][0],
                            #       line_list[i][j][line][1][1][0])

                            x1 = line_list[i][j][line][1][1][0][0]
                            x2 = line_list[i][j][line][1][0][0][0]
                            y1 = line_list[i][j][line][1][1][0][1]
                            y2 = line_list[i][j][line][1][0][0][1]
                            if (x1*x2 < 0 or y1*y2 < 0):
                                # cv2.line(large_nuclei_bay_areas, line_list[i][j][line][0][1],
                                #  line_list[i][j][line][0][0], 255, thickness=1)
                                cv2.line(
                                    current_nuclei, line_list[i][j][line][0][1], line_list[i][j][line][0][0], 0, thickness=2)

                                num_labels, pred_labels = cv2.connectedComponents(
                                    current_nuclei.astype(np.uint8))
                                sizes = nuclei_sizes(num_labels, pred_labels)
                                small_nuclei_size = min(sizes)
                                if small_nuclei_size >= 0:
                                    # plt.subplot(2, 1, 1)
                                    # plt.imshow(current_nuclei, 'gray')

                                    cv2.line(
                                        large_nuclei_bay_areas, line_list[i][j][line][0][1], line_list[i][j][line][0][0], 255, thickness=2)
                                    # plt.subplot(2, 1, 2)
                                    # plt.imshow(large_nuclei_bay_areas, 'gray')
                                    # plt.show()

            return large_nuclei_bay_areas
# the line is drawed on the bay areas, i can make a copy of lagre nuclei and place the line on it to compute the nuclei size

        def replace_old_nuclei_with_separated_nuclei(convexed_large, large_nuclei_bay_areas_with_tunnels, pred_mask_wo_holes):
            pred_mask = convexed_large - large_nuclei_bay_areas_with_tunnels
            pred_mask = cv2.morphologyEx(
                pred_mask, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)
            pred_mask_wo_holes = pred_mask_wo_holes - large_nuclei + pred_mask
            pred_mask_wo_holes = np.array(pred_mask_wo_holes).astype(np.uint8)
            return pred_mask_wo_holes

        # convexed_large: max=255. large_nuclei_bay_areas: max=255
        convexed_large, large_nuclei_bay_areas = find_convexed_nuclei_and_its_bay_areas(
            large_nuclei)
        # plt.imshow(large_nuclei_bay_areas, 'gray')
        # plt.show()

        corner_points_all_bay = find_all_corner_points_in_a_group_of_nuclei(large_nuclei,
                                                                            large_nuclei_bay_areas)
        if len(corner_points_all_bay) > 1:
            normal_vectors = compute_normal_vector(corner_points_all_bay)
            line_list = find_tunnels_for_all_bay_areas(
                corner_points_all_bay, normal_vectors)
            # returned bay areas with tunnels attached
            large_nuclei_bay_areas_with_tunnels = create_tunnels_and_remove_unwanted_tunnels(
                line_list, large_nuclei_bay_areas, large_nuclei)
            # plt.imshow(large_nuclei_bay_areas_with_tunnels, 'gray')
            # plt.show()

            pred_mask_wo_holes = replace_old_nuclei_with_separated_nuclei(
                convexed_large, large_nuclei_bay_areas_with_tunnels, pred_mask_wo_holes)
            # plt.imshow(pred_mask_wo_holes, 'gray')
            # plt.show()
        else:
            # for these with one bay, I can remove the bay by using convexed_large
            pred_mask_wo_holes = pred_mask_wo_holes - large_nuclei + convexed_large

    pred_mask = pred_mask_wo_holes

    # fig, axes = plt.subplots(1, 4, figsize=(8, 4))
    # ax = axes.ravel()

    # ax[0].imshow(old_pred_mask, 'gray')
    # ax[0].set_title('old_pred_mask')
    # ax[1].imshow(pred_mask, 'gray')
    # ax[1].set_title('Pred_mask_after_process')
    # ax[2].imshow(image_gt, 'gray')
    # ax[2].set_title('Ground_Truth')
    # ax[3].imshow(image)
    # ax[3].set_title('original image')
    # plt.show()
    # # compute AJI score
    # aji = AJI(pred_mask, image_gt)

    # fig, axes = plt.subplots(1, 4, figsize=(8, 4))
    # ax = axes.ravel()

    # ax[0].imshow(old_pred_mask, 'gray')
    # ax[0].set_title('old_pred_mask')
    # ax[1].imshow(pred_mask, 'gray')
    # ax[1].set_title('Pred_mask_after_process')
    # ax[2].imshow(image_gt, 'gray')
    # ax[2].set_title('Ground_Truth')
    # ax[3].imshow(image)
    # ax[3].set_title('original image')
    # plt.show()
    # print("AJI (after post-processing):", aji)

    return [pred_mask]
    # normal vector explained
    # https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
    # draw tangent line
    # https://answers.opencv.org/question/129819/finding-distance-between-two-curves/

    # Normal vector and curvature at a point on the curve
    # https://www.delftstack.com/howto/numpy/curvature-formula-numpy/
    # https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Vector_Calculus/2%3A_Vector-Valued_Functions_and_Motion_in_Space/2.3%3A_Curvature_and_Normal_Vectors_of_a_Curve
    # https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Vector_Calculus/2%3A_Vector-Valued_Functions_and_Motion_in_Space/2.3%3A_Curvature_and_Normal_Vectors_of_a_Curve
