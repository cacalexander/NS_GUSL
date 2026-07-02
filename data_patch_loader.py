import glob
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
#from Code.utils.lossfunctions import *
from skimage.util.shape import view_as_windows
import json
import scipy
import cv2
from scipy.io import loadmat


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def recoverImg(tiles, row, col):
    with open('./config.json') as config_file:
        config = json.load(config_file)
    # print(config)
    im_width = config['im_width']
    im_height = config['im_height']
    patch_width = config['patch_width']
    patch_height = config['patch_height']
    # Epochs = config['Epochs']
    img = np.zeros((im_height, im_width, 1))
    # print(img.shape)
    count = 0
    for x in range(0, row):
        for y in range(0, col):
            if x == 0 and y == 0:
                startX = 0
                endX = patch_height
                startY = 0
                endY = patch_width
            else:
                # overlapping area replace previous area
                startX = x*(patch_width)//2
                endX = startX + (patch_width)
                startY = y*(patch_width)//2
                endY = startY + (patch_width)
            # print(startX, startY)
            # print(endX, endY)
            # print(tiles[count].shape)
            # print(img.shape)
            img[startX:endX, startY:endY, :] = tiles[count]
            count = count + 1

    return img


def generate_patches(full_pred_mask, gt, patch_width=256, patch_height=256, stride=50, if_rgb=False):
    X_train = []
    y_train = []
    # patch_width = 256
    # patch_height = 256
    new_imgs = None
    #
    if if_rgb:
        new_imgs = view_as_windows(
            full_pred_mask, (patch_width, patch_height, 3), (stride, stride, 3))
    else:
        new_imgs = view_as_windows(
            full_pred_mask, (patch_width, patch_height), (stride, stride))
    #
    for patch in new_imgs:
        X_train.append(patch)

    print(gt.shape, patch_height, patch_width, stride)

    new_masks = view_as_windows(
        np.squeeze(gt), (patch_width, patch_height), (stride, stride))
    for patch in new_masks:
        y_train.append(patch)
    #
    if if_rgb:
        X_train = np.array(X_train).reshape(-1, patch_width, patch_height, 3)
    else:
        # all patches in a image
        X_train = np.array(X_train).reshape(-1, patch_width, patch_height)
    #
    # print(X_train.shape)
    y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
                                                        patch_width, patch_height)
    return [X_train, y_train]

def generate_patches_new(full_pred_mask, gt, patch_width=256, patch_height=256, stride=50, if_rgb=False):
    X_train = []
    y_train = []
    # patch_width = 256
    # patch_height = 256
    new_imgs = None
    #
    if if_rgb:
        new_imgs = view_as_windows(
            full_pred_mask, (patch_width, patch_height, 6), (stride, stride, 6))
    else:
        new_imgs = view_as_windows(
            full_pred_mask, (patch_width, patch_height), (stride, stride))
    #
    for patch in new_imgs:
        X_train.append(patch)

    print(gt.shape, patch_height, patch_width, stride)

    new_masks = view_as_windows(
        np.squeeze(gt), (patch_width, patch_height), (stride, stride))
    for patch in new_masks:
        y_train.append(patch)
    #
    if if_rgb:
        X_train = np.array(X_train).reshape(-1, patch_width, patch_height, 6)
    else:
        # all patches in a image
        X_train = np.array(X_train).reshape(-1, patch_width, patch_height)
    #
    # print(X_train.shape)
    y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
                                                        patch_width, patch_height)
    return [X_train, y_train]

def image_patch_loader(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
     #   config = json.load(config_file)
    im_width = 1000
    im_height = 1000
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/Training/TissueImages/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/Training/GroundTruth/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale',
                        target_size=[im_width, im_height]))

    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, mask, base]


def image_patch_loader_test(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
        #config = json.load(config_file)
    im_width = 1000
    im_height = 1000
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/Test/TissueImages_PNG/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/Test/GroundTruth/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale',
                        target_size=[im_width, im_height]))

    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, mask, base]

def image_patch_loaderc(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
     #   config = json.load(config_file)
    im_width = 512
    im_height = 512
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/CroNuSeg/TissueImages/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/CroNuSeg/masks/mask_binary/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale',
                        target_size=[im_width, im_height]))

    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, mask, base]

def image_patch_loadercons(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
     #   config = json.load(config_file)
    im_width = 1000
    im_height = 1000
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/CoNSep/Test/Images/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/CoNSep/Test/Labels/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    #mask = img_to_array(load_img(y, color_mode='grayscale',
    #                    target_size=[im_width, im_height]))
    mask = np.asarray(scipy.io.loadmat(y)['inst_map'])
    mask = np.expand_dims(mask, -1)
    mask[mask>0] = 1
    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, mask, base]

def image_patch_loadercpm(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
     #   config = json.load(config_file)
    im_width = 512
    im_height = 512
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/CPM17/test/Images/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/CPM17/test/Labels/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    #mask = img_to_array(load_img(y, color_mode='grayscale',
    #                    target_size=[im_width, im_height]))
    mask = np.asarray(scipy.io.loadmat(y)['inst_map'])
    mask = np.expand_dims(mask, -1)
    mask[mask>0] = 1
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, np.expand_dims(mask_resized,-1), base]

def image_patch_loadercpmt(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
     #   config = json.load(config_file)
    im_width = 512
    im_height = 512
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/CPM17/train/Images/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/CPM17/train/Labels/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    #mask = img_to_array(load_img(y, color_mode='grayscale',
    #                    target_size=[im_width, im_height]))
    mask = np.asarray(scipy.io.loadmat(y)['inst_map'])
    mask = np.expand_dims(mask, -1)
    mask[mask>0] = 1
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, np.expand_dims(mask_resized,-1), base]

def image_patch_loadertnbc(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
     #   config = json.load(config_file)
    im_width = 512
    im_height = 512
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/tnbc/Images/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/tnbc/Labels/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    #mask = img_to_array(load_img(y, color_mode='grayscale',
    #                    target_size=[im_width, im_height]))
    mask = np.asarray(scipy.io.loadmat(y)['inst_map'])
    mask = np.expand_dims(mask, -1)
    mask[mask>0] = 1
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, np.expand_dims(mask_resized,-1), base]


def image_patch_loader_testc(img_idx, mode='Self'):
    #with open('C:/Users/aurel/OneDrive/Documents/MCL/Nuclei Segmentation - Code/Config/config_breast_6.json') as config_file:
        #config = json.load(config_file)
    im_width = 1000
    im_height = 1000
    patch_width = 256
    patch_height = 256
    #print(config['TRAIN_PATH_IMAGES'])
    #print(config['TRAIN_PATH_GT'])
    TRAIN_PATH_IMAGES = '/home/cc98905/Test/TissueImages_PNG/*' #config['TRAIN_PATH_IMAGES']
    TRAIN_PATH_GT = '/home/cc98905/Test/GroundTruth/' #config['TRAIN_PATH_GT']
    # TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
    # TEST_PATH_GT = config['TEST_PATH_GT']

    ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
    ids_train_y = glob.glob(TRAIN_PATH_GT)
    #print("No. of training images = ", len(ids_train_x))
    # ids_test_x = glob.glob(TEST_PATH_IMAGES)
    # ids_test_y = glob.glob(TEST_PATH_GT)
    # print("No. of testing images = ", len(ids_test_x))

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #print("Loading Training Data")
    count = 0
    # for x in (ids_train_x):
    # use the first image
    x = ids_train_x[img_idx]
    base = os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(TRAIN_PATH_GT+fn+'*')[0]
    original_img = img_to_array(load_img(x, color_mode='rgb',
                                        target_size=[im_width, im_height]))

    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale',
                        target_size=[im_width, im_height]))

    #original_img = cv2.resize(
        #original_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)

    #original_img = None

    #mask = cv2.resize(mask, (1000, 1000),
                      #interpolation=cv2.INTER_AREA).astype(np.uint8)

    #mask[mask >= 128] = 255
    #mask[mask < 128] = 0


    # new_imgs = view_as_windows(
    #     original_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))

    # for patch in new_imgs:
    #     X_train.append(patch)
    # new_masks = view_as_windows(
    #     mask, (patch_width, patch_height), (patch_width//2, patch_height//2))
    # for patch in new_masks:
    #     y_train.append(patch)
    # count = count+1

    # # all patches in a image
    # X_train = np.array(X_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height, 3)
    # # print(X_train.shape)
    # y_train = np.array(y_train, dtype=np.uint8).reshape(-1,
    #                                                     patch_width, patch_height)

    # from matplotlib import pyplot as plt
    # original_img = np.array(original_img).astype('uint8')

    return [original_img, mask, base]
# print("Loading Testing Data")
# count = 0
# for x in (ids_test_x):
#     base = os.path.basename(x)
#     fn = os.path.splitext(base)[0]
#     y = glob.glob(config['TEST_PATH_GT']+fn+'*')[0]
#     x_img = img_to_array(load_img(x, color_mode='rgb',
#                          target_size=[im_width, im_height]))
#     x_img = x_img/255.0
#     # Load masks
#     mask = img_to_array(load_img(y, color_mode='grayscale',
#                         target_size=[im_width, im_height]))
#     mask = mask/255.0
#     #X_test[count] = x_img/255.0
#     #y_test[count] = mask/255.0
#     new_imgs = view_as_windows(
#         x_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))
#     for patch in new_imgs:
#         X_test.append(patch)
#     new_masks = view_as_windows(
#         mask, (patch_width, patch_height, 1), (patch_width//2, patch_height//2, 1))
#     for patch in new_masks:
#         y_test.append(patch)
#     count = count+1


# print(len(X_train),len(y_train))
# print(len(X_test),len(y_test))


# X_train = np.array(X_train, dtype=np.float32)
# y_train = np.array(y_train, dtype=np.float32)
# # X_test = np.array(X_test)
# # y_test = np.array(y_test)
# print("before reshape X train:", X_train.shape)

# X_train = X_train.reshape(-1, patch_height, patch_width, 3)
# y_train = y_train.reshape(-1, patch_height, patch_width, 1)
# # X_test = X_test.reshape(-1, patch_height, patch_width, 3)
# # y_test = y_test.reshape(-1, patch_height, patch_width, 1)
# print("after reshape X train:", X_train.shape)
# # print("X train:", X_train.shape)
