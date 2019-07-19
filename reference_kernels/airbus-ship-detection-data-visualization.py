# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: kaggle_airbus_ships
#     language: python
#     name: kaggle_airbus_ships
# ---

# %% [markdown] {"_uuid": "18c735c4a3e7d53bbcd82dfd676240301cb075e0"}
# **<h2>Introduction**

# %% [markdown] {"_uuid": "d30127c5abec33501bff60055ab83187a546a5c6"}
# In this notebook, I try to explore the Airbus Ship Detection Challenge data and get some sense of what types of features may be useful. This is work in progress, so i will keep updating it. I hope you find this helpful. Happy Kaggling :-)

# %% {"_kg_hide-input": true, "_uuid": "f926376d1380568f248346353c42fcf599e6c0e6"}
import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from tqdm import tqdm_notebook, tnrange
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.feature import canny
from skimage.filters import sobel,threshold_otsu, threshold_niblack,threshold_sauvola
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from scipy import signal

import cv2
from PIL import Image
import pdb
from tqdm import tqdm
import seaborn as sns
import os 
from glob import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import zipfile

# %% [markdown]
# <h2> Setting paths

# %% [markdown]
# Input path

# %%
os.environ['DATA_PATH'] = './input'
INPUT_PATH = Path(os.environ['DATA_PATH'])
assert INPUT_PATH.exists()

TRAIN_ZIP_PATH = INPUT_PATH / 'train.zip'
assert TRAIN_ZIP_PATH.exists()
train_zip = zipfile.ZipFile(INPUT_ZIP_PATH.absolute())
files = data_zip.namelist()


# %%
DATA_PATH = INPUT_PATH

TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train/masks")
TEST_DATA = os.path.join(DATA_PATH, "test")
df = pd.read_csv(DATA_PATH+'/train_ship_segmentations.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')


# %% [markdown]
# <h2> Some utility functions

# %%
def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb
def mask_overlay(image, mask):
    """
    Helper function to visualize mask
    """
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img


# %% [markdown]
# <h2>**Plotting Images**

# %% [markdown]
# Lets plot some random images from training set and then few more images with the mask overlayed on top of it.

# %%
nImg = 32  #no. of images that you want to display
np.random.seed(42)
if df.index.name == 'ImageId':
    df = df.reset_index()
if df.index.name != 'ImageId':
    df = df.set_index('ImageId')
    
_train_ids = list(train_ids)
np.random.shuffle(_train_ids)
# _train_ids = _train_ids[:nImg]
tile_size = (256, 256)
n = 8
alpha = 0.3

# m = int(np.ceil(len(_train_ids) * 1.0 / n))
m = int(np.ceil(nImg * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    j = 0
    while j < n:
        counter += 1
        all_masks = np.zeros((768, 768))
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        image_id = _train_ids[counter]
        if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
            continue
        else:
            j += 1
        img = get_image_data(image_id, 'Train')
        
        try:
            img_masks = df.loc[image_id,'EncodedPixels'].tolist()
            for mask in img_masks:
                all_masks += rle_decode(mask)
            all_masks = np.expand_dims(all_masks,axis=2)
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')*255
            
            img_masked = mask_overlay(img, all_masks)
            
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)

            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]

        except Exception as e:
            all_masks = rle_decode(df.loc[image_id,'EncodedPixels'])
            all_masks = np.expand_dims(all_masks,axis=2)*255
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')
            
            img_masked = mask_overlay(img, all_masks)        
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)
    #         img_masked = cv2.bitwise_and(img, img, mask=all_masks)

            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]
#             pdb.set_trace()

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]

        
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image[ys:ye,:,:],cmap='seismic')
    plt.title("Training dataset")
    
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image_masked[ys:ye,:,:])
    plt.title("Training dataset: Lighter Color depicts ship")

# %% [markdown]
# <h3>Plotting Ship Count

# %%
df = df.reset_index()
df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
df.loc[df['EncodedPixels'].isnull().values,'ship_count'] = 0  #see infocusp's comment

# %%
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.distplot(df['ship_count'],kde=False)
plt.title('Ship Count Distribution in Train Set')

print(df['ship_count'].describe())

# %% [markdown]
# <h2>**Plotting Images: Based on Ship Count**

# %% [markdown]
# Let's plot some images having different ship counts and try to see if we are able to glean any differences. This way, we can get some sense of what we're looking at. The images are 768 x 768 pixels each with the mask (in lighter color) overlayed on top of it. 

# %%
df.head()

# %% [markdown]
# <h2> Training Set Images with Ship Count 0 i.e. no ship

# %%
nImg = 32  #no. of images that you want to display
np.random.seed(42)
if df.index.name == 'ImageId':
    df = df.reset_index()
if df.index.name != 'ImageId':
    df = df.set_index('ImageId')
    
_train_ids = list(train_ids)
# _train_ids = list(train_ids[idx])
np.random.shuffle(_train_ids)
# _train_ids = _train_ids[:nImg]
tile_size = (256, 256)
n = 8
alpha = 0.4

# m = int(np.ceil(len(_train_ids) * 1.0 / n))
m = int(np.ceil(nImg * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    j = 0
    while j < n:
        counter += 1

        all_masks = np.zeros((768, 768))
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        image_id = _train_ids[counter]
        if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
            j += 1            
        else:
            continue
        img = get_image_data(image_id, 'Train')
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
        complete_image[ys:ye, xs:xe, :] = img[:,:,:]


    
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image[ys:ye,:,:])
    plt.title("Training Set Ship Count 0 i.e. no ship")

# %% [markdown]
# <h2> Training Set Images with Ship Count between 1 to 5

# %%
nImg = 32  #no. of images that you want to display
np.random.seed(42)
idx = np.ravel(np.where((df['ship_count']<6) ) )
if df.index.name == 'ImageId':
    df = df.reset_index()
_train_ids = list(df.loc[idx,'ImageId'])
if df.index.name != 'ImageId':
    df = df.set_index('ImageId')

np.random.shuffle(_train_ids)

tile_size = (256, 256)
n = 8
alpha = 0.4

m = int(np.ceil(nImg * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    j = 0
    while j < n:
        counter += 1
        all_masks = np.zeros((768, 768))
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        image_id = _train_ids[counter]
        if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
            continue
        else:
            j += 1
        img = get_image_data(image_id, 'Train')
        
        try:
            img_masks = df.loc[image_id,'EncodedPixels'].tolist()
            for mask in img_masks:
                all_masks += rle_decode(mask)
            all_masks = np.expand_dims(all_masks,axis=2)
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')*255
            
            img_masked = mask_overlay(img, all_masks)
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)
    #         img_masked = cv2.bitwise_and(img, img, mask=all_masks)

            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]

        except Exception as e:
            all_masks = rle_decode(df.loc[image_id,'EncodedPixels'])
            all_masks = np.expand_dims(all_masks,axis=2)*255
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')
        
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)
            img_masked = mask_overlay(img, all_masks)
            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
    
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image_masked[ys:ye,:,:])
    plt.title("Training Set Ship Count 1 to 5")

# %% [markdown]
# <h2> Training Set Images with Ship Count 5 to 10

# %%
nImg = 32  #no. of images that you want to display
np.random.seed(42)
idx = np.ravel(np.where((df['ship_count']<11) & (df['ship_count']>5)) )
if df.index.name == 'ImageId':
    df = df.reset_index()
_train_ids = list(df.loc[idx,'ImageId'])
if df.index.name != 'ImageId':
    df = df.set_index('ImageId')

np.random.shuffle(_train_ids)

tile_size = (256, 256)
n = 8
alpha = 0.4

m = int(np.ceil(nImg * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    j = 0
    while j < n:
        counter += 1
        all_masks = np.zeros((768, 768))
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        image_id = _train_ids[counter]
        if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
            continue
        else:
            j += 1
        img = get_image_data(image_id, 'Train')
        
        try:
            img_masks = df.loc[image_id,'EncodedPixels'].tolist()
            for mask in img_masks:
                all_masks += rle_decode(mask)
            all_masks = np.expand_dims(all_masks,axis=2)
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')*255
            
            img_masked = mask_overlay(img, all_masks)
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)
    #         img_masked = cv2.bitwise_and(img, img, mask=all_masks)

            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]

        except Exception as e:
            all_masks = rle_decode(df.loc[image_id,'EncodedPixels'])
            all_masks = np.expand_dims(all_masks,axis=2)*255
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')
        
            img_masked = mask_overlay(img, all_masks)
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)

            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
    
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image_masked[ys:ye,:,:])
    plt.title("Training Set Ship Count 5 to 10")

# %% [markdown]
# <h2>Training Set Images with Ship Count greater than 10

# %% {"_kg_hide-input": true, "_uuid": "2977babea3db6b65cb3a3942e136e18a4873101f"}
nImg = 32  #no. of images that you want to display
np.random.seed(42)
idx = np.ravel(np.where((df['ship_count']>10) ) )
if df.index.name == 'ImageId':
    df = df.reset_index()
_train_ids = list(df.loc[idx,'ImageId'])
if df.index.name != 'ImageId':
    df = df.set_index('ImageId')
# _train_ids = list(train_ids[idx])
np.random.shuffle(_train_ids)
# _train_ids = _train_ids[:nImg]
tile_size = (256, 256)
n = 8
alpha = 0.4

m = int(np.ceil(nImg * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    j = 0
    while j < n:
        counter += 1
        all_masks = np.zeros((768, 768))
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        image_id = _train_ids[counter]
        if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
            continue
        else:
            j += 1
        img = get_image_data(image_id, 'Train')
        
        try:
            img_masks = df.loc[image_id,'EncodedPixels'].tolist()
            for mask in img_masks:
                all_masks += rle_decode(mask)
            all_masks = np.expand_dims(all_masks,axis=2)
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')*255
            
            img_masked = mask_overlay(img, all_masks)
#             img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)
            
            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]
#             pdb.set_trace()

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
            
#             pdb.set_trace()
        except Exception as e:
#             print(e,counter)
            all_masks = rle_decode(df.loc[image_id,'EncodedPixels'])
            all_masks = np.expand_dims(all_masks,axis=2)*255
            all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')
        
            img_masked =  cv2.addWeighted(img, alpha, all_masks, 1 - alpha,0)
    #         img_masked = cv2.bitwise_and(img, img, mask=all_masks)

            img = cv2.resize(img, dsize=tile_size)
            img_masked = cv2.resize(img_masked, dsize=tile_size)

            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]
#             pdb.set_trace()

            img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
    
m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image_masked[ys:ye,:,:])
    plt.title("Training Set Ship Count greater than 10")

# %% {"_kg_hide-input": true, "_uuid": "1b6e6add8ac352017f60b2e9bf401355b8977d94"}
# takes too long.. have to optimize it
# # _train_ids = list(df_train.loc[count_idx[0],'id']+'.png')

# idx = np.ravel(np.where((df['ship_count']<6) & (df['ship_count']>1)) )
# if df.index.name == 'ImageId':
#     df = df.reset_index()
# _train_ids1 = list(df.loc[idx,'ImageId'])

# idx = np.ravel(np.where((df['ship_count']<11) & (df['ship_count']>5)) )
# _train_ids2 = list(df.loc[idx,'ImageId'])

# idx = np.ravel(np.where((df['ship_count']>10) ) )
# _train_ids3 = list(df.loc[idx,'ImageId'])
# df = df.set_index('ImageId')

# # takes a long time...
# # mask_count1 = np.zeros((768, 768,len(_train_ids1)), dtype=np.float32)
# # for n, id_ in tqdm(enumerate(_train_ids1), total=len(_train_ids1)):
# #     image_id = _train_ids1[n]
# #     img_masks = df.loc[image_id,'EncodedPixels'].tolist()
# #     all_masks = np.zeros((768, 768))
# #     for mask in img_masks:
# #         all_masks += rle_decode(mask)
# #     mask_count1[:,:,n] = (all_masks>0).astype('uint8')

# # mean_mask_count1 = mask_count1.mean(axis=2)
# # del mask_count1

# mask_count2 = np.zeros((768, 768,len(_train_ids2)), dtype=np.float32)
# for n, id_ in tqdm(enumerate(_train_ids2), total=len(_train_ids2)):
#     image_id = _train_ids2[n]
#     img_masks = df.loc[image_id,'EncodedPixels'].tolist()
#     all_masks = np.zeros((768, 768))
#     for mask in img_masks:
#         all_masks += rle_decode(mask)
#     mask_count2[:,:,n] = (all_masks>0).astype('uint8')

# mean_mask_count2 = mask_count2.mean(axis=2)
# del mask_count2

# mask_count3 = np.zeros((768, 768,len(_train_ids3)), dtype=np.float32)
# for n, id_ in tqdm(enumerate(_train_ids3), total=len(_train_ids3)):
#     image_id = _train_ids3[n]
#     img_masks = df.loc[image_id,'EncodedPixels'].tolist()
#     all_masks = np.zeros((768, 768))
#     for mask in img_masks:
#         all_masks += rle_decode(mask)
#     mask_count3[:,:,n] = (all_masks>0).astype('uint8')

# mean_mask_count3 = mask_count3.mean(axis=2)
# del mask_count3


# %% {"_kg_hide-input": true, "_uuid": "0a03afddb7f102138cb09d3e6e79989d2a851f21"}
# fig = plt.figure(1,figsize=(30,15))

# # ax = fig.add_subplot(1,3,1)
# # ax.imshow(mean_mask_count1)
# # ax.set_title("Ship Location for Count: 0 to 5")

# ax = fig.add_subplot(1,2,1)
# ax.imshow(mean_mask_count2)
# ax.set_title("Ship Location for Count: 5 to 10")

# ax = fig.add_subplot(1,2,2)
# ax.imshow(mean_mask_count3)
# ax.set_title("Ship Location for Count: 5 to 10")

# plt.suptitle('Mean Masks for different s',y=0.8)

# %% [markdown] {"_uuid": "730aaf122a1e26e85f339dcd48d5cbe02f072529"}
# **<h2>Transforming the Images**

# %% [markdown] {"_uuid": "3fc5f26b5732e9fde19373693f6f9020553e991f"}
# Let's try to transform the images in some way to enhance the contrast between the ship and the background.

# %% [markdown] {"_uuid": "2cc9558e03b0da2a6893182d5e9b72e591964117"}
# <h3>**Smoothing the Image**

# %% {"_uuid": "b1fbefd0c66cbfbf2609a05eff6dbcd8aa40af28"}
from skimage.filters import gaussian,laplace

# %% {"_kg_hide-input": true, "_uuid": "63604095e13f6121891daff222e589dfc6680a3f"}
_train_ids = list(train_ids)
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    image_id = _train_ids[np.random.randint(0,len(_train_ids))]
    ax = fig.add_subplot(3,3,i+1)
    img = get_image_data(image_id, 'Train')
    img = gaussian(img)
    img = cv2.resize(img, dsize=tile_size)
    ax.imshow(img)
    ax.set_title('Smoothed Image')
    
plt.show()
plt.suptitle('Smoothed Images')

# %% {"_uuid": "729250fbe0ba64e10df5318655e70147e5e1b39d"}
from skimage.feature import canny
from skimage.filters import scharr
from skimage import exposure
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value


# %% {"_kg_hide-input": true, "_uuid": "9decd70b744e36866a42fbc91f876fddd4be22a5"}
@adapt_rgb(hsv_value)
def canny_hsv(image):
    return canny(image)

@adapt_rgb(hsv_value)
def scharr_hsv(image):
    return scharr(image)


# %% [markdown] {"_uuid": "a27a8ccc5cad45f7479b825b698e105c3c8062ab"}
# <h2> Extracting some useful features

# %% {"_uuid": "c871dc640f824a9e6f971ce92a3cee094f03c9f4"}
# simple features that can be easily extracted and used for training deep networks
# these features may be used along with original image
np.random.seed(13)
# random.seed(12)
plt.figure(figsize=(30,15))
plt.subplots_adjust(bottom=0.2, top=1.2)  #adjust this to change vertical and horiz. spacings..
nImg = 5  #no. of images to process
j = 0
for _ in range(nImg):
    q = j+1
    image_id = _train_ids[np.random.randint(0,len(_train_ids))]
    ax = fig.add_subplot(3,3,i+1)
    img = get_image_data(image_id, 'Train')
    
#     # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img)
    
    edge_scharr = scharr_hsv(img)
    edge_canny = canny_hsv(img)

    
    plt.subplot(nImg,8,q*8-7)
    plt.imshow(img, cmap='binary')
    plt.title('Original Image')
    
    plt.subplot(nImg,8,q*8-6)
    plt.imshow(img, cmap='binary')
    plt.title('Image Mask')
    
    plt.subplot(nImg,8,q*8-5)    
    plt.imshow(img_rescale, cmap='binary')
    plt.title('Contrast stretching')
    
    plt.subplot(nImg,8,q*8-4)
    plt.imshow(img_eq, cmap='binary')
    plt.title('Equalization')
    
    plt.subplot(nImg,8,q*8-3)
    plt.imshow(img_adapteq, cmap='binary')
    plt.title('Adaptive Equalization')
    
    plt.subplot(nImg,8,q*8-2)
    plt.imshow(edge_scharr, cmap='binary')
    plt.title('Scharr Edge Magnitude')
    
    plt.subplot(nImg,8,q*8-1)
    plt.imshow(edge_canny, cmap='binary')
    plt.title('Canny features')
    j = j + 1

plt.show()

# %% {"_uuid": "2154b1c5e56d5459e8d52c5a4d46e874faeab315"}
