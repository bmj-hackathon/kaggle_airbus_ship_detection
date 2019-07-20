# import the necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import imutils
from imutils.mj_paper import PAPER

import numpy as np
import argparse
import cv2
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import random
import pandas as pd
from pathlib import Path
import zipfile
#%%
import logging
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

#%%
# data_path = Path("/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/ASSETS/Dogs vs Cats")
# image_path = data_path / '12499.jpg'
# image_path = data_path / '12500.jpg'
data_path = Path("/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/DATA/airbus-ship-detection")
assert data_path.exists()
img_zip_path = data_path / 'train_v2.zip'
assert img_zip_path.exists()
record_path = data_path / 'train_ship_segmentations_v2.csv'
assert record_path.exists()

img_zip = zipfile.ZipFile(img_zip_path)





logging.info("{} with {} files".format(img_zip_path.name, len(img_zip.filelist) ))

# train_files = pd.Series([zf.filename for zf in img_zip.filelist])
# df.index.intersection(train_files)

#%%
df = pd.read_csv(record_path)
# df = df.set_index('ImageId')
df.head()

logging.info("{} with {} records".format(record_path.name, len(df)))
logging.info("{} unique file names".format(df['ImageId'].unique().shape[0]))
logging.info("{} records with mask information (ships)".format(df['EncodedPixels'].count()))
# Flag if the record has a mask entry
df['HasShip'] = df['EncodedPixels'].notnull()
# Flag if the record is NOT unique
df['Duplicated'] = df['ImageId'].duplicated()
df['Unique'] = df['Duplicated']==False
logging.info("{} records with mask information (ship)".format(df['HasShip'].value_counts()[True]))

df['ImageId'][df['ImageId'].is_unique]




df['HasShip'] & df['Unique']

df['EncodedPixels'].unique()

df['EncodedPixels'].isnull()==False

df.columns
df['EncodedPixels']
df['Shape'] = df.iloc[0,:].index
img.shape

#%%
# Get 10 files
fnames = [zf.filename for zf in random.sample(img_zip.filelist, 10)]
this_file = random.choice(img_zip.filelist).filename

img = imutils.load_rgb_from_zip(img_zip, this_file)



plt.imshow(img)
plt.show()

#%%
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def plot_hist(img, ax):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        # ax.xlim([0, 256])


# def plot_hist(img, ax):
#     color = ('r', 'g', 'b')
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         plt.plot(histr, color=col)
#         plt.xlim([0, 256])
#     plt.show()
#




#%%
# The image is stored as a np.ndarray
# Attributes"
# .shape - x, y, z
# .min 0
# .max 255
#
# img_original = imutils.open_rgb(image_path)

plt.figure("TITLE")
plt.imshow(img)
plt.show()

plot_hist(img)
# hist = extract_color_histogram(img_original)

#%%
size=(32, 32)
img2 = cv2.resize(img, size)
plt.imshow(img2)
plt.show()

# plot_hist(img)

#%%


fig = plt.figure(figsize=PAPER['A3_LANDSCAPE'], facecolor='white')
fig.suptitle("Test {}".format('TEst'), fontsize=20)

nrows = 2
nrowplots = nrows * 2
ncols = 3
height_ratios = [3,1] * nrows
width_ratios = [1] * ncols
major_rows = [(r, r+1) for r in np.arange(0,nrows * 2,2)]
n_imgs = nrows * ncols
random.seed(42)
fnames = [zf.filename for zf in random.sample(img_zip.filelist, n_imgs)]
imgs = [imutils.load_rgb_from_zip(img_zip, this_file) for this_file in fnames]

fig, axes = plt.subplots(nrowplots, ncols,
                         gridspec_kw={'width_ratios': width_ratios, 'height_ratios':height_ratios })

for icol in range(axes.shape[1]):
    for imain, ihist in major_rows:
        print("Main:",imain,col)
        print("Hist:",ihist,col)
        this_img = imgs.pop(0)

        ax_main = axes[imain, icol]
        print(ax_main)
        ax_main.imshow(this_img)
        ax_main.get_xaxis().set_visible(False)
        ax_main.get_yaxis().set_visible(False)

        ax_hist = axes[ihist, icol]
        plot_hist(this_img, ax_hist)



plt.show()

#%%

for i, img_path in enumerate(sel_img_paths):
    logging.info("{}".format(img_path))
    ax_main = fig.add_subplot(ROWS, COLS, i + 1)
    img = mpl.image.imread(img_path)
    ax_main.imshow(img)
    ax_main.axis('off')
    # plt.title(str_label)
plt.show()

#%%
img2.flatten()

#%%

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax_main = plt.subplots()
ax_main.plot(t, s)

ax_main.set(xlabel='time (s)', ylabel='voltage (mV)',
            title='About as simple as it gets, folks')
ax_main.grid()

fig.savefig("test.png")
plt.show()