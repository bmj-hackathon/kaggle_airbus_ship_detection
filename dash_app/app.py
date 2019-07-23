import dash
print(dash.__version__)
import dash_core_components as dcc
import dash_html_components as dhtml

#%%%%%%%%%%%% LOGGING
import logging
import sys
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# Create formatter
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

#%%%%%%%%%%%% IMPORTS
# import the necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import imutils
from imutils.mj_paper import PAPER

import numpy as np
import argparse
import cv2
import os

import dash

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from mpl_toolkits.mplot3d import Axes3D

import sklearn as sk
import sklearn.cluster

import seaborn as sns
sns.set()

import random
import pandas as pd
from pathlib import Path
import zipfile

#%%%%%%%%%%%% CLASSES
#%%
def plot_hist(img, ax):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

#%% Get all masks given an image ID

def get_ellipsed_images(image_id):
    img = imutils.load_rgb_from_zip(img_zip, image_id)
    logging.info("Loaded {}, size {} with {} ships".format(image_id, img.shape, df_by_image.loc[image_id]['TotalShips']))

    records = df.loc[image_id]

    # Enforce return of dataframe as selection
    if type(records) == pd.core.series.Series:
        records = pd.DataFrame(records)
        records = records.T

    assert len(records) == df_by_image.loc[image_id]['TotalShips']

    # Iterate over each record
    contours = list()
    cnt=0
    for i, rec in records.iterrows():
        cnt+=1
        logging.debug("Processing record {} of {}".format(cnt, image_id))
        mask = imutils.convert_rle_mask(rec['EncodedPixels'])
        contour = imutils.get_contour(mask)
        contours.append(contour)
        # img = imutils.draw_ellipse_and_axis(img, contour, thickness=2)
        img = imutils.fit_draw_ellipse(img, contour, thickness=2)
    return img, contours


#%%%%%%%%%%%% LOAD
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

logging.info("Image data: '{}' loaded from {} with {} files".format('img_zip', img_zip_path.name, len(img_zip.filelist) ))

#%%
df = pd.read_csv(record_path)

logging.info("{} with {} records".format(record_path.name, len(df)))
logging.info("{} unique file names found in df".format(df['ImageId'].unique().shape[0]))
# Flag if the record has a mask entry
df['HasShip'] = df['EncodedPixels'].notnull()
# Flag if the record is NOT unique
df['Duplicated'] = df['ImageId'].duplicated()
df['Unique'] = df['Duplicated']==False

logging.info("{} records with mask information (ship)".format(df['HasShip'].value_counts()[True]))
logging.info("{} images have at least one ship".format(sum(df['HasShip'] & df['Unique'])))

df_by_image = df.groupby('ImageId').agg({'HasShip': ['first', 'sum']})
df_by_image.columns = ['HasShip', 'TotalShips']
df_by_image.sort_values('TotalShips', ascending=False, inplace=True)
df = df.set_index('ImageId')
df_sample = df.head()


#%%%%%%%%%%%% EDA

class Image():
    def __init__(self, image_id):
        """

        :param image_id:

        Attributes:
            image_id    The ID string
            img         The image as an ndarray
            records     DataFrame of records from the original CSV file
            encoding    A string representing the OpenCV encoding of the underlying img ndarray
            ships       A list of Ship dictionary entries
                ship_id         - Hash of the RLE string
                EncodedPixels   - RLE string
                center          -
        """

        self.image_id = image_id
        self.encoding = None
        self.records = None
        self.img = None
        self.contours = None

        logging.info("Image id: {}".format(self.image_id))

    def __str__(self):
        return "Image ID {} {} encoded, with {} ships".format(self.image_id, self.encoding, self.num_ships)

    @property
    def num_ships(self):
        return len(self.records)

    @property
    def shape(self):
        return self.img.shape

    @property
    def shape2D(self):
        return self.img.shape[0:2]

    def get_img_bgr(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    def load(self, image_zip, df):
        """load an image into ndarray as RGB, and load ship records

        :param image_zip:
        :param df:
        :return:
        """

        self.img = imutils.load_rgb_from_zip(image_zip, image_id)
        self.encoding = 'RGB'
        logging.info("Loaded {}, size {} ".format(image_id, self.img.shape))

        self.records = df[df.index == self.image_id]
        assert isinstance(self.records, pd.DataFrame)

        # TODO: check warning
        self.records['ship_id'] = self.records.apply(lambda row: hash(row['EncodedPixels']), axis=1)
        self.records.set_index('ship_id', inplace=True)
        self.records.drop(['HasShip', 'Duplicated', 'Unique'], axis=1, inplace=True)

        logging.info("{} records selected for {}".format(len(self.records), self.image_id))

    def moments(self):
        """ Just a docstring for now
            // spatial moments
    double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    // central moments
    double  mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    // central normalized moments
    double  nu20, nu11, nu02, nu30, nu21, nu12, nu03;
        :return:
        """

    def load_ships(self):
        """Augment the basic df with mask, contour, data

        mask        - ndarray of 0 or 1
        contour     - opencv2 contour object
        moments     -

        :return:
        """
        assert isinstance(self.img, np.ndarray), "No image loaded"
        assert self.num_ships, "No ships in this image"

        # TODO: check warnings
        self.records['mask'] = self.records.apply(lambda row: self.convert_rle_to_mask(row['EncodedPixels'], self.shape2D), axis=1)
        self.records['contour'] = self.records.apply(lambda row: self.get_contour(row['mask']), axis=1)
        self.records['moments'] = self.records.apply(lambda row: cv2.moments(row['contour']), axis=1)

        # def get_x(row): return round(row['moments']['m10'] / row['moments']['m00'])
        def get_x(row): return row['moments']['m10'] / row['moments']['m00']
        # def get_y(row): return round(row['moments']['m01'] / row['moments']['m00'])
        def get_y(row): return row['moments']['m01'] / row['moments']['m00']
        self.records['x'] = self.records.apply(lambda row: get_x(row), axis=1)
        self.records['y'] = self.records.apply(lambda row: get_y(row), axis=1)

        # ( Same as m00!)
        self.records['area'] = self.records.apply(lambda row: cv2.contourArea(row['contour']), axis=1)
        self.records['rotated_rect'] = self.records.apply(lambda row: cv2.minAreaRect(row['contour']), axis=1)
        self.records['angle'] = self.records.apply(lambda row: row['rotated_rect'][2], axis=1)
        # print(self.records['rotated_rect'])
        # print(type(self.records['rotated_rect'].iloc[0]))


        # self.records['area'] = int()
        # self.records['rotated_rect'] = cv2.fitEllipse(c)

        self.records.drop(['mask', 'contour', 'moments', 'rotated_rect', 'EncodedPixels'], axis=1, inplace=True)

    def ship_summary_table(self):
        return self.records.round(1)

    def draw_ellipses_to_canvas(self):
        img = imutils.fit_draw_ellipse(self.img, contour, thickness=2)

    def convert_rle_to_mask(self, rle, shape):
        """convert RLE mask into 2d pixel array"""

        # Initialize a zero canvas (one-dimensional here)
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        # Split each run-length string
        s = rle.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            mask[start:start + length] = 1 # Assign this run to ones
        # Reshape to 2D
        img2 = mask.reshape(shape).T
        return img2

    def get_contour(self, mask):
        """Return a cv2 contour object from a binary 0/1 mask"""

        assert mask.ndim == 2
        assert mask.min() == 0
        assert mask.max() == 1
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 1, "Too many contours in this mask!"
        contour = contours[0]
        # logging.debug("Returning {} fit contours over mask pixels".format(len(contours)))
        return contour

    def k_means(self, num_clusters=2):
        logging.info("Processing {} image of shape {}".format(self.encoding, self.img.shape))
        data = self.img / 255
        logging.info("Scaled values to 0-1 range".format())
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        logging.info("Reshape to pixel list {}".format(data.shape))

        kmeans = sk.cluster.MiniBatchKMeans(2)
        kmeans.fit(data)
        logging.info("Fit {} pixels {} clusters".format(data.shape[0], num_clusters))
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        for c_name, c_count, c_position in zip(unique, counts, kmeans.cluster_centers_):
            logging.info("\tCluster {} at {} with {:0.1%} of the pixels".format(c_name, np.around(c_position, 3), c_count/data.shape[0])),

        if len(unique) == 2:
            dist = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
            logging.debug("Distance between c1 and c2: {}".format(dist))
        return kmeans
        # all_new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

# from server import server
import importlib.util
path_image_class = Path().cwd() / 'src' / '3_EDA'
assert path_image_class.exists(), "Can't find {}".format(path_image_class)
spec = importlib.util.spec_from_file_location("eda_00", path_image_class)


#%%%%%%%%%%%% LOAD 1 IMAGE
image_id = df_by_image.index[2] # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df)
image.load_ships()
r = image.records

image.ship_summary_table()
kmeans = image.k_means()

# Convert the in-memory image to .jpg format
import base64
retval, buffer = cv2.imencode('.jpg', image.get_img_bgr())
# Convert to base64 string
jpg_as_text = base64.b64encode(buffer)
# jpg_as_text = str(jpg_as_text)
jpg_as_text = jpg_as_text.decode()
# print(jpg_as_text)
logging.info("Image encoded to jpg base64 string".format())
print(jpg_as_text)

# cv2.imencode('.jpg', image.img)[1].tostring('base64')
# jpg_img.toString('base64')
# cv2
# cv2.imwrite(image.image_id, jpg_img)
# this_image_path = Path(image.image_id)
# assert this_image_path.exists()
# logging.info("Saved image to disk: {}".format(this_image_path.absolute()))
#
# encoded_image = base64.b64encode()


#%%%%%%%%%%%% DASH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dhtml.Div(children=[
    dhtml.H1(children='Hello Test1'),
    dhtml.Div(children='''
        Dash: Test app number 1 ... !
    '''),
    dcc.Graph(
        id='example1',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Test title 12'
            }
        }
    ),
    dhtml.H1(children="Image {}".format(image.image_id)),
    dhtml.H3(children='TEST H3'),
    dhtml.Div(children=[
        dhtml.H2(children="Image {}".format(image.image_id)),

        # dhtml.Img(src="data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAAUA AAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO 9TXL0Y4OHwAAAABJRU5ErkJggg=="),
        dhtml.Img(src="data:image/png;base64, {}".format(jpg_as_text))
    ])
    # dhtml.Div(InteractiveImage('image', 'dash_app.png'), className='six columns'),

])

if __name__ == '__main__':
    app.run_server(debug=True)

