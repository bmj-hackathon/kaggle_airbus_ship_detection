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


#%%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys
path_image_class = Path().cwd() / '..' / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
print(sys.path)
from eda_00 import Image, convert_rgb_img_to_b64string


#%%%%%%%%%%%% LOAD 1 IMAGE INSTANCE
image_id = df_by_image.index[2] # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df)
image.load_ships()
r = image.records

image.ship_summary_table()
kmeans = image.k_means()

jpg_data = convert_rgb_img_to_b64string(image.img)


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
        dhtml.Img(src="data:image/png;base64, {}".format(jpg_data))
    ])
    # dhtml.Div(InteractiveImage('image', 'dash_app.png'), className='six columns'),

])

if __name__ == '__main__':
    app.run_server(debug=True)

