import dash

print(dash.__version__)
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table

# %%%%%%%%%%%% LOGGING
import logging
import sys

logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

# Create formatter
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

# %%%%%%%%%%%% IMPORTS
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


# %%%%%%%%%%%% CLASSES
# %%
def plot_hist(img, ax):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)


# %% Get all masks given an image ID

def get_ellipsed_images(image_id):
    img = imutils.load_rgb_from_zip(img_zip, image_id)
    logging.info(
        "Loaded {}, size {} with {} ships".format(image_id, img.shape, df_by_image.loc[image_id]['TotalShips']))

    records = df.loc[image_id]

    # Enforce return of dataframe as selection
    if type(records) == pd.core.series.Series:
        records = pd.DataFrame(records)
        records = records.T

    assert len(records) == df_by_image.loc[image_id]['TotalShips']

    # Iterate over each record
    contours = list()
    cnt = 0
    for i, rec in records.iterrows():
        cnt += 1
        logging.debug("Processing record {} of {}".format(cnt, image_id))
        mask = imutils.convert_rle_mask(rec['EncodedPixels'])
        contour = imutils.get_contour(mask)
        contours.append(contour)
        # img = imutils.draw_ellipse_and_axis(img, contour, thickness=2)
        img = imutils.fit_draw_ellipse(img, contour, thickness=2)
    return img, contours


# %%%%%%%%%%%% LOAD
# %%
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

logging.info(
    "Image data: '{}' loaded from {} with {} files".format('img_zip', img_zip_path.name, len(img_zip.filelist)))

# %%
df = pd.read_csv(record_path)

logging.info("{} with {} records".format(record_path.name, len(df)))
logging.info("{} unique file names found in df".format(df['ImageId'].unique().shape[0]))
# Flag if the record has a mask entry
df['HasShip'] = df['EncodedPixels'].notnull()
# Flag if the record is NOT unique
df['Duplicated'] = df['ImageId'].duplicated()
df['Unique'] = df['Duplicated'] == False

logging.info("{} records with mask information (ship)".format(df['HasShip'].value_counts()[True]))
logging.info("{} images have at least one ship".format(sum(df['HasShip'] & df['Unique'])))

df_by_image = df.groupby('ImageId').agg({'HasShip': ['first', 'sum']})
df_by_image.columns = ['HasShip', 'TotalShips']
df_by_image.sort_values('TotalShips', ascending=False, inplace=True)
df = df.set_index('ImageId')
df_sample = df.head()

# %%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys

path_image_class = Path().cwd() / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
print(sys.path)
from eda_00 import Image, convert_rgb_img_to_b64string

# %%%%%%%%%%%% LOAD 1 IMAGE INSTANCE
image_id = df_by_image.index[2]  # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df)
image.load_ships()

# %%%%%%%%%%%% Perform kmeans
# kmeans = image.k_means()

# %%%%%%%%%%%% Build summary table
df_ships = image.ship_summary_table()

# %%%%%%%%%%%% Get base image
jpg_base_image = convert_rgb_img_to_b64string(image.img)

# %%%%%%%%%%%% Get ellipse image
ndarray_ellipse_image = image.draw_ellipses_img()
jpg_ellipse_image = convert_rgb_img_to_b64string(ndarray_ellipse_image)

# %%%%%%%%%%%% DASH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
assets_url_path = Path.cwd() / 'dash_app' / 'assets'
assert assets_url_path.exists()
app = dash.Dash(__name__, assets_url_path=str(assets_url_path), external_stylesheets=external_stylesheets)

MAX_SHIPS = 15

app.layout = html.Div(children=[

    # Main title
    html.H1(children='Satellite Data visualization', className='title'),
    # Sub-text
    # dhtml.Div(children='''
    #     Dash: Test app number 1 ... !
    # '''),

    html.Div([
        html.H2("Ship selection"),
        html.H4("Select number of ships to filter on:"),
        dcc.Slider(
            id='slider-ship-num',
            className='slider',
            min=0,
            max=15,
            step=1,
            value=5,
            marks={n: '{}'.format(n) for n in range(MAX_SHIPS+1)},
        ),
        html.P("Selected:"),
        html.Div(id='slider-output-container'),

        html.P("10 images from the filter are randomly selected."),

        html.P("Select the image for analysis below:"),

        dcc.Slider(
            id='slider-ship-id',
            className='slider',
            min=1,
            max=10,
            step=1,
            # value=1,
            # marks={n: '{}'.format(n) for n in range(10)},
        ),

        # dcc.Dropdown(
        #     id='my-dropdown',
        #     options=[
        #         {'label': 'New York City', 'value': 'NYC'},
        #         {'label': 'Montreal', 'value': 'MTL'},
        #         {'label': 'San Francisco', 'value': 'SF'}
        #     ],
        #     value=image.image_id
        # ),
        html.Div(id='output-container'),

        html.Div(id='output-container2'),

    ], className="section-container"),

    # html.H1(children="Image {}".format(image.image_id)),
    html.H1("Selected image:"),
    html.H1(id='image_id'),
    html.H3(id='start-status'),

    # 2 columns, data table | base ellipse image
    html.Div([
        html.Div([
            html.H3('Ship data'),
            dash_table.DataTable(
                id='ship-data-table',
                columns=[{"name": i, "id": i} for i in df_ships.columns],
            ),
        ], className="six columns"),
        html.Div([
            html.H3('Image'),
            html.Img(id='base-ship-image')
        ], className="six columns")
    ], className="row"),

    html.H3(children='TEST H3'),
    html.Div(children=[
        html.H2(children="image {}".format(image.image_id)),
    ]),

    # html.H1(children="Image {}".format(image.image_id)),

    # 2 columns, data table | base ellipse image
    html.Div([
        html.Div([
            html.H3('Ship data OLD'),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df_ships.columns],
                data=df_ships.to_dict('records'),
            ),
        ], className="six columns"),
        html.Div([
            html.H3('Image'),
            html.Img(src="data:image/png;base64, {}".format(jpg_ellipse_image))
        ], className="six columns")
    ], className="row"),

    html.H3(children='TEST H3'),
    html.Div(children=[
        html.H2(children="image {}".format(image.image_id)),
    ]),

])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


#%%-----------------
# Get the ship number
#--------------------
# TODO: NOTE: The return values are mapped 1:1 in the list of Outputs!
@app.callback([
    dash.dependencies.Output('slider-ship-id', 'marks'),
    dash.dependencies.Output('slider-output-container', 'children'),
],
    [dash.dependencies.Input('slider-ship-num', 'value')]
)  # END DECORATOR
def update_output(value):
    images = df_by_image.loc[df_by_image['TotalShips'] == value].index.to_series()
    images_sample = images.sample(10)

    # dropdown_options = [{'label':id, 'value':id} for id in images.sample(10)]
    # dropdown_options = [{'label': id, 'value': id} for id in images.sample(10)]

    slider_marks = {i+1 : '{}'.format(img_id.split('.')[0]) for i, img_id in enumerate(images_sample)}
    slider_marks = {img_id : '{}'.format(img_id.split('.')[0]) for i, img_id in enumerate(images_sample)}
    # slider_marks = {n: '{}'.format(n) for n in range(10)}
    print(slider_marks)
    return slider_marks, images_sample[0], len(images)
    # return "{} images have {} ships".format(len(images), value)


#%%-----------------
# Get the image ID from the slider
#--------------------
@app.callback(
    dash.dependencies.Output('image_id', 'children'),
    [dash.dependencies.Input('slider-ship-id', 'value')]
)  # END DECORATOR
def update_image_id(value):
    return value


#%%-----------------
# Build the summary table and the image
#--------------------
@app.callback(
    [
        dash.dependencies.Output('ship-data-table', 'data'),
        dash.dependencies.Output('base-ship-image', 'src'),
    ],
    [dash.dependencies.Input('image_id', 'children')]
)
def get_image_data(image_id_index_number):
    # Instantiate the image object
    print("Getting image number {}".format(image_id_index_number))
    image_id = df.iloc[image_id_index_number, :].index
    print("image_id=",image_id)
    image = Image(image_id)
    image.load(img_zip, df)
    image.load_ships()

    # Build summary table
    df_ships = image.ship_summary_table()

    # Get ellipse image
    ndarray_ellipse_image = image.draw_ellipses_img()
    jpg_ellipse_image = convert_rgb_img_to_b64string(ndarray_ellipse_image)
    image_source_string = "data:image/png;base64, {}".format(jpg_ellipse_image)
    data = df_ships.to_dict('records')
    return data, image_source_string


if __name__ == '__main__':
    app.run_server(debug=True)
