import dash

print(dash.__version__)
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
import plotly.graph_objs as go
import plotly.express as px
# from helpers import make_dash_table, create_plot

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

# %%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys

path_image_class = Path().cwd() / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
print(sys.path)
from eda_00_Image_class import Image, convert_rgb_img_to_b64string, fit_kmeans_pixels, convert_rgb_img_to_b64string_straight

#%%
try:
    from callbacks import register_callbacks
    from figures import test_fig
except:
    sys.path.append( str(Path.cwd().joinpath('dash_app')) )
    from callbacks import register_callbacks
    from figures import test_fig

# %% General, entry point
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
assets_url_path = Path.cwd() / 'dash_app' / 'assets'

assert assets_url_path.exists()
app = dash.Dash(__name__, assets_url_path=str(assets_url_path), external_stylesheets=external_stylesheets)

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# %%%%%%%%%%%% LOAD
data_path = Path("/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/DATA/airbus-ship-detection")
assert data_path.exists()
img_zip_path = data_path / 'train_v2.zip'
assert img_zip_path.exists()
record_path = data_path / 'train_ship_segmentations_v2.csv'
assert record_path.exists()

img_zip = zipfile.ZipFile(img_zip_path)

logging.info(
    "Image data: '{}' loaded from {} with {} files".format('img_zip', img_zip_path.name, len(img_zip.filelist)))

# %% LOAD CSV
df_test = pd.read_csv(record_path)

logging.info("{} with {} records".format(record_path.name, len(df_test)))
logging.info("{} unique file names found in df".format(df_test['ImageId'].unique().shape[0]))
# Flag if the record has a mask entry
df_test['HasShip'] = df_test['EncodedPixels'].notnull()
# Flag if the record is NOT unique
df_test['Duplicated'] = df_test['ImageId'].duplicated()
df_test['Unique'] = df_test['Duplicated'] == False

logging.info("{} records with mask information (ship)".format(df_test['HasShip'].value_counts()[True]))
logging.info("{} images have at least one ship".format(sum(df_test['HasShip'] & df_test['Unique'])))

df_by_image = df_test.groupby('ImageId').agg({'HasShip': ['first', 'sum']})
df_by_image.columns = ['HasShip', 'TotalShips']
df_by_image.sort_values('TotalShips', ascending=False, inplace=True)
df_test = df_test.set_index('ImageId')
df_test['index_number'] = range(0, len(df_test))
df_sample = df_test.head()

# %%%%%%%%%%%% LOAD 1 IMAGE INSTANCE
image_id = df_by_image.index[2]  # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df_test)
image.load_ships()

# %% KMeans

kmeans = image.k_means(num_clusters=2)

kmeans_img = fit_kmeans_pixels(image.img, kmeans)

# Build an image HTML object
kmeans_img_str = convert_rgb_img_to_b64string_straight(kmeans_img * 255)
image_source_string = "data:image/png;base64, {}".format(kmeans_img_str)

html_kmeans_img_STATIC = html.Img(src=image_source_string)

N_points = 20000

# Generate a list of 20000 indices
kmeans_img_flat = kmeans_img.reshape(kmeans_img.shape[0]*kmeans_img.shape[1],kmeans_img.shape[2])
rng = np.random.RandomState(0)
i = rng.permutation(kmeans_img_flat.shape[0])[:N_points]
colors_i = kmeans_img_flat[i]
labels_i = kmeans.labels_[i]
R, G, B = kmeans_img_flat[i].T
#
#
fig_kmeans_scatter = go.Figure(data=[go.Scatter3d(x=R, y=G, z=B, mode='markers')])
fig_kmeans_scatter.update_layout(scene = dict(
                    xaxis_title='R',
                    yaxis_title='G',
                    zaxis_title='B'),
                    # width=700,
                    margin=dict(r=20, b=10, l=10, t=10),
                    height=500,
)

fig_kmeans_scatter.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(255, 220, 220)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(220, 255, 220)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(220, 220, 255)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )
# html_fig_kmeans_scatter = html.Div([dcc.Graph(
#     id='basic-interactions',
#     figure=fig_kmeans_scatter,
# )]),

#raise




# %%%%%%%%%%%% DASH
MAX_SHIPS = 15
DOM = list()

#%% Section:  Main title
DOM.append(html.H1(children='Satellite Data visualization', className='title'))

#%% Section: Ship number select
DOM.append(
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
            marks={n: '{}'.format(n) for n in range(MAX_SHIPS + 1)},
        ),
        html.P([
            html.Span("Found "),
            html.Span(id='text-ship-count'),
            html.Span(" images with "),
            html.Span(id='text-ship-count2'),
            html.Span(" ships.")
        ]),

        html.Button('Get random image', id='button-get-random'),
        html.Div(id='container-button-basic',
                 children='Enter a value and press submit'),

        html.Div(id='output-container'),

        html.Div(id='output-container2'),

    ], className="section-container")
)

#%% Section: Summary table and image
DOM.append(html.H3(children=[html.Span("Selected image:"),html.Span(id='image_id')]))
DOM.append(
    html.Div([
        html.Div([
            html.H3('Ship data'),
            dash_table.DataTable(
                id='ship-data-table',
                # columns=[{"name": i, "id": i} for i in df_ships.columns],
            ),
        ], className="six columns"),
        html.Div([
            html.H3('Image'),
            html.Img(id='base-ship-image')
        ], className="six columns")
    ], className="row")
)

# Section: Kmeans Cluster Select
DOM.append(
    html.Div([
        html.H2("K-Means segmentation "),
        html.H4("Select number of clusters:"),
        dcc.Slider(
            id='slider-cluster-counts',
            className='slider',
            min=0,
            max=10,
            step=1,
            value=2,
            marks={n: '{}'.format(n) for n in range(10 + 1)},
        ),
        html.P("", id='empty-para'),
        html.Button('Perform KMeans clustering', id='button-start-kmeans'),
    ], className="section-container")
)


#%% Kmeans
app.layout = html.Div(children=DOM + [

    html_kmeans_img_STATIC,
    # html_fig_kmeans_scatter,

    html.Div(children=[dcc.Graph(
        id='kmeans-scatter',
        figure=fig_kmeans_scatter,
    )], style={'display': 'inline-block', 'width': '100%', 'height': '80%'}),
])


register_callbacks(app, df_test, df_by_image, img_zip, Image)

if __name__ == '__main__':
    app.run_server(debug=True)


