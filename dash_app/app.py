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


# %%%%%%%%%%%% CLASSES
# %%
def plot_hist(img, ax):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)


# %%

def create_plot(x, y, z, size, color, name, xlabel="LogP", ylabel="pkA", zlabel="Solubility (mg/ml)",
                plot_type="scatter3d", markers=[], ):
    colorscale = [
        [0, "rgb(244,236,21)"],
        [0.3, "rgb(249,210,41)"],
        [0.4, "rgb(134,191,118)"],
        [0.5, "rgb(37,180,167)"],
        [0.65, "rgb(17,123,215)"],
        [1, "rgb(54,50,153)"],
    ]

    data = [
        {
            "x": x,
            "y": y,
            "z": z,
            "mode": "markers",
            "marker": {
                "colorscale": colorscale,
                "colorbar": {"title": "Molecular<br>Weight"},
                "line": {"color": "#444"},
                "reversescale": True,
                "sizeref": 45,
                "sizemode": "diameter",
                "opacity": 0.7,
                "size": size,
                "color": color,
            },
            "text": name,
            "type": plot_type,
        }
    ]

    if plot_type in ["histogram2d", "scatter"]:
        del data[0]["z"]

    if plot_type == "histogram2d":
        # Scatter plot overlay on 2d Histogram
        data[0]["type"] = "scatter"
        data.append(
            {
                "x": x,
                "y": y,
                "type": "histogram2d",
                "colorscale": "Greys",
                "showscale": False,
            }
        )

    layout = _create_layout(plot_type, xlabel, ylabel)

    if len(markers) > 0:
        data = data + _add_markers(data, markers, plot_type=plot_type)

    return {"data": data, "layout": layout}


def _create_axis(axis_type, variation="Linear", title=None):
    """
    Creates a 2d or 3d axis.
    :params axis_type: 2d or 3d axis
    :params variation: axis type (log, line, linear, etc)
    :parmas title: axis title
    :returns: plotly axis dictionnary
    """

    if axis_type not in ["3d", "2d"]:
        return None

    default_style = {
        "background": "rgb(230, 230, 230)",
        "gridcolor": "rgb(255, 255, 255)",
        "zerolinecolor": "rgb(255, 255, 255)",
    }

    if axis_type == "3d":
        return {
            "showbackground": True,
            "backgroundcolor": default_style["background"],
            "gridcolor": default_style["gridcolor"],
            "title": title,
            "type": variation,
            "zerolinecolor": default_style["zerolinecolor"],
        }

    if axis_type == "2d":
        return {
            "xgap": 10,
            "ygap": 10,
            "backgroundcolor": default_style["background"],
            "gridcolor": default_style["gridcolor"],
            "title": title,
            "zerolinecolor": default_style["zerolinecolor"],
            "color": "#444",
        }


def _black_out_axis(axis):
    axis["showgrid"] = False
    axis["zeroline"] = False
    axis["color"] = "white"
    return axis


def _create_layout(layout_type, xlabel, ylabel):
    """ Return dash plot layout. """

    base_layout = {
        "font": {"family": "Raleway"},
        "hovermode": "closest",
        "margin": {"r": 20, "t": 0, "l": 0, "b": 0},
        "showlegend": False,
    }

    if layout_type == "scatter3d":
        base_layout["scene"] = {
            "xaxis": _create_axis(axis_type="3d", title=xlabel),
            "yaxis": _create_axis(axis_type="3d", title=ylabel),
            "zaxis": _create_axis(axis_type="3d", title=xlabel, variation="log"),
            "camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 0.08, "y": 2.2, "z": 0.08},
            },
        }

    elif layout_type == "histogram2d":
        base_layout["xaxis"] = _black_out_axis(
            _create_axis(axis_type="2d", title=xlabel)
        )
        base_layout["yaxis"] = _black_out_axis(
            _create_axis(axis_type="2d", title=ylabel)
        )
        base_layout["plot_bgcolor"] = "black"
        base_layout["paper_bgcolor"] = "black"
        base_layout["font"]["color"] = "white"

    elif layout_type == "scatter":
        base_layout["xaxis"] = _create_axis(axis_type="2d", title=xlabel)
        base_layout["yaxis"] = _create_axis(axis_type="2d", title=ylabel)
        base_layout["plot_bgcolor"] = "rgb(230, 230, 230)"
        base_layout["paper_bgcolor"] = "rgb(230, 230, 230)"

    return base_layout


# %% Get all masks given an image ID

def get_ellipsed_images(image_id):
    img = imutils.load_rgb_from_zip(img_zip, image_id)
    logging.info(
        "Loaded {}, size {} with {} ships".format(image_id, img.shape, df_by_image.loc[image_id]['TotalShips']))

    records = df_test.loc[image_id]

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

# %%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys

path_image_class = Path().cwd() / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
print(sys.path)
from eda_00_Image_class import Image, convert_rgb_img_to_b64string

# %%%%%%%%%%%% LOAD 1 IMAGE INSTANCE
image_id = df_by_image.index[2]  # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df_test)
image.load_ships()

# %%%% TEST GRAPHING
# FIGURE = create_plot(
#     x=df_test["PKA"],
#     y=df_test["LOGP"],
#     z=df_test["SOL"],
#     size=df_test["MW"],
#     color=df_test["MW"],
#     name=df_test["NAME"],
# )


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

# TODO: TESTING DF HERE
df_us_ag = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv")

app.layout = html.Div(children=[

    # Main title
    html.H1(children='Satellite Data visualization', className='title'),
    # Sub-text
    # dhtml.Div(children='''
    #     Dash: Test app number 1 ... !
    # '''),
    ###### SECTION: Ship selection
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
        # html.P(),
        # html.Div(,

        # html.P("10 images from the filter are randomly selected in the dropdown."),
        #
        # dcc.Dropdown(
        #     id='dropdown-ship-id',
        #     options=[
        #         {'label': 'New York City', 'value': 'NYC'},
        #         {'label': 'Montreal', 'value': 'MTL'},
        #         {'label': 'San Francisco', 'value': 'SF'}
        #     ],
        #     value=image.image_id
        # ),

        # html.P(''),
        # html.P("Alternatively, select an image from the filtered list at random"),

        html.Button('Get random image', id='button-get-random'),
        html.Div(id='container-button-basic',
                 children='Enter a value and press submit'),

        html.Div(id='output-container'),

        html.Div(id='output-container2'),

    ], className="section-container"),

    ###### SECTION: IMAGE DATA DISPLAY

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

    ###### SECTION: KMeans SETUP
    html.Hr(),
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
    ], className="section-container"),

    ###### SECTION: IMAGE DATA DISPLAY
    #
    # # html.H1(children="Image {}".format(image.image_id)),
    # html.H1("KMeans results"),
    # # html.H1(id='image_id'),
    # # html.H3(id='start-status'),
    # iris = px.data.iris()
    # fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
    #               color='species')
    # fig.show()

    # dcc.Graph(
    #     id="clickable-graph",
    #     hoverData={"points": [{"pointNumber": 0}]},
    #     figure=FIGURE,
    # ),

    ###### SECTION: TESTING GRAPHS
    html.Div([
        html.Div([html.H1("United States Exports")],
                 style={'textAlign': "center", "padding-bottom": "10", "padding-top": "10"}),
        html.Div(
            [html.Div(
                dcc.Dropdown(id="select-xaxis", options=[{'label': i.title(), 'value': i} for i in df_test.columns[3:]],
                             value='beef', ), className="four columns",
                style={"display": "block", "margin-left": "auto",
                       "margin-right": "auto", "width": "33%"}),
             html.Div(dcc.Dropdown(id="select-yaxis",
                                   options=[{'label': i.title(), 'value': i} for i in df_test.columns[3:]],
                                   value='pork', ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "33%"}),
             html.Div(dcc.Dropdown(id="select-zaxis",
                                   options=[{'label': i.title(), 'value': i} for i in df_test.columns[3:]],
                                   value='poultry', ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "33%"})
             ], className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),
        html.Div([dcc.Graph(id="my-graph")])
    ], className="container")

])


# %%-----------------
# Get the ship number
# --------------------

# TODO: NOTE: The return values are mapped 1:1 in the list of Outputs!
@app.callback(
    # dash.dependencies.Output('dropdown-ship-id', 'options'),
    [
        dash.dependencies.Output('text-ship-count', 'children'),
        dash.dependencies.Output('text-ship-count2', 'children'),
    ],
    [dash.dependencies.Input('slider-ship-num', 'value')]
)  # END DECORATOR
def update_output(value):
    images = df_by_image.loc[df_by_image['TotalShips'] == value].index.to_series()
    images_sample = images.sample(10)

    dropdown_options = [{'label': id, 'value': id} for id in images_sample]
    # dropdown_options = [{'label': id, 'value': id} for id in images.sample(10)]

    # slider_marks = {i+1 : '{}'.format(img_id.split('.')[0]) for i, img_id in enumerate(images_sample)}
    # slider_marks = {img_id : '{}'.format(img_id.split('.')[0]) for i, img_id in enumerate(images_sample)}
    # slider_marks = {n: '{}'.format(n) for n in range(10)}
    # print(slider_marks)
    # return dropdown_options, len(images)
    return len(images), value
    # return "{} images have {} ships".format(len(images), value)


# %%-----------------
# Get the image ID from the slider
# BROKEN
# --------------------
if 0:
    @app.callback(
        dash.dependencies.Output('image_id', 'children'),
        [dash.dependencies.Input('slider-ship-id', 'value')]
    )  # END DECORATOR
    def update_image_id_slider(value):
        return value


# %%-----------------
# Get a random image ID
# --------------------
@app.callback(
    dash.dependencies.Output('image_id', 'children'),
    [dash.dependencies.Input('button-get-random', 'n_clicks'),
     dash.dependencies.Input('slider-ship-num', 'value')]
)  # END DECORATOR
def button_random(n_clicks, value):
    images = df_by_image.loc[df_by_image['TotalShips'] == value]
    selected_id = images.sample().index[0]
    print("Selected {} from {} images (TotalShips = {})".format(selected_id, len(images), value))
    return selected_id


# %%-----------------
# Get the image ID from the dropdown
# --------------------
# @app.callback(
#     dash.dependencies.Output('image_id', 'children'),
#     [dash.dependencies.Input('dropdown-ship-id', 'value')]
# )  # END DECORATOR
# def update_image_id_dropdown(value):
#     return value

# %%-----------------
# Build the summary table and the image
# --------------------
@app.callback(
    [
        dash.dependencies.Output('ship-data-table', 'data'),
        dash.dependencies.Output('base-ship-image', 'src'),
    ],
    [dash.dependencies.Input('image_id', 'children')]
)
def get_image_data(image_id):
    # Instantiate the image object
    # print("Getting image number {}".format(image_id_index_number))
    # image_id = df.iloc[image_id_index_number, :].index
    # print("image_id=",image_id)
    image = Image(image_id)
    image.load(img_zip, df_test)
    print("GET IMAGE DATA,  num_ships = ", image.num_ships)
    if not image.num_ships:
        jpg_ellipse_image = convert_rgb_img_to_b64string(image.img)
        image_source_string = "data:image/png;base64, {}".format(jpg_ellipse_image)
        empty_data = [{'ship': 'None', 'index_number': 'n/a', 'x': 'n/a', 'y': 'n/a', 'area': 'n/a', 'angle': 'n/a'}]
        return empty_data, image_source_string

    image.load_ships()

    # Build summary table
    df_ships = image.ship_summary_table()

    # Get ellipse image
    ndarray_ellipse_image = image.draw_ellipses_img()
    jpg_ellipse_image = convert_rgb_img_to_b64string(ndarray_ellipse_image)
    image_source_string = "data:image/png;base64, {}".format(jpg_ellipse_image)
    data = df_ships.to_dict('records')
    # print(data)
    return data, image_source_string


# %%-----------------
# K Means button
# --------------------
if 0:
    @app.callback(
        dash.dependencies.Output('empty-para', 'children'),
        [dash.dependencies.Input('button-start-kmeans', 'n_clicks'),
         dash.dependencies.Input('slider-cluster-counts', 'value'),
         dash.dependencies.Input('image_id', 'children')]
    )  # END DECORATOR
    def button_random(n_clicks, n_clusters, image_id):
        print("K Means with {} clusters on image {}".format(n_clusters, image_id))

        image = Image(image_id)
        image.load(img_zip, df_test)
        kmeans = image.k_means(n_clusters)

        data = image.img / 255
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

        logging.info("Fit {} pixels {} clusters".format(data.shape[0], n_clusters))
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        # for c_name, c_count, c_position in zip(unique, counts, kmeans.cluster_centers_):
        #     logging.info("\tCluster {} at {} with {:0.1%} of the pixels".format(c_name, np.around(c_position, 3),
        #                                                                         c_count / data.shape[0])),
        #
        # if len(unique) == 2:
        #     dist = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
        #     logging.info("Distance between c1 and c2: {}".format(dist))

        # images = df_by_image.loc[df_by_image['TotalShips'] == value]
        # selected_id = images.sample().index[0]
        # print("Selected {} from {} images (TotalShips = {})".format(selected_id, len(images), value))
        # return selected_id


# %% TESTING GRAPH
# https://github.com/plotly/simple-example-chart-apps/blob/master/dash-3dscatterplot/apps/main.py
@app.callback(
    dash.dependencies.Output("my-graph", "figure"),
    [dash.dependencies.Input("select-xaxis", "value"),
     dash.dependencies.Input("select-yaxis", "value"),
     dash.dependencies.Input("select-zaxis", "value")]

)
def ugdate_figure(selected_x, selected_y, selected_z):
    z = df_us_ag[selected_z]
    trace = [go.Scatter3d(
        x=df_us_ag[selected_x], y=df_us_ag[selected_y], z=df_us_ag[selected_z],
        mode='markers', marker={'size': 8, 'color': z, 'colorscale': 'Blackbody', 'opacity': 0.8, "showscale": True,
                                "colorbar": {"thickness": 15, "len": 0.5, "x": 0.8, "y": 0.6, }, })]
    return {"data": trace,
            "layout": go.Layout(
                height=700, title=f"Exports<br>{selected_x.title(), selected_y.title(), selected_z.title()}",
                paper_bgcolor="#f3f3f3",
                scene={"aspectmode": "cube", "xaxis": {"title": f"{selected_x.title()} (USD)", },
                       "yaxis": {"title": f"{selected_y.title()} (USD)", },
                       "zaxis": {"title": f"{selected_z.title()} (USD)", }})
            }


# %% General, entry point

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)


# %% OLD
def gen_2_cols_OBSELETE():
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


def gen_slider_OBSELETE():
    return dcc.Slider(
        id='slider-ship-id',
        className='slider',
        min=1,
        max=10,
        step=1,
        # value=1,
        # marks={n: '{}'.format(n) for n in range(10)},
    )
