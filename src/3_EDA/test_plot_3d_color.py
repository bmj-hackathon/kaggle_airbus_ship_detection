from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
# from PIL import Image
import plotly.io as pio
pio.renderers.default = "browser"
import plotly as plotly
import numpy as np
import plotly.graph_objs as go
import cv2
import base64
#%%
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


# %%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys

path_image_class = Path().cwd() / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
# print(sys.path)
from eda_00_Image_class import SimpleImage, ShipImage, convert_rgb_img_to_b64string, fit_kmeans_pixels, convert_rgb_img_to_b64string_straight

import eda_00_Image_class as mj_img_utils

#%%
data_path = Path(r"/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/DATA/airbus-ship-detection/sample_images")
image_name = "28db2ad2c.jpg"
image_name = "28d931eb9.jpg"
image_path = data_path / image_name

img = SimpleImage.load_from_path(image_path)
chans = img.get_channels()
# print(chans)

chans['R'].shape
np.max(chans['R'])
img_str = img.get_b64_jpg()
# cv2.cvtColor(chans['R'], cv2.COLOR)

#%%
channel_imgs = mj_img_utils.get_channels_b64(img)
channel_imgs

fig = plotly.subplots.make_subplots(rows=1, cols=3)
# for c in enumerate(channel_imgs):
#     fig.add_trace()

# fig.add_trace(
#     go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
#     row=1, col=1
# )
#
# fig.add_trace(
#     go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
#     row=1, col=2
# )
fig.add_trace(
    go.Scatter(
        x=[0, 768 * 1],
        y=[0, 768 * 1],
        mode="markers",
        marker_opacity=0
    )
)
fig.add_trace(
    go.Scatter(
        x=[0, 768 * 1],
        y=[0, 768 * 1],
        mode="markers",
        marker_opacity=0
    )
)
fig.add_trace(
    go.Scatter(
        x=[0, 768 * 1],
        y=[0, 768 * 1],
        mode="markers",
        marker_opacity=0
    )
)
def get_img_layout(ref):
    this = go.layout.Image(
        x=0,
        sizex=768,
        y=768,
        sizey=768,
        xref="x{}".format(ref),
        yref="y{}".format(ref),
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source='data:image/jpg;base64,{}'.format(img_str))
    return this

# Add image
fig.update_layout(
    images=[get_img_layout(""),get_img_layout(2),get_img_layout(3)]
)
# for x in fig.layout:
#     print(x)
fig.update_layout(height=600, width=800, title_text="Subplots")
fig.show()

raise

#%%
def get_image_figure(size, b64_img_str, scale):
    # Create figure
    fig = go.Figure()

    # Constants
    img_width = size[0]
    img_height = size[1]

    scale_factor = scale

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.update_layout(
        images=[go.layout.Image(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source='data:image/jpg;base64,{}'.format(b64_img_str))])

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig


this_fig = get_image_figure(img.shape[0:2], img.get_b64_jpg(), 0.5)
this_fig.show()



this_fig = get_image_figure(img.shape[0:2], jpg_as_text, 0.5)
this_fig.show()


#%%
# this_figure = {
#     'data': [],
#     'layout': {
#         'xaxis': {
#             'range': RANGE
#         },
#         'yaxis': {
#             'range': RANGE,
#             'scaleanchor': 'x',
#             'scaleratio': 1
#         },
#         'height': 600,
#         'images': [{
#             'xref': 'x',
#             'yref': 'y',
#             'x': RANGE[0],
#             'y': RANGE[1],
#             'sizex': RANGE[1] - RANGE[0],
#             'sizey': RANGE[1] - RANGE[0],
#             'sizing': 'stretch',
#             'layer': 'below',
#             'source': 'data:image/png;base64,{}'.format(encoded_image)
#         }],
#         'dragmode': 'select'  # or 'lasso'
#     }


#%%
if 0:
    fig = pyplot.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d") # 3D plot with scalar values in each axis

    im = Image.open(image_path)
    r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))

    axis.scatter(r, g, b, c="#ff0000", marker="o")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    pyplot.show()


#%%

if 0:
    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t

    test_fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    test_fig.show()

    # plot_url = plotly.plot(fig, filename='box-plot', auto_open=False)