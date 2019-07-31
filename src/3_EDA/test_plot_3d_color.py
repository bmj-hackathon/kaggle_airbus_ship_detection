from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from PIL import Image

data_path = Path(r"/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/DATA/airbus-ship-detection/sample_images")
image_name = "28d931eb9.jpg"
image_name = "28db2ad2c.jpg"
image_path = data_path / image_name

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
import plotly.io as pio
pio.renderers.default = "browser"

import numpy as np
import plotly.graph_objs as go

t = np.linspace(0, 10, 50)
x, y, z = np.cos(t), np.sin(t), t

test_fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
test_fig.show()

# plot_url = plotly.plot(fig, filename='box-plot', auto_open=False)