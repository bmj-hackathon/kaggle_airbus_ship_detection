
import plotly.graph_objects as go
import numpy as np

import plotly.io as pio
pio.renderers
pio.renderers.default = "browser"
# Helix equation
t = np.linspace(0, 10, 50)

x, y, z = pixel_locs.T

# x, y, z = np.cos(t), np.sin(t), t

markers = dict(
    color=[f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})' for _ in
           range(25)],
    size=10)


fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=markers)])
fig.show()
