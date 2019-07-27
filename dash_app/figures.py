# %% TEST 3D scatter

# iris = px.data.iris()
# print(go.Scatter3d)
# print("Iris", iris)
# trace=[go.Scatter3d(iris, x='sepal_length', y='sepal_width', z='petal_width', color='species')]
import numpy as np
import plotly.graph_objs as go

t = np.linspace(0, 10, 50)
x, y, z = np.cos(t), np.sin(t), t

test_fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])