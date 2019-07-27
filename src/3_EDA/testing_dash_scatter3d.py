import plotly.express as px
iris = px.data.iris()
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
fig.show()


def scatter3D(value):
    data=[dict(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            type='scatter3d',
            text=None,
            marker=dict(
                size=12,
                opacity=0.8
                )
            )
        ]