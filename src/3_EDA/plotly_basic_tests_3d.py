import plotly.graph_objects as go
import numpy as np

import plotly.io as pio

#%%

# pio.renderers
# pio.renderers.default = "browser"
# Helix equation
# t = np.linspace(0, 10, 50)

R, G, B = pixel_locs.T

# x, y, z = np.cos(t), np.sin(t), t


fig = go.Figure()

for label in np.unique(labels).tolist():
    this_cluster_mask = labels == label
    these_colors = np.unique(colors[this_cluster_mask])
    these_colors = these_colors * 255
    these_colors = these_colors.astype(int).tolist()
    this_color_string = "rgb({},{},{})".format(these_colors[0], these_colors[2], these_colors[2])
    logging.info("Label {} with {} color {}".format(label, np.sum(this_cluster_mask), this_color_string))

    cluster_string = "Cluster {}, {:0.1%}".format(label, np.sum(this_cluster_mask) / len(this_cluster_mask))

    markers = dict(color=this_color_string, size=2)
    # markers = dict( color=[f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})' for _ in range(25)], size=10)
    fig.add_trace(go.Scatter3d(
        x=R[this_cluster_mask], y=G[this_cluster_mask], z=B[this_cluster_mask],
        mode='markers',
        marker=markers,
        name=cluster_string,
    ))

# ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))
bg_color = "rgb(230,230,230)"
fig.update_layout(scene=dict(
    xaxis=dict(
        # backgroundcolor="rgb(255, 220, 220)",
        backgroundcolor=bg_color,
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white", ),
    yaxis=dict(
        # backgroundcolor="rgb(220, 255, 220)",
        backgroundcolor=bg_color,
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white"),
    zaxis=dict(
        # backgroundcolor="rgb(220, 220, 255)",
        backgroundcolor=bg_color,
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white", ),
    xaxis_title='Red',
    yaxis_title='Green',
    zaxis_title='Blue'),
)

fig.show()
#%%
def get_kmeans_figure(image):
    image_id = df_by_image.index[100]  # Select an image with 15 ships
    # image_id = df_by_image.index[10] # Select an image with 15 ships
    image = Image(image_id)
    image.load(img_zip, df)
    image.load_ships()
    r = image.records

    image.ship_summary_table()
    kmeans = image.k_means(num_clusters=2)