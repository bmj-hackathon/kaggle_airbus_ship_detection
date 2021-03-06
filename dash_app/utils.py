# %%
import plotly.graph_objs as go
import numpy as np
import logging


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


# %%%%%%%%%%%% CLASSES
# %%
def plot_hist(img, ax):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

def get_RGB(array):
    colors = list()
    for c in array:
        print(c, type(c))
        # print(c.tolist())
        this_color = [int(el*255) for el in c.tolist()]
        # print()
        # raise
        colors.append(this_color)
        # colors.append(this_color)
    return colors

# %%
# TODO: THIS FUNCTION IS DOUBLED HERE! FROM eda_00.py!
def get_kmeans_color(img, _kmeans):
    """asdf

    :param ax: The axis to plot into
    :param img: The orignal image
    :param _kmeans: The FIT kmeans object
    :return:
    """

    original_pixels = img / 255
    new_shape = original_pixels.shape[0] * original_pixels.shape[1], original_pixels.shape[2]
    original_pixel_locations_flat = original_pixels.reshape(new_shape)

    new_pixel_colors = _kmeans.cluster_centers_[_kmeans.predict(original_pixel_locations_flat)]
    logging.info("New pixels, shape {}".format(new_pixel_colors.shape))

    logging.info("Colors: {}".format(get_RGB(np.unique(new_pixel_colors, axis=0))))
    cluster_labels = np.unique(_kmeans.labels_).tolist()
    logging.info("{} custers: {}".format(len(cluster_labels), cluster_labels))

    N_points = 20000
    # Generate a list of 20000 indices
    rng = np.random.RandomState(0)
    i = rng.permutation(original_pixel_locations_flat.shape[0])[:N_points]
    logging.info("Sampling {} points".format(len(i)))

    pixel_locations = original_pixel_locations_flat[i]
    logging.info("Returning pixel locations: {}".format(pixel_locations.shape))
    color_vec_i = new_pixel_colors[i]
    logging.info("Returning colors: {}".format(color_vec_i.shape))
    labels_vec_i = _kmeans.labels_[i]
    logging.info("Returning labels: {}".format(labels_vec_i))

    return pixel_locations, color_vec_i, labels_vec_i


def plot_kmeans_color2(pixel_locs, colors, labels):
    fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
    ax = plt.axes(projection="3d")
    R, G, B = pixel_locs.T

    for label in np.unique(labels).tolist():
        this_cluster_mask = labels == label
        ax.scatter(R[this_cluster_mask], G[this_cluster_mask], B[this_cluster_mask], color=colors[this_cluster_mask],
                   depthshade=False)

    ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))


# %% KMeans figure
def get_kmeans_figure(image, kmeans):
    pixel_locs, colors, labels = get_kmeans_color(image.img, kmeans)
    pixel_locs = pixel_locs * 255
    R, G, B = pixel_locs.T

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
            zerolinecolor="white",
            range=[0, 255]
        ),
        yaxis=dict(
            # backgroundcolor="rgb(220, 255, 220)",
            backgroundcolor=bg_color,
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
            range=[0, 255],
        ),
        zaxis=dict(
            # backgroundcolor="rgb(220, 220, 255)",
            backgroundcolor=bg_color,
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
            range=[0, 255],
        ),
        xaxis_title='Red',
        yaxis_title='Green',
        zaxis_title='Blue',
    ),
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800)
    return fig
