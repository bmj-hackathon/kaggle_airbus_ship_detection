import dash
from utils import *
from pathlib import Path
import logging
import numpy as np
import dash_html_components as html

# %%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys

path_image_class = Path().cwd() / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
# print(sys.path)
from eda_00_Image_class import ShipImage, convert_rgb_img_to_b64string, fit_kmeans_pixels, convert_rgb_img_to_b64string_straight

#def register_callbacks(app, df, df_by_image, img_zip, Image):
def register_callbacks(app, df, df_by_image, img_zip):


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
        # images = df.loc[df['TotalShips'] == value].index.to_series()
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
            dash.dependencies.Output('ship-data-table', 'columns'),
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
        image = ShipImage(image_id)
        image.load(img_zip, df)
        logging.info("GET IMAGE DATA,  num_ships = ".format(image.num_ships))
        if not image.num_ships:
            jpg_ellipse_image = convert_rgb_img_to_b64string(image.img)
            image_source_string = "data:image/png;base64, {}".format(jpg_ellipse_image)
            # TODO: Messy way to make an empty table
            col_heads = [{"name": c, "id": c} for c in ['ship', 'index_number', 'x', 'y', 'area', 'angle',]]
            empty_data = [{'ship': 'None', 'index_number': 'n/a', 'x': 'n/a', 'y': 'n/a', 'area': 'n/a', 'angle': 'n/a'}]
            return col_heads, empty_data, image_source_string

        image.load_ships()

        # Build summary table
        df_ships = image.ship_summary_table()

        # Get ellipse image
        ndarray_ellipse_image = image.draw_ellipses_img()
        # print("Original Image:")
        # print(ndarray_ellipse_image)
        # print('ndarray_ellipse_image', ndarray_ellipse_image.shape)
        # jpg_ellipse_image = convert_rgb_img_to_b64string(ndarray_ellipse_image)
        jpg_ellipse_image = convert_rgb_img_to_b64string(ndarray_ellipse_image)
        image_source_string = "data:image/png;base64, {}".format(jpg_ellipse_image)
        col_heads = [{"name": i, "id": i} for i in df_ships.columns]
        # print(col_heads)
        data = df_ships.to_dict('records')
        # print(data)
        return col_heads, data, image_source_string


    #%%-----------------
    # K Means button
    # --------------------
    # if 0:
    @app.callback(
        output=[
            dash.dependencies.Output('empty-para', 'children'),
            dash.dependencies.Output('kmeans-picture-LIVE', 'src'),
            dash.dependencies.Output('kmeans-scatter-LIVE', 'figure'),
            dash.dependencies.Output('kmeans-summary-string', 'children'),
        ],

        inputs=[
            dash.dependencies.Input('button-start-kmeans', 'n_clicks'),
            # dash.dependencies.Input('slider-cluster-counts', 'value'),
            # dash.dependencies.Input('image_id', 'children')
         ],

        state=[
            dash.dependencies.State('slider-cluster-counts', 'value'),
            dash.dependencies.State('image_id', 'children'),
        ]
    )  # END DECORATOR
    def kmeans_update(n_clicks, n_clusters, image_id):
        if image_id==None:
            print("Sample image loaded for start!")
            image_id = '1defabda3.jpg'
        print('n_clicks', n_clicks, 'n_clusters', n_clusters, 'image_id', image_id)
        print("START K Means with {} clusters on image {}".format(n_clusters, image_id))

        image = ShipImage(image_id)
        image.load(img_zip, df)
        kmeans = image.k_means(n_clusters)

        data = image.img / 255
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

        logging.info("Fit {} pixels {} clusters".format(data.shape[0], n_clusters))

        # Get the image
        kmeans_img = fit_kmeans_pixels(image.img, kmeans)
        kmeans_img_canvas = kmeans_img
        kmeans_img_canvas = kmeans_img * 255
        kmeans_img_canvas = kmeans_img_canvas.astype(int)


        # kmeans_img_canvas = kmeans_img



        # print("KMEANS IMAGE CANVAS:")
        # print(kmeans_img_canvas)
        # print('kmeans_img_canvas', type(kmeans_img_canvas), kmeans_img_canvas.dtype, kmeans_img_canvas.shape)
        kmeans_img_canvas = kmeans_img_canvas.astype(np.uint8)

        # Build an image HTML object
        # kmeans_img_str = convert_rgb_img_to_b64string_straight(kmeans_img_canvas)
        kmeans_img_str = convert_rgb_img_to_b64string(kmeans_img_canvas)
        image_source_string = "data:image/png;base64, {}".format(kmeans_img_str)
        html_kmeans_img_STATIC = html.Img(src=image_source_string)

        # Build the scatter plot
        fig = get_kmeans_figure(image, kmeans)



        # Get the text summary
        summary_string = f"""
##### **KMeans fit results**
    num_clusters = {n_clusters}"""

        def add_md_line(md_str, new_line):
            md_str += "\n\n" + new_line
            return md_str

        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        for c_name, c_count, c_position in zip(unique, counts, kmeans.cluster_centers_):
            this_col = np.around(c_position, 3) * 255
            lg_str = "\tCluster {} at {} with {:0.1%} of the pixels".format(c_name, this_col, c_count/data.shape[0])
            logging.info(lg_str)
            summary_string = add_md_line(summary_string,lg_str)

        if len(unique) == 2:
            dist = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
            lg_str = "\tDistance between c1 and c2: {}".format(dist)
            logging.info(lg_str)
            summary_string = add_md_line(summary_string,lg_str)

        # print('Summary String: ', summary_string)
        return "", image_source_string, fig, summary_string


    # %% TESTING GRAPH


    # # https://github.com/plotly/simple-example-chart-apps/blob/master/dash-3dscatterplot/apps/main.py
    # @app.callback(
    #     dash.dependencies.Output("my-graph", "figure"),
    #     []
    # )
    # def update_figure(selected_x, selected_y, selected_z):
    #     iris = px.data.iris()
    #     trace = [go.Scatter3d(iris, x='sepal_length', y='sepal_width', z='petal_width', color='species')]
    #     return {"data": trace,
    #             "layout": go.Layout(
    #                 height=700, title=f"TITLE HERE",
    #                 paper_bgcolor="#f3f3f3",
    #                 scene={"aspectmode": "cube",
    #                        "xaxis": {"title": f"X", },
    #                        "yaxis": {"title": f"Y", },
    #                        "zaxis": {"title": f"Z", }})
    #             }
