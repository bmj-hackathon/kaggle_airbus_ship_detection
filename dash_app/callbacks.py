import dash
from utils import *
from pathlib import Path

# %%%%%%%%%%%% LOAD IMAGE CLASS

# TODO: This is just a patch for now, local dev!
import sys

path_image_class = Path().cwd() / 'src' / '3_EDA'
path_image_class = path_image_class.resolve()
sys.path.append(str(path_image_class.absolute()))
print(sys.path)
from eda_00_Image_class import Image, convert_rgb_img_to_b64string, fit_kmeans_pixels, convert_rgb_img_to_b64string_straight

def register_callbacks(app, df, df_by_image, img_zip, Image):


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
        image.load(img_zip, df)
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
