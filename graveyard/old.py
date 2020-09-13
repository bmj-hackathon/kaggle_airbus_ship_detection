
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
