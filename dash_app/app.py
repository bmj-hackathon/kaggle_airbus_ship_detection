import dash
print(dash.__version__)
import dash_core_components as dcc
import dash_html_components as dhtml

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dhtml.Div(children=[
    dhtml.H1(children='Hello Test1'),
    dhtml.Div(children='''
        Dash: Test app number 1 ... !
    '''),
    dcc.Graph(
        id='example1',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Test title 12'
            }
        }
    ),
    dhtml.H3(children='TEST H3'),
    dhtml.Div(children=[
        dhtml.H2(children='Test h2 inside DIV'),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)

