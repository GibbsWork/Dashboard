from dash import Dash
from dash.dependencies import Input, Output
from plotly.graph_objs import *
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objects as go
import yfinance as yf
import datetime

# Enter tickers to compare
tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'GOOGL', 'FB', 'NVDA', 'AVGO', 'PEP', 'COST', 'ADBE', 'CMCSA',
           'CSCO', 'INTC', 'TMUS', 'AMD', 'TXN', 'QCOM', 'AMGN', 'HON', 'INTU', 'AMAT', 'PYPL', 'ADP', 'BKNG', 'SBUX',
           'CHTR', 'MDLZ', 'ADI', 'NFLX', 'MU', 'ISRG', 'GILD', 'LRCX', 'REGN', 'CSX', 'VRTX', 'FISV', 'ATVI', 'MRNA',
           'KLAC']

# Comparison start date
startDate = '2015-01-01'
# Get data until yesterday
endDate = datetime.datetime.today() - datetime.timedelta(days=1)
data = yf.download(tickers, start=startDate, end=endDate)
# Get download data into PD dataframe, replace with live API in future

app = Dash(__name__,
           external_stylesheets=[dbc.themes.LUX],
           suppress_callback_exceptions=True
           )

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.H3('Spread-Trading Application')
        ], width=9),
    ], justify='center', style={"marginLeft": "-10%"}),

    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id="dropdown1",
                options=[{'label': x, 'value': x} for x in tickers],
                value=tickers[0],
                persistence=True,
                style={"width": "20rem"}
            ),
            dcc.Dropdown(
                id="dropdown2",
                options=[{'label': x, 'value': x} for x in tickers],
                value=tickers[0],
                persistence=True,
                style={"width": "20rem"}
            ),
        ], width=6, style={"display": "flex", "marginLeft": "-30rem"}),
    ], justify='center', style={"marginLeft": "-15rem"}),
    dcc.Graph(id='spread'),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H3(f'Raw Data'), width=9),
        dbc.Col(width=2)
    ], justify='center'),

    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                html.Div([
                    dcc.Dropdown(
                        id="sub1",
                        options=[{'label': x, 'value': x} for x in tickers],
                        value=tickers[0],
                        persistence=True,
                        style={"width": "50%", "marginLeft": "5%"}
                    ),
                    dcc.Graph(id="subgraph1")
                ]
                ), width=6),
            dbc.Col(
                html.Div([
                    dcc.Dropdown(
                        id="sub2",
                        options=[{'label': x, 'value': x} for x in tickers],
                        value=tickers[0],
                        persistence=True,
                        style={"width": "50%", "marginLeft": "5%"}
                    ),
                    dcc.Graph(id="subgraph2")
                ]
                ), width=6),

        ]
    ),
])


@app.callback(
    Output("spread", "figure"),
    Input("dropdown1", "value"),
    Input("dropdown2", "value"),
)
def updater(dropdown1, dropdown2):
    # basic function to  compare the stocks at the same start point
    leg1 = data['Close'][dropdown1]
    leg2 = data['Close'][dropdown2]
    l1_factor = 100 / (leg1[0])
    l2_factor = 100 / (leg2[0])
    spread = (leg1 * l1_factor) - (leg2 * l2_factor)

    def config_fig(x, y):
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            showlegend=False,
        )
        )

        fig.update_layout(
            margin=dict(l=20, r=10, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
            yaxis_range=[min(y), max(y)],
            modebar_add="togglespikelines",
        )

        fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey')

        return go.Figure(data=fig, layout=layout)

    fig = go.Figure()

    spread_fig = config_fig(spread.squeeze().index, spread.squeeze())

    return go.Figure(spread_fig)


@app.callback(
    Output("subgraph1", "figure"),
    Output("subgraph2", "figure"),
    Input("sub1", "value"),
    Input("sub2", "value"),
)
def updater(sub1, sub2):
    # Get the close price of the raw data
    sub1_data = data["Close"][sub1]
    sub2_data = data["Close"][sub2]

    def config_fig(x, y):
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            # name='spread',
            showlegend=False,
        )
        )

        fig.update_layout(
            margin=dict(l=20, r=10, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
            yaxis_range=[min(y), max(y)],
            modebar_add="togglespikelines",
        )

        fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey')
        return go.Figure(data=fig, layout=layout)

    def config_fig1(x, y):
        fig1.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            # name='spread',
            showlegend=False,
        )
        )

        fig1.update_layout(
            margin=dict(l=20, r=10, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
            yaxis_range=[min(y), max(y)],
            modebar_add="togglespikelines",
        )

        fig1.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey')

        return go.Figure(data=fig1, layout=layout)

    fig = go.Figure()
    fig1 = go.Figure()

    fig = config_fig(sub1_data.squeeze().index, sub1_data.squeeze())
    fig1 = config_fig1(sub2_data.squeeze().index, sub2_data.squeeze())
    return go.Figure(fig), go.Figure(fig1)


if __name__ == '__main__':
    # Host the dashboard
    portNumber = 8052
    app.run_server(debug=False, port=portNumber)
