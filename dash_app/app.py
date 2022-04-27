import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

from pathlib import Path

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

MAIN_DIR = Path(".").absolute().parent
BASEBALL_DIR = MAIN_DIR / "baseball_data"
POLIT_DIR = MAIN_DIR / "political_data"

teams_df = pd.read_csv(BASEBALL_DIR / "core" / "Teams.csv")
players_df = pd.read_csv(BASEBALL_DIR / "core" / "People.csv")
batting_df = pd.read_csv(BASEBALL_DIR / "core" / "Batting.csv")
pitching_df = pd.read_csv(BASEBALL_DIR / "core" / "Pitching.csv")
appearances_df = pd.read_csv(BASEBALL_DIR / "core" / "Appearances.csv")

# Remote old years
batting_df = batting_df[batting_df["yearID"] >= 1899]
# Filter only to qualifying players
teams_df["minAB"] = 3.1 * teams_df['G']
batting_min = pd.merge(batting_df, teams_df[['minAB', 'yearID', 'teamID']], on=['yearID', 'teamID'])
batting_min = batting_min[batting_min["minAB"] < batting_min["AB"]]

# Add in full name data
batting_full = pd.merge(batting_min, players_df, on=["playerID"])
batting_full["fullName"] = batting_full["nameFirst"] + " " + batting_full["nameLast"]
# plot scatter plot
batting_plot = px.scatter(batting_full, x="SB", y="HR", hover_name="fullName", hover_data=["yearID", "G"])

# Filter year and pitches
pitching_df = pitching_df[pitching_df["yearID"] >= 1899]
teams_df["minOUT"] = teams_df["G"] * 3
pitching_min = pd.merge(pitching_df, teams_df[['minOUT', 'yearID', 'teamID']], on=['yearID', 'teamID'])
pitching_min = pitching_min[pitching_min["minOUT"] < pitching_min["IPouts"]]
# Add in full names
pitching_full = pd.merge(pitching_min, players_df, on=["playerID"])
pitching_full["fullName"] = pitching_full["nameFirst"] + " " + pitching_full["nameLast"]

# This actually looks a little interesting. Might try to implement this:
# https://plotly.com/python/range-slider/
pitching_plot = px.scatter(pitching_full,
                           x="SO", y="ERA",
                           color="yearID",
                           range_color=[1899, 2022],
                           hover_name="fullName", hover_data=["yearID", "G"])


@app.callback(
    Output('pitching_plot', 'figure'),
    [Input('pitching-range-slider', 'value'),
     Input('pitching_dropdown', 'value')])
def update_pitching(year_range, x_value):
    data = pitching_full[(year_range[0] <= pitching_full["yearID"]) & (pitching_full["yearID"] <= year_range[1])]
    pitching_plot = px.scatter(data,
                                    x=x_value, y="ERA",
                                    color="yearID",
                                    range_color=[1899, 2022],
                                    hover_name="fullName", hover_data=["yearID", "G"])

    pitching_plot.update_layout(transition_duration=1000)

    return pitching_plot


@app.callback(
    Output('batting_plot', 'figure'),
    [Input('batting-range-slider', 'value'),
     Input('batting_dropdown', 'value')])
def update_hitting(year_range, x_value):
    data = batting_full[(year_range[0] <= batting_full["yearID"]) & (batting_full["yearID"] <= year_range[1])]
    batting_plot = px.scatter(data,
                                    x=x_value, y="HR",
                                    color="yearID",
                                    range_color=[1899, 2022],
                                    hover_name="fullName", hover_data=["yearID", "G"])

    batting_plot.update_layout(transition_duration=1000)

    return batting_plot


app.layout = html.Div(children=[
    html.H1(children="Baseball Dashboard"),
    html.A(children="We're making this app, boyyyyyyyy", href="https://wikipedia.com"),
    dcc.Graph(id="pitching_plot", figure=pitching_plot),
    dcc.RangeSlider(1899, 2021, marks={x: str(x) for x in range(1899, 2022, 20)},
                    value=[1899, 2021],
                    updatemode='drag',
                    id='pitching-range-slider', tooltip={'always_visible': True}),
    dcc.Dropdown(id='pitching_dropdown',
                 options=[
                     {'label': 'Wins', 'value': 'W'},
                     {'label': 'Strikeouts', 'value': 'SO'},
                     {'label': 'Earned Runs', 'value': 'ER'}],
                 value='SO',
                 searchable=True,
                 placeholder='Please select...',
                 clearable=True,
                 style={'width': '48%', 'display': 'inline-block'}),
    dcc.Graph(id="batting_plot", figure=batting_plot),
    dcc.RangeSlider(1899, 2021, marks={x: str(x) for x in range(1899, 2022, 20)},
                    value=[1899, 2021],
                    updatemode='drag',
                    id='batting-range-slider', tooltip={'always_visible': True}),
    dcc.Dropdown(id='batting_dropdown',
                 options=[
                     {'label': 'Home Runs', 'value': 'HR'},
                     {'label': 'Hits', 'value': 'H'},
                     {'label': 'Stolen Bases', 'value': 'SB'}],
                 value='SB',
                 searchable=True,
                 placeholder='Please select...',
                 clearable=True,
                 style={'width': '48%', 'display': 'inline-block'}),
])

if __name__ == "__main__":
    app.run_server(debug=True)
