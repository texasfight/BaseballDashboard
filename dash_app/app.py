import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import pandas as pd

from pathlib import Path


app = Dash(__name__)

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
batting_heavy_min = pd.merge(batting_df, teams_df[['minAB', 'yearID', 'teamID']], on=['yearID', 'teamID'])
batting_heavy_min = batting_heavy_min[batting_heavy_min["minAB"] < batting_heavy_min["AB"]]

# Add in full name data
batting_heavy_data = pd.merge(batting_heavy_min, players_df, on=["playerID"])
batting_heavy_data["fullName"] = batting_heavy_data["nameFirst"] + " " + batting_heavy_data["nameLast"]
# plot scatter plot
stolen_homer_plot = px.scatter(batting_heavy_data, x="SB", y="HR", hover_name="fullName", hover_data=["yearID", "G"])

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
strikeout_era_plot = px.scatter(pitching_full,
                                x="SO", y="ERA",
                                color="yearID",
                                hover_name="fullName", hover_data=["yearID", "G"])

@app.callback(
    Output('pitching_plot', 'figure'),
    [Input('my-range-slider', 'value')])
def update_output(value):
    data = pitching_full[(value[0] < pitching_full["yearID"]) & (pitching_full["yearID"] < value[1])]
    return px.scatter(data,
                      x="SO", y="ERA",
                      color="yearID",
                      hover_name="fullName", hover_data=["yearID", "G"])


app.layout = html.Div(children=[
    html.H1(children="Baseball Dashboard"),
    html.A(children="We're making this app, boyyyyyyyy", href="https://wikipedia.com"),
    dcc.Graph(id="pitching_plot", figure=strikeout_era_plot),
    dcc.RangeSlider(1899, 2021, value=[1899, 2021], id='my-range-slider', marks=None, tooltip={'always_visible': True}),
    dcc.Graph(id="batting_plot", figure=stolen_homer_plot)
])


if __name__ == "__main__":
    app.run_server(debug=True)
