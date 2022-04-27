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
batting_advanced_df = pd.read_csv(BASEBALL_DIR / 'advanced' / 'batting_advanced.csv')
pitching_advanced_df = pd.read_csv(BASEBALL_DIR / 'advanced' / 'pitching_advanced.csv')

# Remote old years
batting_df = batting_df[batting_df["yearID"] >= 1899]
# Filter only to qualifying players
teams_df["minAB"] = 3.1 * teams_df['G']
batting_min = pd.merge(batting_df, teams_df[['minAB', 'yearID', 'teamID']], on=['yearID', 'teamID'])
batting_min = batting_min[batting_min["minAB"] < batting_min["AB"]]

# Add in full name data
batting_full = pd.merge(batting_min, players_df, on=["playerID"])
batting_full["fullName"] = batting_full["nameFirst"] + " " + batting_full["nameLast"]

batting_advanced_df = batting_advanced_df.rename({'Tm': 'teamID'}, axis=1)
batting_advanced_df = pd.merge(batting_full, batting_advanced_df, on=['yearID', 'bbrefID', 'teamID', 'G'])

batting_advanced_df = batting_advanced_df[
    ['playerID', 'bbrefID', 'yearID', 'fullName', 'teamID', 'birthState', 'birthCountry', 'G', 'AB', 'R', 'H', '2B',
     '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'HBP', 'SH', 'SF', 'WAR', 'oWAR', 'dWAR']]

batting_correlation = batting_advanced_df.corr()
batting_correlation_matrix = px.imshow(batting_correlation,
                                       text_auto=True,
                                       aspect='auto', zmax=1, zmin=-1,
                                       color_continuous_scale=px.colors.diverging.Fall)

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

pitching_advanced_df = pitching_advanced_df.rename({'Tm': 'teamID'}, axis=1)
pitching_advanced_df = pd.merge(pitching_full, pitching_advanced_df, on=['yearID', 'bbrefID', 'teamID', 'G', 'GS'])

pitching_advanced_df = pitching_advanced_df[
    ['playerID', 'bbrefID', 'yearID', 'fullName', 'teamID', 'birthState', 'birthCountry', 'W', 'GS', 'G', 'SV', 'ER',
     'SO', 'BB', 'ERA', 'BAOpp', 'WP', 'HR', 'gmLI', 'WAR']]

pitching_correlation = pitching_advanced_df.corr()
pitching_correlation_matrix = px.imshow(pitching_correlation,
                                        text_auto=True,
                                        aspect='auto', zmax=1, zmin=-1,
                                        color_continuous_scale=px.colors.diverging.Fall)

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
     Input('pitching_dropdown_x', 'value'),
     Input('pitching_dropdown_y', 'value')])
def update_pitching(year_range, x_value, y_value):
    data = pitching_full[(year_range[0] <= pitching_full["yearID"]) & (pitching_full["yearID"] <= year_range[1])]
    pitching_plot = px.scatter(data,
                               title='Pitching Statistics',
                               x=x_value, y=y_value,
                               color="yearID",
                               range_color=[1899, 2022],
                               hover_name="fullName", hover_data=["yearID", "G"])

    pitching_plot.update_layout(transition_duration=1000)

    return pitching_plot


@app.callback(
    Output('batting_plot', 'figure'),
    [Input('batting-range-slider', 'value'),
     Input('batting_dropdown_x', 'value'),
     Input('batting_dropdown_y', 'value')])
def update_hitting(year_range, x_value, y_value):
    data = batting_full[(year_range[0] <= batting_full["yearID"]) & (batting_full["yearID"] <= year_range[1])]
    batting_plot = px.scatter(data,
                              title='Batting Statistics',
                              x=x_value, y=y_value,
                              color="yearID",
                              range_color=[1899, 2022],
                              hover_name="fullName", hover_data=["yearID", "G"])

    batting_plot.update_layout(transition_duration=1000)

    return batting_plot


app.layout = html.Div(children=[
    html.H1(children="Baseball Dashboard"),
    html.A(children="Click here if you are bored...", href="https://www.youtube.com/watch?v=ECRcCIg0K50"),
    html.Div(
        dbc.Card([
            dcc.Graph(id="pitching_plot", figure=pitching_plot),
            dcc.RangeSlider(1899, 2021, marks={x: str(x) for x in range(1899, 2022, 20)},
                            value=[1899, 2021],
                            updatemode='drag',
                            id='pitching-range-slider',
                            tooltip={'always_visible': True}),
            'X-Axis',
            dcc.Dropdown(id='pitching_dropdown_x',
                         options=[
                             {'label': 'Wins', 'value': 'W'},
                             {'label': 'Games Started', 'value': 'GS'},
                             {'label': 'Games', 'value': 'G'},
                             {'label': 'Saves', 'value': 'SV'},
                             {'label': 'Earned Runs', 'value': 'ER'},
                             {'label': 'Strike Outs', 'value': 'SO'},
                             {'label': 'Walks', 'value': 'BB'},
                             {'label': 'Opponent Batting Average', 'value': 'BAOpp'},
                             {'label': 'Wild Pitches', 'value': 'WP'},
                             {'label': 'Home Runs', 'value': 'HR'},
                             {'label': 'Leverage Index', 'value': 'gmLI'},
                             {'label': 'Wins Above Replacement', 'value': 'WAR'}],
                         value='G',
                         searchable=True,
                         placeholder='Please select...',
                         clearable=True,
                         className="w-50 p-2"),
            'Y-Axis',
            dcc.Dropdown(id='pitching_dropdown_y',
                         options=[
                             {'label': 'Wins', 'value': 'W'},
                             {'label': 'Games Started', 'value': 'GS'},
                             {'label': 'Games', 'value': 'G'},
                             {'label': 'Saves', 'value': 'SV'},
                             {'label': 'Earned Runs', 'value': 'ER'},
                             {'label': 'Strike Outs', 'value': 'SO'},
                             {'label': 'Walks', 'value': 'BB'},
                             {'label': 'Opponent Batting Average', 'value': 'BAOpp'},
                             {'label': 'Wild Pitches', 'value': 'WP'},
                             {'label': 'Home Runs', 'value': 'HR'},
                             {'label': 'Leverage Index', 'value': 'gmLI'},
                             {'label': 'Wins Above Replacement', 'value': 'WAR'}],
                         value='W',
                         searchable=True,
                         placeholder='Please select...',
                         clearable=True,
                         className="w-50 p-2"),
                  ], color="secondary"), className="w-75 mx-auto p-2"
            ),

    html.Div(
        dbc.Card([
            dcc.Graph(id="batting_plot", figure=batting_plot),
            dcc.RangeSlider(1899, 2021, marks={x: str(x) for x in range(1899, 2022, 20)},
                            value=[1899, 2021],
                            updatemode='drag',
                            id='batting-range-slider',
                            tooltip={'always_visible': True}),
            'X-Axis',
            dcc.Dropdown(id='batting_dropdown_x',
                         options=[
                             {'label': 'Home Runs', 'value': 'HR'},
                             {'label': 'Hits', 'value': 'H'},
                             {'label': 'Runs Batted In', 'value': 'RBI'},
                             {'label': 'Runs', 'value': 'R'},
                             {'label': 'Stolen Bases', 'value': 'SB'},
                             {'label': 'Caught Stealing', 'value': 'CS'},
                             {'label': 'Wins Above Replacement', 'value': 'WAR'},
                             {'label': 'Offensive Wins Above Replacement', 'value': 'oWAR'},
                             {'label': 'Defensive Wins Above Replacement', 'value': 'dWAR'},
                             {'label': 'Strike Outs', 'value': 'SO'},
                             {'label': 'Walks', 'value': 'BB'},
                             {'label': 'Hit by Pitch', 'value': 'HBP'},
                             {'label': 'Games Played', 'value': 'G'},
                             {'label': 'At Bats', 'value': 'AB'},
                             {'label': 'Doubles', 'value': '2B'},
                             {'label': 'Triples', 'value': '3B'},
                             {'label': 'Sacrifice Bunts', 'value': 'SH'},
                             {'label': 'Sacrifice Flies', 'value': 'SF'}],
                         value='G',
                         searchable=True,
                         placeholder='Please select...',
                         clearable=True,
                         className="w-50 p-2"),
            'Y-Axis',
            dcc.Dropdown(id='batting_dropdown_y',
                         options=[
                             {'label': 'Home Runs', 'value': 'HR'},
                             {'label': 'Hits', 'value': 'H'},
                             {'label': 'Runs Batted In', 'value': 'RBI'},
                             {'label': 'Runs', 'value': 'R'},
                             {'label': 'Stolen Bases', 'value': 'SB'},
                             {'label': 'Caught Stealing', 'value': 'CS'},
                             {'label': 'Wins Above Replacement', 'value': 'WAR'},
                             {'label': 'Offensive Wins Above Replacement', 'value': 'oWAR'},
                             {'label': 'Defensive Wins Above Replacement', 'value': 'dWAR'},
                             {'label': 'Strike Outs', 'value': 'SO'},
                             {'label': 'Walks', 'value': 'BB'},
                             {'label': 'Hit by Pitch', 'value': 'HBP'},
                             {'label': 'Games Played', 'value': 'G'},
                             {'label': 'At Bats', 'value': 'AB'},
                             {'label': 'Doubles', 'value': '2B'},
                             {'label': 'Triples', 'value': '3B'},
                             {'label': 'Sacrifice Bunts', 'value': 'SH'},
                             {'label': 'Sacrifice Flies', 'value': 'SF'}],
                         value='H',
                         searchable=True,
                         placeholder='Please select...',
                         clearable=True,
                         className="w-50 p-2"),
                  ], color="secondary"), className="w-75 mx-auto p-2"
            ),

    html.Div(
        dbc.Card([
            dcc.Graph(id="Pitching Correlation Matrix", figure=pitching_correlation_matrix),
                   ], color="secondary"), className="w-75 mx-auto p-2"
            ),
    
   html.Div(
        dbc.Card([
            dcc.Graph(id="Batting Correlation Matrix", figure=batting_correlation_matrix),
                   ], color="secondary"), className="w-75 mx-auto p-2"
            ),
        ])
if __name__ == "__main__":
    app.run_server(debug=True)