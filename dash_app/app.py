import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import flask
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

server = flask.Flask(__name__)

app = Dash(__name__,
           server=server,
           external_stylesheets=[dbc.themes.LUX])

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

pres_df = pd.read_csv(POLIT_DIR / "Presidents.csv").rename({"Year": "yearID"}, axis=1).drop("Republican", axis=1)
congress_df = pd.read_csv(POLIT_DIR / "Representatives.csv").rename({"Year": "yearID"}, axis=1)
senators_df = pd.read_csv(POLIT_DIR / "Senators.csv").rename({"Year": "yearID"}, axis=1)

# Remove old years
batting_df = batting_df[batting_df["yearID"] >= 1899]
# Filter only to qualifying players
teams_df["minAB"] = 3.1 * teams_df['G']
batting_min = pd.merge(batting_df, teams_df[['minAB', 'yearID', 'teamID']], on=['yearID', 'teamID'])
batting_min["PA"] = batting_min[["AB", "HBP", "BB", "SF", "SH"]].sum(axis=1, skipna=True)
batting_min = batting_min[batting_min["minAB"] < batting_min["PA"]]
# Add in full name data
batting_full = pd.merge(batting_min, players_df, on=["playerID"])
batting_full["fullName"] = batting_full["nameFirst"] + " " + batting_full["nameLast"]

# batting_advanced_df = batting_advanced_df.rename({'Tm': 'teamID'}, axis=1)
batting_advanced_df = pd.merge(batting_full, batting_advanced_df.drop("G", axis=1), on=['yearID', 'bbrefID'])
batting_advanced_df = batting_advanced_df[['playerID', 'bbrefID', 'yearID', 'fullName', 'teamID', 'birthState',
                                           'birthCountry', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS',
                                           'BB', 'SO', 'HBP', 'SH', 'SF', 'WAR', 'oWAR', 'dWAR']]

# Batting Scatter Plot
batting_plot = px.scatter(batting_full, x="SB", y="HR", hover_name="fullName", hover_data=["yearID", "G"])

# Batting Correlation Matrix between Statistics
batting_correlation = batting_advanced_df.corr()
batting_correlation_matrix = px.imshow(batting_correlation, text_auto=True, aspect='auto', zmax=1, zmin=-1,
                                       title='Batting Statistics Correlation Matrix',
                                       color_continuous_scale=px.colors.diverging.Fall)

# Batting Political Data
batting_political = pd.merge(batting_advanced_df, pres_df, on="yearID").rename({"Democrat": "demPres"},
                                                                               axis=1)
batting_political = pd.merge(batting_political,
                             senators_df[["yearID", "% Democrats"]],
                             on="yearID").rename({"% Democrats": "demSenate"}, axis=1)

batting_political = pd.merge(batting_political,
                             congress_df[["yearID", "% Democrats"]],
                             on="yearID").rename({"% Democrats": "demCongress"}, axis=1)

idx = pd.MultiIndex.from_product((batting_political['playerID'].unique(), batting_political['demPres'].unique()),
                                 names=["playerID", "demPres"])
bat_pres_comp = batting_political.groupby(["playerID", "demPres"]).agg({"WAR": "mean"}).reindex(idx)
bat_pres_comp = bat_pres_comp.fillna(0)
batting_political_score = bat_pres_comp.groupby("playerID").agg(np.subtract.reduce)
batting_political_score.columns = ['Political Score']

# Pitching Data - Filter Year and Pitches
pitching_df = pitching_df[pitching_df["yearID"] >= 1899]
teams_df["minOUT"] = teams_df["G"] * 3
pitching_min = pd.merge(pitching_df, teams_df[['minOUT', 'yearID', 'teamID']], on=['yearID', 'teamID'])
pitching_min = pitching_min[pitching_min["minOUT"] < pitching_min["IPouts"]]
# Add in Full Names for Pitchers
pitching_full = pd.merge(pitching_min, players_df, on=["playerID"])
pitching_full["fullName"] = pitching_full["nameFirst"] + " " + pitching_full["nameLast"]
# pitching_advanced_df = pitching_advanced_df.rename({'Tm': 'teamID'}, axis=1)
pitching_advanced_df = pd.merge(pitching_full, pitching_advanced_df.drop(['G', 'GS'], axis=1), on=['yearID', 'bbrefID'])

pitching_advanced_df = pitching_advanced_df[['playerID', 'bbrefID', 'yearID', 'fullName', 'teamID', 'birthState',
                                             'birthCountry', 'W', 'GS', 'G', 'SV', 'ER', 'SO', 'BB', 'ERA', 'BAOpp',
                                             'WP', 'HR', 'gmLI', 'WAR']]
# Pitching Scatter Plot
pitching_plot = px.scatter(pitching_full,
                           x="SO", y="ERA",
                           color="yearID",
                           range_color=[1899, 2022],
                           hover_name="fullName", hover_data=["yearID", "G"])

# Pitching Correlation Between Statistics
pitching_correlation = pitching_advanced_df.corr()
pitching_correlation_matrix = px.imshow(pitching_correlation, text_auto=True, aspect='auto', zmax=1, zmin=-1,
                                        title='Pitching Statistics Correlation Matrix',
                                        color_continuous_scale=px.colors.diverging.Fall)

# Pitching Political
pitching_political = pd.merge(pitching_advanced_df, pres_df, on="yearID").rename({"Democrat": "demPres"},
                                                                                 axis=1)
pitching_political = pd.merge(pitching_political,
                              senators_df[["yearID", "% Democrats"]],
                              on="yearID").rename({"% Democrats": "demSenate"}, axis=1)

pitching_political = pd.merge(pitching_political,
                              congress_df[["yearID", "% Democrats"]],
                              on="yearID").rename({"% Democrats": "demCongress"}, axis=1)
idx = pd.MultiIndex.from_product((pitching_political['playerID'].unique(), pitching_political['demPres'].unique()),
                                 names=["playerID", "demPres"])
pitch_pres_comp = pitching_political.groupby(["playerID", "demPres"]).agg({"WAR": "mean"}).reindex(idx)
pitch_pres_comp = pitch_pres_comp.fillna(0)
pitching_political_score = pitch_pres_comp.groupby("playerID").agg(np.subtract.reduce)
pitching_political_score.columns = ['Political Score']

batting_heavy_advanced_poli = pd.merge(batting_advanced_df, batting_political_score, on='playerID')
batting_heavy_advanced_main = batting_heavy_advanced_poli.iloc[:, 7:26]
batting_heavy_advanced_main = batting_heavy_advanced_main.fillna(value=0, axis=0)
scaler = StandardScaler()
batting_heavy_advanced_main_scale = pd.DataFrame(scaler.fit_transform(batting_heavy_advanced_main), columns=['G', 'AB',
                                                                                                             'R', 'H',
                                                                                                             '2B', '3B',
                                                                                                             'HR',
                                                                                                             'RBI',
                                                                                                             'SB', 'CS',
                                                                                                             'BB', 'SO',
                                                                                                             'HBP',
                                                                                                             'SH',
                                                                                                             'SF',
                                                                                                             'WAR',
                                                                                                             'oWAR',
                                                                                                             'dWAR',
                                                                                                             'Political Score'])
inertia = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(batting_heavy_advanced_main_scale)
    inertia.append(kmeans.inertia_)

kmeans = KMeans(
    n_clusters=4, init="k-means++",
    n_init=10,
    tol=1e-04, random_state=42
)
kmeans.fit(batting_heavy_advanced_main_scale)
batting_heavy_advanced_main_scale['label'] = kmeans.labels_
polar = batting_heavy_advanced_main_scale.groupby("label").mean().reset_index()
polar = pd.melt(polar, id_vars=["label"])
batting_flower_kmeans = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,
                                      height=800, width=1200, title='Batting Cluster Features - Kmeans')

pie = batting_heavy_advanced_main_scale.groupby('label').size().reset_index()
pie.columns = ['label', 'value']
batting_pie_fig = px.pie(pie, values='value', names='label')

pca_num_components = 2
reduced_data = PCA(n_components=pca_num_components).fit_transform(
    batting_heavy_advanced_main_scale.drop('label', axis=1))
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
batting_cluster_kmeans = px.scatter(results, x="pca1", y="pca2", color=batting_heavy_advanced_main_scale['label'],
                                    hover_name=batting_advanced_df['fullName'], title='Batting Cluster - Kmeans')

pitching_full_advanced_poli = pd.merge(pitching_advanced_df, pitching_political_score, on='playerID')
pitching_full_advanced_main = pitching_full_advanced_poli.iloc[:, 7:21]
pitching_full_advanced_main = pitching_full_advanced_main.fillna(value=0, axis=0)

scaler = StandardScaler()
pitching_full_advanced_main_scale = pd.DataFrame(scaler.fit_transform(pitching_full_advanced_main),
                                                 columns=['W', 'GS', 'G', 'SV', 'ER', 'SO', 'BB', 'ERA', 'BAOpp',
                                                          'WP', 'HR', 'gmLI', 'WAR', 'Political Score'])

kmeans = KMeans(
    n_clusters=3, init="k-means++",
    n_init=10,
    tol=1e-04, random_state=42
)

kmeans.fit(pitching_full_advanced_main_scale)
pitching_full_advanced_main_scale['label'] = kmeans.labels_

pca_num_components = 2

reduced_data = PCA(n_components=pca_num_components).fit_transform(
    pitching_full_advanced_main_scale.drop('label', axis=1))
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
pitching_cluster_kmeans = px.scatter(results, x="pca1", y="pca2", title='Pitching Cluster - Kmeans',
                                     color=pitching_full_advanced_main_scale['label'],
                                     hover_name=pitching_full_advanced_poli['fullName'])


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
        dbc.Row([
            dbc.Col(dbc.Card([
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
            ], color="secondary")),
            dbc.Col(dbc.Card([
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
            ], color="secondary"))], className="p-2"
        )),

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
    html.Div(
        dbc.Card([
            dcc.Graph(id="Batting Cluster Flower - Kmeans", figure=batting_flower_kmeans),
        ], color="secondary"), className="w-75 mx-auto p-2"
    ),

    html.Div(
        dbc.Card([
            dcc.Graph(id="Batting Cluster - Kmeans", figure=batting_cluster_kmeans),
        ], color="secondary"), className="w-75 mx-auto p-2"
    ),

    html.Div(
        dbc.Card([
            dcc.Graph(id="Pitching Cluster - Kmeans", figure=pitching_cluster_kmeans),
        ], color="secondary"), className="w-75 mx-auto p-2"
    ),
])
if __name__ == "__main__":
    app.run_server(debug=True)
