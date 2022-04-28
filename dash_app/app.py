# Plotting and webpage packages
import plotly.express as px
from plotly.graph_objects import Figure
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
from dash_daq import BooleanSwitch
import flask

# Data manipulation
import numpy as np
import pandas as pd

# Machine Learning Algorithms
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Helper Built-in libs
from pathlib import Path
from typing import Dict
import warnings

# Plotly uses a deprecated pandas function and it doesn't stop warning us about it during runtime
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.chained_assignment = None

# Custom dataset processing class
from data_processing import Dataset

# Initialize the path of the root directory
MAIN_DIR = Path(".").absolute().parent
BASEBALL_DIR = MAIN_DIR / "baseball_data"
POLIT_DIR = MAIN_DIR / "political_data"

# Set global vars
BATTING_FINAL_COLS = ['playerID', 'bbrefID', 'yearID', 'fullName', 'teamID', 'birthState', 'birthCountry', 'G', 'AB',
                      'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'HBP', 'SH', 'SF', 'WAR', 'oWAR',
                      'dWAR', 'Political Score']
PITCHING_FINAL_COLS = ['playerID', 'bbrefID', 'yearID', 'fullName', 'teamID', 'birthState', 'birthCountry', 'W', 'GS',
                       'G', 'SV', 'ER', 'SO', 'BB', 'ERA', 'BAOpp', 'WP', 'HR', 'gmLI', 'WAR', 'Political Score']

BATTING_NUM_COLS = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'HBP', 'SH', 'SF', 'WAR',
                    'oWAR', 'dWAR', 'Political Score']

PITCHING_NUM_COLS = ['W', 'GS', 'G', 'SV', 'ER', 'SO', 'BB', 'ERA',
                     'BAOpp', 'WP', 'HR', 'gmLI', 'WAR', 'Political Score']

BATTING_OPTIONS = {
    'HR': 'Home Runs',
    'H': 'Hits',
    'RBI': 'Runs Batted In',
    'R': 'Runs',
    'SB': 'Stolen Bases',
    'CS': 'Caught Stealing',
    'WAR': 'Wins Above Replacement',
    'oWAR': 'Offensive Wins Above Replacement',
    'dWAR': 'Defensive Wins Above Replacement',
    'SO': 'Strike Outs',
    'BB': 'Walks',
    'HBP': 'Hit by Pitch',
    'G': 'Games Played',
    'AB': 'At Bats',
    '2B': 'Doubles',
    '3B': 'Triples',
    'SH': 'Sacrifice Bunts',
    'SF': 'Sacrifice Flies'}

PITCHING_OPTIONS = {
    'W': 'Wins',
    'GS': 'Games Started',
    'G': 'Games',
    'SV': 'Saves',
    'ER': 'Earned Runs',
    'SO': 'Strike Outs',
    'BB': 'Walks',
    'BAOpp': 'Opponent Batting Average',
    'WP': 'Wild Pitches',
    'HR': 'Home Runs',
    'gmLI': 'Leverage Index',
    'WAR': 'Wins Above Replacement'}

# Read in simple counting stats for pitching and batting
batting_df = pd.read_csv(BASEBALL_DIR / "core" / "Batting.csv")
pitching_df = pd.read_csv(BASEBALL_DIR / "core" / "Pitching.csv")

# Read in the batting and pitching advanced stats
batting_advanced_df = pd.read_csv(BASEBALL_DIR / 'advanced' / 'batting_advanced.csv')
pitching_advanced_df = pd.read_csv(BASEBALL_DIR / 'advanced' / 'pitching_advanced.csv')

# Initialize the server
server = flask.Flask(__name__)
app = Dash(__name__,
           server=server,
           external_stylesheets=[dbc.themes.LUX])

# Initialize the datasets and perform processing
pitching_data = Dataset(pitching_df, pitching_advanced_df, False, PITCHING_FINAL_COLS, PITCHING_NUM_COLS)
batting_data = Dataset(batting_df, batting_advanced_df, True, BATTING_FINAL_COLS, BATTING_NUM_COLS)

# Pitching Scatter Plot
pitching_plot = px.scatter(pitching_data.counting_df,
                           x="G", y="WAR",
                           color="yearID",
                           range_color=[1899, 2022],
                           hover_name="fullName", hover_data=["yearID", "G"])

# Batting Scatter Plot
# Pitching Scatter Plot
batting_plot = px.scatter(batting_data.counting_df,
                          x="G", y="WAR",
                          color="yearID",
                          range_color=[1899, 2022],
                          hover_name="fullName", hover_data=["yearID", "G"])

# Pitching Correlation Between Statistics
pitching_correlation = pitching_data.counting_df.corr()
pitching_correlation_matrix = px.imshow(pitching_correlation, text_auto=True, aspect='auto', zmax=1, zmin=-1,
                                        title='Pitching Statistics Correlation Matrix',
                                        color_continuous_scale=px.colors.diverging.Fall)

# Batting Correlation Between Statistics
batting_correlation = batting_data.counting_df.corr()
batting_correlation_matrix = px.imshow(batting_correlation, text_auto=True, aspect='auto', zmax=1, zmin=-1,
                                       title='Pitching Statistics Correlation Matrix',
                                       color_continuous_scale=px.colors.diverging.Fall)

# Career Political Scores
pitching_career_political = pitching_data.counting_df.groupby("playerID").agg(
    {
        "Political Score": "max",
        "WAR": "sum",
        "fullName": "first"
    })
pitching_polit_plot = px.scatter(pitching_career_political, x="WAR", y="Political Score", hover_name="fullName",
                                 title="Political Score vs. Career Cumulative WAR<br><sup>Positive values are "
                                       "Republican-leaning, while negative values are Democrat-leaning</sup>")

batting_career_political = batting_data.counting_df.groupby("playerID").agg(
    {
        "Political Score": "max",
        "WAR": "sum",
        "fullName": "first"
    })
batting_polit_plot = px.scatter(batting_career_political, x="WAR", y="Political Score", hover_name="fullName",
                                title="Political Score vs. Career Cumulative WAR<br><sup>Positive values are "
                                      "Republican-leaning, while negative values are Democrat-leaning</sup>")

# KMeans Clustering
# Batting
kmeans = KMeans(
    n_clusters=4, init="k-means++",
    n_init=10,
    tol=1e-04, random_state=42
)

kmeans_batting = batting_data.scaled_df.copy()
kmeans.fit(kmeans_batting)
kmeans_batting['label'] = kmeans.labels_

# Flower Plot
kmeans_polar_batting = kmeans_batting.groupby("label").mean().reset_index()
kmeans_polar_batting = pd.melt(kmeans_polar_batting, id_vars=["label"])
batting_flower_kmeans = px.line_polar(kmeans_polar_batting, r="value", theta="variable", color="label",
                                      line_close=True, title='Batting Cluster Features - Kmeans')

# PCA visualization of highly dimensional clusters
pca_num_components = 2
reduced_data = PCA(n_components=pca_num_components).fit_transform(kmeans_batting.drop('label', axis=1))
batting_pca_results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
batting_cluster_kmeans = px.scatter(batting_pca_results, x="pca1", y="pca2", color=kmeans_batting['label'],
                                    hover_name=batting_data.counting_df['fullName'], title='Batting Cluster - Kmeans')

# Pitching
kmeans = KMeans(
    n_clusters=3, init="k-means++",
    n_init=10,
    tol=1e-04, random_state=42
)
kmeans_pitching = pitching_data.scaled_df.copy()
kmeans.fit(kmeans_pitching)
kmeans_pitching['label'] = kmeans.labels_

# Pitching flower plot
kmeans_polar_pitching = kmeans_pitching.groupby("label").mean().reset_index()
kmeans_polar_pitching = pd.melt(kmeans_polar_pitching, id_vars=["label"])
pitching_flower_kmeans = px.line_polar(kmeans_polar_pitching, r="value", theta="variable", color="label",
                                       line_close=True, title='Pitching Cluster Features - Kmeans')

# Pitching PCA
reduced_data = PCA(n_components=pca_num_components).fit_transform(
    kmeans_pitching.drop('label', axis=1))
pitching_pca_results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
pitching_cluster_kmeans = px.scatter(pitching_pca_results, x="pca1", y="pca2", title='Pitching Cluster - Kmeans',
                                     color=kmeans_pitching['label'],
                                     hover_name=pitching_data.counting_df['fullName'])

# BATTING
# Fit and predict clusters using GMM
gmm_batting = batting_data.scaled_df.copy()
gmm = GMM(n_components=4, random_state=42).fit(gmm_batting)
gmm_batting['label'] = gmm.predict(gmm_batting)

batting_cluster_gmm = px.scatter(batting_pca_results, x="pca1", y="pca2", color=gmm_batting['label'],
                                 hover_name=batting_data.counting_df['fullName'], title='Batting Cluster - GMM')

# Transform data for polar plot
gmm_polar_batting = gmm_batting.groupby("label").mean().reset_index()
gmm_polar_batting = pd.melt(gmm_polar_batting, id_vars=["label"])

# PITCHING
# Fit and predict clusters using GMM
gmm_pitching = pitching_data.scaled_df.copy()
gmm = GMM(n_components=3, random_state=42).fit(gmm_pitching)
gmm_pitching['label'] = gmm.predict(gmm_pitching)

pitching_cluster_gmm = px.scatter(pitching_pca_results, x="pca1", y="pca2", title='Pitching Cluster - GMM',
                                  color=gmm_pitching['label'],
                                  hover_name=pitching_data.counting_df['fullName'])

# Transform data for polar plot
gmm_polar_pitching = gmm_pitching.groupby("label").mean().reset_index()
gmm_polar_pitching = pd.melt(gmm_polar_pitching, id_vars=["label"])

# KNN Similarity Scoring
# Pitching
pitching_neighbors = NearestNeighbors(n_neighbors=6, n_jobs=-1)
pitching_neighbors.fit(pitching_data.scaled_df)
pitch_scores, pitch_neighbor_indices = pitching_neighbors.kneighbors(
    pitching_data.scaled_df.iloc[0, :].values.reshape(1, -1))

# Generate a formatted table for output
pitch_similarity_table = pitching_data.counting_df.iloc[pitch_neighbor_indices.flatten()]
pitch_similarity_table.loc[:, "Similarity"] = (np.exp(pitch_scores) ** -1).T
pitch_similarity_table = pitch_similarity_table[["Similarity", ] +
                                                [x for x in pitch_similarity_table.columns if x != "Similarity"]
                                                ].round(2)

# Batting
batting_neighbors = NearestNeighbors(n_neighbors=6, n_jobs=-1)
batting_neighbors.fit(batting_data.scaled_df)
bat_scores, bat_neighbor_indices = batting_neighbors.kneighbors(batting_data.scaled_df.iloc[0, :].values.reshape(1, -1))

# Generate a formatted table for output
bat_similarity_table = batting_data.counting_df.iloc[bat_neighbor_indices.flatten()]
bat_similarity_table.loc[:, "Similarity"] = (np.exp(bat_scores) ** -1).T
bat_similarity_table = bat_similarity_table[["Similarity", ] +
                                            [x for x in bat_similarity_table.columns if x != "Similarity"]
                                            ].round(2)


@app.callback(
    Output('pitching_plot', 'figure'),
    [Input('pitching-range-slider', 'value'),
     Input('pitching_dropdown_x', 'value'),
     Input('pitching_dropdown_y', 'value')])
def update_pitching(year_range, x_value, y_value):
    data = pitching_data.counting_df[(year_range[0] <= pitching_data.counting_df["yearID"]) &
                                     (pitching_data.counting_df["yearID"] <= year_range[1])]
    new_plot = px.scatter(data,
                          title='Pitching Statistics',
                          x=x_value, y=y_value,
                          color="yearID",
                          range_color=[1899, 2022],
                          hover_name="fullName", hover_data=["yearID", "G"])

    new_plot.update_layout(transition_duration=1000)

    return new_plot


@app.callback(
    Output('batting_plot', 'figure'),
    [Input('batting-range-slider', 'value'),
     Input('batting_dropdown_x', 'value'),
     Input('batting_dropdown_y', 'value')])
def update_hitting(year_range, x_value, y_value):
    data = batting_data.counting_df[(year_range[0] <= batting_data.counting_df["yearID"]) &
                                    (batting_data.counting_df["yearID"] <= year_range[1])]
    new_plot = px.scatter(data,
                          title='Batting Statistics',
                          x=x_value, y=y_value,
                          color="yearID",
                          range_color=[1899, 2022],
                          hover_name="fullName", hover_data=["yearID", "G"])

    new_plot.update_layout(transition_duration=1000)

    return new_plot


def generate_similarity_card(category: str, original_data: pd.DataFrame, similarity_table: pd.DataFrame):
    unwanted_columns = ["playerID", "bbrefID", "birthState", "birthCountry"]
    layout = html.Div(
        dbc.Card([
            html.H3(f"Similarity Scoring - {category}", className="mx-auto"),
            dbc.Row(children=[
                dbc.Col(children=[
                    'Player',
                    dcc.Dropdown(id=f"{category}_player_dropdown",
                                 options=original_data["fullName"].unique(),
                                 searchable=True,
                                 value=original_data["fullName"].unique()[0],
                                 placeholder='Please select...',
                                 clearable=True)
                ]),
                dbc.Col(children=[
                    'Year',
                    dcc.Dropdown(id=f"{category}_year_dropdown",
                                 options=original_data["yearID"].unique(),
                                 searchable=True,
                                 value=1899,
                                 placeholder='Please select...',
                                 clearable=True)
                ])], className="p-2"
            ),
            dash_table.DataTable(similarity_table.drop(unwanted_columns, axis=1).to_dict('records'),
                                 [{"name": i, "id": i} for i in similarity_table.drop(unwanted_columns,
                                                                                      axis=1).columns],
                                 id=f"{category}_table"),
        ], color="secondary"), className="w-75 mx-auto p-2"
    )
    return layout


def generate_super_plot(category: str, options: Dict[str, str], figure: Figure):
    """
    Helper function to generate the super-plots at the top of our page.
    """
    card_layout = html.Div(dbc.Card([
        dcc.Graph(id=f"{category}_plot", figure=figure),
        dcc.RangeSlider(1899, 2021, marks={x: str(x) for x in range(1899, 2022, 20)},
                        value=[1899, 2021],
                        updatemode='drag',
                        id=f"{category}-range-slider",
                        tooltip={'always_visible': True}),
        dbc.Row(children=[
            dbc.Col(children=[
                'X-Axis',
                dcc.Dropdown(id=f"{category}_dropdown_x",
                             options=options,
                             searchable=True,
                             value="G",
                             placeholder='Please select...',
                             clearable=True)
            ]),
            dbc.Col(children=[
                'Y-Axis',
                dcc.Dropdown(id=f"{category}_dropdown_y",
                             options=options,
                             searchable=True,
                             value="WAR",
                             placeholder='Please select...',
                             clearable=True)
            ])]
        )], color="secondary", className="p-2"), className="w-75 mx-auto p-2")
    return card_layout


def generate_card(name: str, figure: Figure):
    """
    Generates generic cards for each figure.
    """
    card_layout = html.Div(
        dbc.Card([
            dcc.Graph(id=name, figure=figure),
        ], color="secondary"), className="w-75 mx-auto p-2"
    )
    return card_layout


@app.callback(
    [Output('pitching_year_dropdown', 'options'),
     Output('pitching_year_dropdown', 'value')],
    Input('pitching_player_dropdown', 'value')
)
def set_pitching_years(player):
    pitching_years = pitching_data.counting_df[pitching_data.counting_df["fullName"] == player].loc[:,
                     "yearID"].unique()
    return pitching_years, pitching_years[0]


@app.callback(
    Output('pitching_table', 'data'),
    [Input('pitching_year_dropdown', 'value'),
     Input('pitching_player_dropdown', 'value')]
)
def update_pitching_similarity_table(year, player):
    unwanted_columns = ["playerID", "bbrefID", "birthState", "birthCountry"]
    player_index = pitching_data.counting_df[(pitching_data.counting_df["yearID"] == year) &
                                             (pitching_data.counting_df["fullName"] == player)].index.tolist()[0]
    pitch_scores, pitch_neighbor_indices = pitching_neighbors.kneighbors(
        pitching_data.scaled_df.iloc[player_index, :].values.reshape(1, -1))

    # Generate a formatted table for output
    pitch_similarity_table = pitching_data.counting_df.iloc[pitch_neighbor_indices.flatten()]
    pitch_similarity_table.loc[:, "Similarity"] = (np.exp(pitch_scores) ** -1).T
    pitch_similarity_table = pitch_similarity_table[["Similarity", ] +
                                                    [x for x in pitch_similarity_table.columns if x != "Similarity"]
                                                    ].round(2)

    return pitch_similarity_table.drop(unwanted_columns, axis=1).to_dict('records')


@app.callback(
    [Output('batting_year_dropdown', 'options'),
     Output('batting_year_dropdown', 'value')],
    Input('batting_player_dropdown', 'value')
)
def set_batting_years(player):
    batting_years = batting_data.counting_df[batting_data.counting_df["fullName"] == player].loc[:,
                     "yearID"].unique()
    return batting_years, batting_years[0]


@app.callback(
    Output('batting_table', 'data'),
    [Input('batting_year_dropdown', 'value'),
     Input('batting_player_dropdown', 'value')]
)
def update_batting_similarity_table(year, player):
    unwanted_columns = ["playerID", "bbrefID", "birthState", "birthCountry"]
    player_index = batting_data.counting_df[(batting_data.counting_df["yearID"] == year) &
                                             (batting_data.counting_df["fullName"] == player)].index.tolist()[0]
    bat_scores, bat_neighbor_indices = batting_neighbors.kneighbors(
        batting_data.scaled_df.iloc[player_index, :].values.reshape(1, -1))

    # Generate a formatted table for output
    bat_similarity_table = batting_data.counting_df.iloc[bat_neighbor_indices.flatten()]
    bat_similarity_table.loc[:, "Similarity"] = (np.exp(bat_scores) ** -1).T
    bat_similarity_table = bat_similarity_table[["Similarity", ] +
                                                    [x for x in bat_similarity_table.columns if x != "Similarity"]
                                                    ].round(2)

    return bat_similarity_table.drop(unwanted_columns, axis=1).to_dict('records')


@app.callback(
    [Output('Batting Cluster Flower - Kmeans', 'figure'),
     Output('Batting Cluster - Kmeans', 'figure')],
    [Input('cluster_switch_bat', 'on')]
)
def update_batting_clustering(event: bool):
    if not event:
        flower_plot = px.line_polar(gmm_polar_batting, r="value", theta="variable", color="label",
                                    line_close=True, title='Pitching Cluster Features - GMM')
        pca_plot = px.scatter(batting_pca_results, x="pca1", y="pca2", title='Batting Cluster - GMM',
                              color=gmm_batting['label'],
                              hover_name=batting_data.counting_df['fullName'])
    else:
        flower_plot = px.line_polar(kmeans_polar_batting, r="value", theta="variable", color="label",
                                    line_close=True, title='Batting Cluster Features - Kmeans')
        pca_plot = px.scatter(batting_pca_results, x="pca1", y="pca2", title='Batting Cluster - Kmeans',
                              color=kmeans_batting['label'],
                              hover_name=batting_data.counting_df['fullName'])

    return flower_plot, pca_plot


@app.callback(
    [Output('Pitching Cluster Flower - Kmeans', 'figure'),
     Output('Pitching Cluster - Kmeans', 'figure')],
    [Input('cluster_switch_pitch', 'on')]
)
def update_batting_clustering(event: bool):
    if not event:
        flower_plot = px.line_polar(gmm_polar_pitching, r="value", theta="variable", color="label",
                                    line_close=True, title='Pitching Cluster Features - GMM')
        pca_plot = px.scatter(pitching_pca_results, x="pca1", y="pca2", title='Pitching Cluster - GMM',
                              color=gmm_pitching['label'],
                              hover_name=pitching_data.counting_df['fullName'])
    else:
        flower_plot = px.line_polar(kmeans_polar_pitching, r="value", theta="variable", color="label",
                                    line_close=True, title='Pitching Cluster Features - Kmeans')
        pca_plot = px.scatter(pitching_pca_results, x="pca1", y="pca2", title='Pitching Cluster - Kmeans',
                              color=kmeans_pitching['label'],
                              hover_name=pitching_data.counting_df['fullName'])

    return flower_plot, pca_plot


@app.callback(
    Output('plots', 'children'),
    [Input('pitching_switch', 'on')])
def switch_catgories(event: bool):
    if event:
        plots = [
            generate_super_plot("pitching", PITCHING_OPTIONS, pitching_plot),
        generate_similarity_card("pitching", pitching_data.counting_df, pitch_similarity_table),
            generate_card("Pitching Political", pitching_polit_plot),
            generate_card("Pitching Correlation Matrix", figure=pitching_correlation_matrix),
            html.Div(BooleanSwitch(id="cluster_switch_pitch", on=True, label="Switch Clustering")),
            generate_card("Pitching Cluster Flower - Kmeans", figure=pitching_flower_kmeans),
            generate_card("Pitching Cluster - Kmeans", figure=pitching_cluster_kmeans)

        ]
    else:
        plots = [
            generate_super_plot("batting", BATTING_OPTIONS, batting_plot),
            generate_similarity_card("batting", batting_data.counting_df, bat_similarity_table),
            generate_card("Batting Political", batting_polit_plot),
            generate_card("Batting Correlation Matrix", figure=batting_correlation_matrix),
            html.Div(BooleanSwitch(id="cluster_switch_bat", on=True, label="Switch Clustering")),
            generate_card("Batting Cluster Flower - Kmeans", figure=batting_flower_kmeans),
            generate_card("Batting Cluster - Kmeans", figure=batting_cluster_kmeans),
        ]
    return plots


app.layout = html.Div(children=[
    html.H1(children="Baseball Dashboard", className="p-2"),
    html.A(children="Click here if you're bored...",
           href="https://www.youtube.com/watch?v=ECRcCIg0K50",
           className="p-2"),
    html.Div(BooleanSwitch(id="pitching_switch", on=True, label="Pitching Switch"), className="p-2"),

    html.Div(children=[
        generate_super_plot("pitching", PITCHING_OPTIONS, pitching_plot),
        generate_similarity_card("pitching", pitching_data.counting_df, pitch_similarity_table),
        generate_card("Pitching Political", pitching_polit_plot),
        generate_card("Pitching Correlation Matrix", figure=pitching_correlation_matrix),
        html.Div(BooleanSwitch(id="cluster_switch_pitch", on=True, label="Switch Clustering")),
        generate_card("Pitching Cluster Flower - Kmeans", figure=pitching_flower_kmeans),
        generate_card("Pitching Cluster - Kmeans", figure=pitching_cluster_kmeans)

    ], id="plots")

], )
if __name__ == "__main__":
    app.run_server(debug=True)
