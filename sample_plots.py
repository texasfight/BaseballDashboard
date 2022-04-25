import plotly.express as px
import pandas as pd

from pathlib import Path

MAIN_DIR = Path(".").absolute()
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

stolen_homer_plot.show()

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

strikeout_era_plot.show()
