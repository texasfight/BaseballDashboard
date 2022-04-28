import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from typing import List

# Initialize the path of the root directory
MAIN_DIR = Path(".").absolute().parent
BASEBALL_DIR = MAIN_DIR / "baseball_data"
POLIT_DIR = MAIN_DIR / "political_data"


class Dataset:
    """
    Holding class for processing the data of counting stats for pitching or batting
    """

    # Read in team and personal metadata
    teams_df = pd.read_csv(BASEBALL_DIR / "core" / "Teams.csv")
    players_df = pd.read_csv(BASEBALL_DIR / "core" / "People.csv")

    # Read in political data and quickly re-format it
    pres_df = pd.read_csv(POLIT_DIR / "Presidents.csv").rename({"Year": "yearID"}, axis=1).drop("Republican", axis=1)
    congress_df = pd.read_csv(POLIT_DIR / "Representatives.csv").rename({"Year": "yearID"}, axis=1)
    senators_df = pd.read_csv(POLIT_DIR / "Senators.csv").rename({"Year": "yearID"}, axis=1)

    def __init__(self,
                 counting_df: pd.DataFrame, advanced_df: pd.DataFrame,
                 batting: bool, final_columns: List[str], numerical_columns: List[str],
                 base_year: int = 1899):
        if batting:
            qualifier = "PA"
            qual_threshold = "minAB"
            # Calculate batting plate appearances
            counting_df[qualifier] = counting_df[["AB", "HBP", "BB", "SF", "SH"]].sum(axis=1, skipna=True)
            self.teams_df[qual_threshold] = 3.1 * self.teams_df['G']
        else:
            qualifier = "IPouts"
            qual_threshold = "minOUT"
            self.teams_df[qual_threshold] = self.teams_df["G"] * 3

        # Remove older years
        self.counting_df = counting_df[counting_df["yearID"] >= base_year]

        # Merge in the team info to only have qualifying players
        self.counting_df = pd.merge(self.counting_df,
                                    self.teams_df[[qual_threshold, 'yearID', 'teamID']],
                                    on=['yearID', 'teamID'])
        self.counting_df = self.counting_df[self.counting_df[qual_threshold] < self.counting_df[qualifier]]

        # Get the name information for each player
        self.counting_df = pd.merge(self.counting_df, self.players_df, on="playerID")
        self.counting_df["fullName"] = self.counting_df["nameFirst"] + " " + self.counting_df["nameLast"]

        # Merge in advanced stats and drop duplicates between tables
        if batting:
            self.counting_df = pd.merge(self.counting_df,
                                        advanced_df.drop("G", axis=1),
                                        on=['yearID', 'bbrefID'])
        else:
            self.counting_df = pd.merge(self.counting_df,
                                        advanced_df.drop(['G', 'GS'], axis=1),
                                        on=['yearID', 'bbrefID'])

        political_df = pd.merge(self.counting_df, self.pres_df, on="yearID").rename({"Democrat": "demPres"},
                                                                                    axis=1)
        political_df = pd.merge(political_df,
                                self.senators_df[["yearID", "% Democrats"]],
                                on="yearID").rename({"% Democrats": "demSenate"}, axis=1)

        political_df = pd.merge(political_df,
                                self.congress_df[["yearID", "% Democrats"]],
                                on="yearID").rename({"% Democrats": "demCongress"}, axis=1)

        idx = pd.MultiIndex.from_product(
            (political_df["playerID"].unique(), political_df['demPres'].unique()),
            names=["playerID", "demPres"])

        pres_split = political_df.groupby(["playerID", "demPres"]).agg({"WAR": "mean"}).reindex(
            idx)
        pres_split = pres_split.fillna(0)
        pres_score = pres_split.groupby("playerID").agg(np.subtract.reduce)
        pres_score.columns = ['Political Score']

        self.counting_df = pd.merge(self.counting_df, pres_score, on="playerID")

        # Cut down to only the final columns we care about for analysis
        self.counting_df = self.counting_df[final_columns]

        # Do the scaling and numerical breakdown for modelling
        self.numerical_df = self.counting_df[numerical_columns].fillna(value=0, axis=0)
        self.scaler = StandardScaler()
        self.scaled_df = pd.DataFrame(self.scaler.fit_transform(self.numerical_df), columns=numerical_columns)
