import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from typing import Union, List


class Scraper:
    def __init__(self, category: Union[str, List[str]]):
        if isinstance(category, str):
            self.category = [category, ]
        else:
            self.category = category

    @staticmethod
    def generate_dataframe(soup: BeautifulSoup, player_id: str):
        # We found the hidden table! Read in as a DataFrame
        df = pd.read_html(str(soup))[0]
        # Add unique bbref ID
        df["bbrefID"] = player_id

        # remove garbage "years" and rename to match the given data
        df = df.rename({"Year": "yearID"}, axis=1)
        df["yearID"] = pd.to_numeric(df["yearID"], errors="coerce", downcast="integer")
        df = df.dropna(subset="yearID")
        df.loc[:, "yearID"] = df.loc[:, "yearID"].astype("int16")

        return df

    def parse_data(self, player_id: str) -> List[pd.DataFrame]:
        """
        player_id: unique ID on BBRef
        category: string or list of strings for the hidden tables we need to parse for
        Given a player_id, grabs the BaseballReference HTML and parses for advanced stats
        """
        # Grab and process HTML for player's URL
        url = f'https://www.baseball-reference.com/players/{player_id[0]}/{player_id}.shtml'
        webpage = requests.get(url)
        page_soup = BeautifulSoup(webpage.content, 'lxml')

        # Because we can have multiple categories, output needs to be a list
        output = []
        for label in self.category:
            # We want to make sure that we return `None` if we don't get any hits for a given table type
            advanced_df = None
            # Pray that the table isn't hidden away in a comment
            table_soup = page_soup.find('table', {"id": label})
            if table_soup:
                advanced_df = self.generate_dataframe(table_soup, player_id)
                output.append(advanced_df)
                continue

            # Accept the crushing reality that it is hidden and iterate through all comments
            for hidden_content in page_soup.find_all(text=lambda text: isinstance(text, Comment)):
                # Parsing each comment back into its own soup
                hidden_soup = BeautifulSoup(hidden_content, "lxml")
                try:
                    # Break early if we find the table
                    table_soup = hidden_soup.find_all('table', {"id": label})[0]
                    advanced_df = self.generate_dataframe(table_soup, player_id)
                    break
                except IndexError:
                    continue

            output.append(advanced_df)

        return output
