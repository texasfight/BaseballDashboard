import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
from typing import Union, List


class Scraper:
    def __init__(self, category: Union[str, List[str]]):
        self.category = category

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
            advanced_df = None
            for hidden_content in page_soup.find_all(text=lambda text: isinstance(text, Comment)):
                # Parsing comments for the hidden table
                hidden_soup = BeautifulSoup(hidden_content, "lxml")
                table_soup = hidden_soup.find('table', {"id": label})
                if table_soup:
                    # We found the hidden table! Read in as a DataFrame
                    advanced_df = pd.read_html(str(table_soup))[0]
                    # Add unique bbref ID
                    advanced_df["bbrefID"] = player_id

                    # remove garbage "years" and rename to match the given data
                    advanced_df = advanced_df.rename({"Year": "yearID"}, axis=1)
                    advanced_df["yearID"] = pd.to_numeric(advanced_df["yearID"], errors="coerce", downcast="integer")
                    advanced_df = advanced_df.dropna(subset="yearID")
                    break
            # We want to make sure that we return `None` if we don't get any hits for a given table type
            output.append(advanced_df)
        return output
