import scrapy
import pandas as pd

player_ids = pd.read_csv("../../player_ids.csv").to_numpy()

class QuotesSpider(scrapy.Spider):
    name = "bbref"
    start_urls = [f'https://www.baseball-reference.com/players/{x[0]}/{x}.shtml' for x in player_ids]

    def parse(self, response, **kwargs):

        self.log(f'Saved file {response.url}')