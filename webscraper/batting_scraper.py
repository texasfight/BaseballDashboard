from bs4 import MarkupResemblesLocatorWarning
import pandas as pd
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from webscraper import Scraper

import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning, module='bs4')


if __name__ == "__main__":
    # Set scraper to grab advanced hitting stats
    webscraper = Scraper("batting_value")

    # Read in player IDs
    player_ids = pd.read_csv("batting_ids.csv").to_numpy().flatten()

    # Initialize multiprocessor workload, bypassing GIL
    with Pool(cpu_count() * 2) as thread_pool:
        # Output of parse_data is iterable, so we need to just grab the first item since
        # we're only grabbing batting data, then re-combine
        output = [x[0] for x in thread_pool.map(webscraper.parse_data, player_ids)]
        final_df = pd.concat(output, axis=0)

    # Export final batting statistics
    final_df.to_csv("../baseball_data/advanced/batting_advanced.csv", index=False)
