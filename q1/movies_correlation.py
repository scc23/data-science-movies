# Question:
# Do the various criteria for success (critic reviews, audience reviews, profit/loss) correlate with each other?
# Is there something you can say about better or worse kinds of “success”?

import sys
import numpy as np
import pandas as pd


def main():
    # Read JSON files into Pandas DataFrames
    omdb_filename = "../movies/data/omdb-data.json.gz"
    rotten_filename = "../movies/data/rotten-tomatoes.json.gz"
    wikidata_filename = "../movies/data/wikidata-movies.json.gz"
    omdb = pd.read_json(omdb_filename, lines=True)
    rotten = pd.read_json(rotten_filename, lines=True)
    wikidata = pd.read_json(wikidata_filename, lines=True)
    
    # Clean data
    # print(omdb.columns.values)
    omdb = omdb[
        ['imdb_id',
         'omdb_awards'          # awards won: text describing awards won by the movie
        ]
    ]
    # print(omdb)
    
    # print(rotten.columns.values)
    rotten = rotten[
        ['imdb_id',
         'audience_ratings',    # audience ratings: the count of audience reviews
         'audience_average',    # audience average rating (out of 5)
         'audience_percent',    # audience percent who "liked it" (out of 100)
         'critic_average',      # critic average rating (out of 10)
         'critic_percent'       # critic percent who gave a positive review (out of 100)
        ]
    ]
    # print(rotten)

    # print(wikidata.columns.values)
    wikidata = wikidata[
        ['imdb_id',
         'made_profit'          # made profit? Boolean calculated from 'cost' and 'box office'
        ]
    ]
    # print(wikidata)


if __name__ == "__main__":
    main()
