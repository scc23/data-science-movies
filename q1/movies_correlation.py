# Question:
# Do the various criteria for success (critic reviews, audience reviews, profit/loss) correlate with each other?
# Is there something you can say about better or worse kinds of “success”?

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def main():
    # Read JSON files into Pandas DataFrames
    omdb_filename = "../movies/data/omdb-data.json.gz"
    rotten_filename = "../movies/data/rotten-tomatoes.json.gz"
    wikidata_filename = "../movies/data/wikidata-movies.json.gz"
    omdb = pd.read_json(omdb_filename, lines=True)
    rotten = pd.read_json(rotten_filename, lines=True)
    wikidata = pd.read_json(wikidata_filename, lines=True)
    
    # Clean data by extracting only the necessary columns.
    # print(omdb.columns.values)
    omdb = omdb[
        ['imdb_id',
         'omdb_awards'          # awards won: text describing awards won by the movie
        ]
    ]
    omdb = omdb.sort_values(by=['imdb_id'])
    omdb = omdb.set_index('imdb_id')
    
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
    rotten = rotten.sort_values(by=['imdb_id'])
    rotten = rotten.set_index('imdb_id')

    # print(wikidata.columns.values)
    wikidata = wikidata[
        ['imdb_id',
         'made_profit'          # made profit? Boolean calculated from 'cost' and 'box office'
        ]
    ]
    wikidata = wikidata.sort_values(by=['imdb_id'])
    wikidata = wikidata.set_index('imdb_id')


    # Join the DataFrames by index (imdb_id)
    movies = omdb.join(rotten).join(wikidata)

    movies['critic_average'] = movies['critic_average'].apply(lambda x: x / 2)
    # movies['audience_reviews'] = movies[['audience_average', 'audience_percent']].apply(tuple, axis=1)          # tuple of [audience_average, audience_percent]
    # movies['critic_reviews'] = movies[['critic_average', 'critic_percent']].apply(tuple, axis=1)                # tuple of [critic_average, critic_percent]
    # movies = movies.drop(columns=['audience_ratings', 'audience_average', 'audience_percent', 'critic_average', 'critic_percent'])

    # critic_reviews_vs_audience_reviews = movies[['critic_reviews', 'audience_reviews']]
    # Remove rows with NaN values
    # critic_reviews_vs_audience_reviews = critic_reviews_vs_audience_reviews[~critic_reviews_vs_audience_reviews.critic_reviews.apply(lambda x: np.isnan(x[0]) & np.isnan(x[1]))]
    # critic_reviews_vs_audience_reviews = critic_reviews_vs_audience_reviews[~critic_reviews_vs_audience_reviews.audience_reviews.apply(lambda x: np.isnan(x[0]) & np.isnan(x[1]))]

    # critic_review_vs_profit = movies[['critic_reviews', 'audience_reviews']]
    # audience_reviews_vs_profit = movies[['audience_reviews', 'made_profit']]
    

    # Compute the correlation coefficients
    critic_percent_vs_audience_reviews = movies[['critic_percent', 'audience_percent']].dropna()
    critic_percent_vs_made_profit = movies[['critic_percent', 'made_profit']].dropna()
    audience_percent_vs_made_profit = movies[['audience_percent', 'made_profit']].dropna()

    critic_reviews_vs_audience_reviews = stats.linregress(critic_percent_vs_audience_reviews['critic_percent'], critic_percent_vs_audience_reviews['audience_percent']).rvalue
    critic_reviews_vs_profit = stats.linregress(critic_percent_vs_made_profit['critic_percent'], critic_percent_vs_made_profit['made_profit']).rvalue
    audience_reviews_vs_profit = stats.linregress(audience_percent_vs_made_profit['audience_percent'], audience_percent_vs_made_profit['made_profit']).rvalue

    print("Correlation coefficient between critic reviews and audience reviews: {}".format(critic_reviews_vs_audience_reviews))
    print("Correlation coefficient between critic reviews and profit/loss: {}".format(critic_reviews_vs_profit))
    print("Correlation coefficient between audience reviews and profit/loss: {}".format(audience_reviews_vs_profit))


if __name__ == "__main__":
    main()
