# Predict the success of movies by popularity based on plot summaries (NLP).

# 1. Predict genres from plot summaries.
# 2. Analyze the trends of movie genres over time.
# 3. Correlate the genres to the current trend to observe which genres are popular.


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import string
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def remove_punctuations(summary):
    for punc in string.punctuation:
        summary = summary.replace(punc, '')
    return summary

def tokenize(x):
    tokens = word_tokenize(x)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def main():
    # Read JSON files into Pandas DataFrames
    omdb_filename = "./movies/data/omdb-data.json.gz"
    rotten_filename = "./movies/data/rotten-tomatoes.json.gz"
    wikidata_filename = "./movies/data/wikidata-movies.json.gz"
    genres_filename = "./movies/data/genres.json.gz"
    omdb = pd.read_json(omdb_filename, lines=True)
    rotten = pd.read_json(rotten_filename, lines=True)
    wikidata = pd.read_json(wikidata_filename, lines=True)
    genres = pd.read_json(genres_filename, lines=True)

    # Create DataFrame of plot summaries with corresponding imdb id
    plot_summaries = omdb[['imdb_id', 'omdb_plot']]
    plot_summaries = plot_summaries.sort_values(by=['imdb_id'])
    plot_summaries = plot_summaries.set_index('imdb_id')

    wikidata = wikidata.sort_values(by=['imdb_id'])
    wikidata = wikidata.set_index('imdb_id')
    wikidata = wikidata[
        [
         # 'publication_date',
         # 'wikidata_id',
         'genre'
        ]
    ]
    # Drop the entries that have no publication date.
    # wikidata = wikidata.dropna()
    movies_data = pd.merge(wikidata, plot_summaries, on='imdb_id')
    # Remove movies with no plot summary.
    movies_data = movies_data[movies_data['omdb_plot'] != 'N/A']
    # Convert plot summaries to lowercase.
    movies_data['omdb_plot'] = movies_data['omdb_plot'].str.lower()
    # Remove all punctuations in plot summaries.
    movies_data['omdb_plot'] = movies_data['omdb_plot'].apply(remove_punctuations)
    # Tokenize strings
    movies_data['omdb_plot'] = movies_data['omdb_plot'].apply(tokenize)
    # Remove stop words.
    stop_words = stopwords.words('english')
    stop_words.append('platform')
    stop_words.append('film')
    movies_data['omdb_plot'] = movies_data['omdb_plot'].apply(lambda x: [word for word in x if word not in stop_words])
    movies_data['omdb_text'] = movies_data['omdb_plot'].apply(lambda x: ' '.join(x))

    # Create and generate word cloud image.
    omdb_text = movies_data['omdb_text'].values
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white').generate(str(omdb_text))
    # Display the generated word cloud.
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    # print(movies_data)

    

    movies_data.to_csv('genre_predictions.csv', index=False)
    


if __name__ == "__main__":
    main()
