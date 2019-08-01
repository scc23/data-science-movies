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
from sklearn.preprocessing import MultiLabelBinarizer


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

    # Convert genres DataFrame to a dictionary of genre_code:genre_label pairs.
    genre_map = pd.Series(genres.genre_label.values,index=genres.wikidata_id).to_dict()

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
    
    # Clean data.
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
    # movies_data['omdb_text'] = movies_data['omdb_plot'].apply(lambda x: ' '.join(x))
    plot_summaries_words = movies_data['omdb_plot'].apply(lambda x: ' '.join(x))

    # Create and generate word cloud image.
    # summaries_text = plot_summaries_words.values
    # wordcloud = WordCloud(
    #     width=800,
    #     height=400,
    #     background_color='white').generate(str(summaries_text))
    # Display the generated word cloud.
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.tight_layout(pad=0)
    # plt.show()

    # Get all the genres from the movies.
    genres_all = []
    for index, row in movies_data.iterrows():
        genres_all.append(row['genre'])

    # Flatten the genres list.
    genres = []
    for sublist in genres_all:
        for item in sublist:
            genres.append(item)

    # Get the distinct genres.
    genres = list(set(genres))
    
    # This DataFrame will separate all the distinct genres into separate columns, and show which movie is associated with which individual genre.
    movies_data2 = movies_data
    for genre_code in genres:
        movies_data2[genre_code] = 0

    for index, row in movies_data2.iterrows():
        for genre_code in row['genre']:
            movies_data2.loc[index, genre_code] = movies_data2.loc[index, genre_code] + 1

    # print(movies_data2)

    # Get the number of counts for each genre.
    genres_count = []
    for col in movies_data2.columns[2:]:
        genres_count.append(movies_data2[col].sum())

    # Get the English label names for each genre.
    genre_labels = []
    for label in genres:
        genre_labels.append(genre_map.get(label))

    # print(genre_labels)

    # print(len(genres))
    # print(len(genres_count))
    

    # multilabel_binarizer = MultiLabelBinarizer()


    # movies_data2.to_csv('genre_predictions.csv', index=False)
    # movies_data.to_csv('genre_predictions.csv', index=False)
    

if __name__ == "__main__":
    main()
