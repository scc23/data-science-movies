# Predict movie genres based on their plot summaries (NLP).

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import nltk

nltk.download('stopwords')
nltk.download('punkt')


# Function that removes punctuations from the plot summaries
def remove_punctuations(summary):
    for punc in string.punctuation:
        summary = summary.replace(punc, '')
    return summary

# Function that tokenizes the plot summaries into words.
def tokenize(x):
    tokens = word_tokenize(x)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

# Function that generates a word cloud of the most common words.
def generate_word_cloud(plot_summaries_words):
    print('Generating word cloud image...')
    summaries_text = plot_summaries_words.values
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white').generate(str(summaries_text))
    # Display the generated word cloud.
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Main function.
def main():
    # Read JSON files into Pandas DataFrames
    print('Reading data into DataFrames...')
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
    print('Cleaning data...')
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
    movies_data['clean_summary'] = movies_data['omdb_plot'].apply(lambda x: ' '.join(x))
    plot_summaries_words = movies_data['omdb_plot'].apply(lambda x: ' '.join(x))

    # Create and generate word cloud image.
    # generate_word_cloud(plot_summaries_words); return

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

    # Get the number of counts for each genre.
    # genres_count = []
    # for col in movies_data2.columns[2:]:
    #     genres_count.append(movies_data2[col].sum())

    # Get the English label names for each genre.
    # genre_labels = []
    # for label in genres:
    #     genre_labels.append(genre_map.get(label))
    # print(genre_labels)

    print('Converting plot summaries to features...')
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(movies_data2['genre'])

    # Extract features from cleaned plot summaries by usng tf-idf.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=500)  # Use the 500 most frequent words in the data as features.

    # Split data into train and validation data sets.
    X = movies_data2['clean_summary']
    y = multilabel_binarizer.transform(movies_data2['genre'])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=9)
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # Create tf-idf features.
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid)

    # Build the genre prediction model.
    print('Building prediction model...')
    lr = LogisticRegression()
    model = OneVsRestClassifier(lr)

    # Train the model.
    print('Training model...')
    model.fit(X_train_tfidf, y_train)

    # Print the score of the model on a validation subset of the data.
    # print('Score of the model: {}'.format(model.score(X_valid_tfidf, y_valid)))

    # Predict on validation data set.
    print('Making predictions...')
    y_prediction = model.predict_proba(X_valid_tfidf)
    t = 0.25 # Threshold value.
    y_prediction = (y_prediction >= t).astype(int)
    predictions = multilabel_binarizer.inverse_transform(y_prediction)
    # print(predictions)
    res = pd.Series(predictions)
    
    print(res)
    
    print('\nf1 score: {}\n'.format(f1_score(y_valid, y_prediction, average="micro")))


    # Show 10 genre predictions, and compare it with the actual genres.
    def make_predictions(data):
        data_tfidf = tfidf_vectorizer.transform([data])
        data_prediction = model.predict_proba(data_tfidf)
        t = 0.25
        data_prediction = (data_prediction >= t).astype(int)
        return multilabel_binarizer.inverse_transform(data_prediction)

    for i in range(10):
        data = X_valid.sample(1).index[0]

        predicted_genre = make_predictions(X_valid[data])
        actual_genre = movies_data2['genre'][data]

        predicted_genre_labels = []
        for code_set in predicted_genre:
            for code in code_set:
                predicted_genre_labels.append(genre_map.get(code))
        
        actual_genre_labels = []
        for code in actual_genre:
            actual_genre_labels.append(genre_map.get(code))

        print('IMDB ID: {}'.format(data))
        print('\tPredicted genre: {}\n\tActual genre: {}\n'.format(predicted_genre_labels, actual_genre_labels))

    
if __name__ == "__main__":
    main()
