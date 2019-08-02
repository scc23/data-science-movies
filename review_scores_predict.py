# Question:
# Can you predict review scores from other data we have about the movie? Maybe genre
# determines a lot about the success of a movie? Or maybe the actors?

# TODO:
# add a seaborn genre categorical graph (see reference below)
# look into PCA for dimensionality reductions
# look into classification_report, confusion_matrix

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

OUTPUT_TEMPLATE = (
    # 'MLP regressor (genre):      {mlp_reg_genre:.3g}'
    # 'MLP regressor (coo):        {mlp_reg_coo:.3g}'
    'MLP regressor (cast):       {mlp_reg_cast:.3g}'
)


def get_data(file):
    return pd.read_json(file, lines=True)


def normalize_audience_percent(df):
    return (df['audience_percent'] // 10).astype(int)


def main():
    # Read JSON files into Pandas DataFrames
    rotten_filename = "../movies/data/rotten-tomatoes.json.gz"
    wikidata_filename = "../movies/data/wikidata-movies.json.gz"

    # create data frames
    rotten = get_data(rotten_filename)
    wiki = get_data(wikidata_filename)

    # merge wikidata and rottendata
    wiki_rotten = pd.merge(
        left=wiki,
        right=rotten,
        left_on='rotten_tomatoes_id',
        right_on='rotten_tomatoes_id'
    )

    # print(wiki_rotten.isnull().sum())

    # get country_of_origin
    by_coo = wiki_rotten[[
        'wikidata_id',
        'rotten_tomatoes_id',
        'country_of_origin',
        'audience_percent'
    ]]

    by_cast = wiki_rotten[[
        'wikidata_id',
        'rotten_tomatoes_id',
        'cast_member',
        'audience_percent'
    ]]

    ###################################################################################################
    ################################# train an MLP Regressor on genre #################################
    ###################################################################################################

    # drop unwanted columns i.e. columns that have a lot of null values
    wiki_rotten = wiki_rotten[[
        'wikidata_id',
        'rotten_tomatoes_id',
        'audience_percent',
        'genre'
    ]]

    # drop rows with null values
    wiki_rotten = wiki_rotten.dropna().reset_index(drop=True)

    # implemented from: https: // stackoverflow.com/questions/52989660/pandas-python-how-to-one-hot-encode-a-column-that-contains-an-array-of-strings
    # one-hot encode the genre column
    mlb = MultiLabelBinarizer()
    wiki_rotten = wiki_rotten.join(
        pd.DataFrame(
            mlb.fit_transform(wiki_rotten['genre']),
            columns=mlb.classes_
        )
    ).drop(['genre'], axis=1)

    # converst percentages to numbers ranging b/w 0-10
    wiki_rotten['audience_percent'] = normalize_audience_percent(wiki_rotten)

    # excluding wikidata_id, rotten_tomatoes_id, &, genre from X_columns
    X_columns_genre = list(wiki_rotten.columns.values)[3:]
    y_column_genre = 'audience_percent'

    X_genre = wiki_rotten[X_columns_genre]
    y_genre = wiki_rotten[y_column_genre]

    X_train_genre, X_valid_genre, y_train_genre, y_valid_genre = train_test_split(
        X_genre, y_genre)

    # # mlp regressor
    # neurons_per_layer = len(X_columns_genre)
    # mlp_reg_genre = MLPRegressor(
    #     hidden_layer_sizes=(neurons_per_layer, neurons_per_layer),
    #     activation='logistic'
    # )
    # mlp_reg_genre.fit(X_train_genre, y_train_genre)

    ###################################################################################################
    ########################### train an MLP Regressor on country_of_origin ###########################
    ###################################################################################################

    by_coo = by_coo.dropna().reset_index(drop=True)

    # one hot encode the by_coo dataframe
    one_hot = pd.get_dummies(by_coo['country_of_origin'])
    by_coo = by_coo.drop('country_of_origin', axis=1)
    by_coo = by_coo.join(one_hot)
    by_coo['audience_percent'] = normalize_audience_percent(by_coo)

    X_columns_coo = list(by_coo.columns.values)[3:]
    y_column_coo = 'audience_percent'

    X_coo = by_coo[X_columns_coo]
    y_coo = by_coo[y_column_coo]

    X_train_coo, X_valid_coo, y_train_coo, y_valid_coo = train_test_split(
        X_coo, y_coo)

    # # mlp regressor
    # neurons_per_layer = len(X_columns_coo)
    # mlp_reg_coo = MLPRegressor(hidden_layer_sizes=(
    #     neurons_per_layer, neurons_per_layer))
    # mlp_reg_coo.fit(X_train_coo, y_train_coo)

    ###################################################################################################
    ############################## train an MLP Regressor on cast_member ##############################
    ###################################################################################################

    by_cast = by_cast.dropna().reset_index(drop=True)
    by_cast = by_cast.join(
        pd.DataFrame(
            mlb.fit_transform(by_cast['cast_member']),
            columns=mlb.classes_
        )
    ).drop(['cast_member'], axis=1)
    by_cast['audience_percent'] = normalize_audience_percent(by_cast)

    X_columns_cast = list(by_cast.columns.values)[3:]
    y_column_cast = 'audience_percent'

    X_cast = by_cast[X_columns_cast]
    y_cast = by_cast[y_column_cast]

    X_train_cast, X_valid_cast, y_train_cast, y_valid_cast = train_test_split(
        X_cast, y_cast)

    # mlp regressor
    neurons_per_layer_cast = len(X_columns_cast)
    mlp_reg_cast = MLPRegressor(hidden_layer_sizes=(
        neurons_per_layer_cast, neurons_per_layer_cast))
    mlp_reg_cast.fit(X_train_cast, y_train_cast)

    print(OUTPUT_TEMPLATE.format(
        # mlp_reg_genre=mlp_reg_genre.score(X_valid_genre, y_valid_genre)
        # mlp_reg_coo=mlp_reg_coo.score(X_valid_coo, y_valid_coo)
        mlp_reg_cast=mlp_reg_cast.score(X_valid_cast, y_valid_cast)
    ))


if __name__ == "__main__":
    main()


# References:
# 1. https://www.datacamp.com/community/tutorials/categorical-data
# 2. https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
# 3. https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
# Bayesian classifier: 0.00185
# kNN classifier:      0.0148 (10 Neighbors)
# SVM classifier:      0.181
# MLP classifier:      0.0247
# MLP regressor:       0.229 (with just genre)
# MLP regressor (coo): 0.0397
