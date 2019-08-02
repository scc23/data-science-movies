# CMPT 353 Project

https://coursys.sfu.ca/2019su-cmpt-353-d1/pages/Project<br/>
https://coursys.sfu.ca/2019su-cmpt-353-d1/pages/ProjectMovies

## Group Members

Sherman Chao Wen Chow (301232684)<br/>
Ahsan Naveed (301228556)<br/>

## Required Libraries

- Pandas
- NumPy
- matplotlib
- scipy
- sklearn
- nltk (our code will use this library to download stop words to be used for NLP)
- wordcloud

## How to Run Code

All of our code is located in the main project directory.

### 1. Movie Success Criteria Correlation

Run the following command: 'python3 movies_correlation.py'<br/>

- This Python script will output correlation coefficients of various movie success criteria and display a scatterplot.

### 2. Movie Genre Predictions

Run the following command: 'python3 plot_summary_predict.py'<br/>

- This Python script will clean our data, train a model, perform some analysis, and output the f1 score of our movie genre predictions as well as a couple of example genre predictions.

### 3. Movie Success Predictions

Run the following command: 'python3 review_scores_predict.py'<br/>

- This Python script will print out the MLP Regressor model score after being trained on genre based data. Different datasets can be uncommented to train model on dataset with features other than the genre.
