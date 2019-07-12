import sys
import numpy as np
import pandas as pd


def main():
    omdb_filename = "../movies/data/omdb-data.json.gz"
    rotten_filename = "../movies/data/rotten-tomatoes.json.gz"
    wikidata_filename = "../movies/data/wikidata-movies.json.gz"
    omdb = pd.read_json(omdb_filename, lines=True)
    rotten = pd.read_json(rotten_filename, lines=True)
    wikidata = pd.read_json(wikidata_filename, lines=True)
    print(omdb.columns.values)
    print()
    print(rotten.columns.values)
    print()
    print(wikidata.columns.values)


if __name__ == "__main__":
    main()
