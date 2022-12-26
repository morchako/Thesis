import pandas as pd
from sklearn.metrics import pairwise_distances

def createDestanceMatrix(df):
    # Select the columns to use for calculating distances
    X = df[['column1', 'column2', 'column3']]
    # Calculate the distance matrix
    distance_matrix = pairwise_distances(X)
    return distance_matrix