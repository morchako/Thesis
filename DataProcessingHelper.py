import pandas as pd
from sklearn.metrics import pairwise_distances

def createDestanceMatrix(df):
    # Select the columns to use for calculating distances
    X = df[['column1', 'column2', 'column3']]
    # Calculate the distance matrix
    distance_matrix = pairwise_distances(X)
    return distance_matrix

def normalize(df):
    result = df.copy()
    # for feature_name in df.columns:
    #     max_value = df[feature_name].max()
    #     min_value = df[feature_name].min()
    #     result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    for feature_name in df.columns:
        s = result[feature_name]
        s[s<6000] = 0
        s[s>=6000] = 1
    return result

def get_inner_hem_matrix(df):
    # Get the dimensions of the DataFrame
    rows, cols = df.shape

    # Initialize the inner hem matrix with all zeros
    inner_hem_matrix = [[0 for j in range(cols // 2)] for i in range(rows // 2)]

    # Iterate through the DataFrame and fill in the inner hem matrix
    for i in range(rows // 2):
        for j in range(cols // 2):
            inner_hem_matrix[i][j] = df.iloc[i, j]

    return inner_hem_matrix

def inner_multiplication(df):
#   matrix = df.values
    return df.dot(df.T)
#    return matrix@matrix.T

