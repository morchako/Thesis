import pandas as pd
import numpy as np
from ClassificationAlgorithms import *

def ReproducingResult_LASSO_SVM_kfold(dataset,pred_col):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Load the data into a pandas DataFrame
    df = dataset

    # Extract the features and target
    X = df.drop(pred_col, axis=1)
    y = df[pred_col]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert X back to a pandas.DataFrame
    X = pd.DataFrame(X, columns=df.drop(pred_col, axis=1).columns)

    # Define the number of folds for cross-validation
    n_folds = 10

    # Define the LASSO model
    lasso = Lasso(alpha=1.0)

    # Use cross-validation to evaluate the accuracy of the LASSO model for different feature subsets
    scores = []
    feature_names = X.columns.tolist()
    for i in range(X.shape[1]):
        score = np.average(cross_val_score(lasso, X.iloc[:, :i + 1], y, cv=KFold(n_folds)))
        scores.append(score)

    # Select the feature subset with the highest accuracy
    best_features = np.argmax(scores) + 1
    selected_features = feature_names[:best_features]
    print("Selected features using Lasso:", selected_features)

    # Train the SVM model on the selected feature subset
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=0)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Evaluate the model's performance on the test data
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy using SVM:", accuracy)

    RocAndAuc(X_train, X_test, y_train, y_test)


def ReproducingResult_LASSO_SVM(dataset,pred_col):
    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer

    # Load the data into a pandas DataFrame
    df = dataset

    # Extract the features and target
    X = df.drop(pred_col, axis=1)
    y = df[pred_col]
    # Initialize the imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Fit the imputer on the data
    imputer.fit(X)

    # Transform the data to fill in missing values
    X = imputer.transform(X)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # Initialize the Lasso model with a high value of alpha
    lasso = Lasso(alpha=0.001, max_iter=10000)

    # Fit the Lasso model on the training data
    lasso.fit(X_train, y_train)

    # Use the SelectFromModel object to select features with non-zero coefficients
    sfm = SelectFromModel(lasso, threshold=0.0001)
    sfm.fit(X_train, y_train)

    # Print the number of selected features
    n_features = sfm.transform(X_train).shape[1]
    print("Number of selected features: {}".format(n_features))

    # Transform the data to include only the selected features
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.svm import NuSVC
    # Initialize and fit the SVM classifier
    #clf = CallXGBoost(X_train, y_train)
    #clf = LinearSVC(max_iter=10000)
    clf = NuSVC()
    #clf = SVC(max_iter=10000)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Print the accuracy of the model
    acc = clf.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(acc * 100))

    from sklearn.model_selection import KFold, cross_val_score
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(clf, X, y, cv=k_folds)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))

    RocAndAuc(X_train, X_test, y_train, y_test)

