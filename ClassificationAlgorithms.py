from Constants import *
from InputAndOutputService import *
#from Statistics import *


def CallXGBoost(X_train, y_train):
    PrintAndWriteToLog("using XGboost")
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def CallNeuralNetwork(X_train, y_train):
    PrintAndWriteToLog("using sklearn.neural_network MLPClassifier")
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(100,2), random_state=1, solver='lbfgs')
    classifier.fit(X_train, y_train)
    return classifier

def CallSVM(X_train, y_train):
    PrintAndWriteToLog("using SVM")
    from sklearn.svm import SVC
    #classifier = SVC(kernel = 'linear',degree = 5, random_state = 42, max_iter=10000,probability=True)
    classifier = SVC(kernel = 'poly', degree = 3,probability=True)
    classifier.fit(X_train, y_train)
    return classifier

def CallLinearSVC(X_train, y_train):
    PrintAndWriteToLog("using LinearSVC")
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(max_iter=100000)
    #classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    return classifier

def CallKNN(X_train, y_train):
    PrintAndWriteToLog("using KNN")
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    return classifier

def CallLogisticRegression(X_train, y_train):
    PrintAndWriteToLog("using Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter= 10000, solver = 'lbfgs')
    #classifier = LogisticRegression(max_iter= 10000,penalty='l1', solver='liblinear', C=1.0) #LASSO
    classifier.fit(X_train, y_train)
    return classifier

def CallLogisticRegressionCV(X_train, y_train):
    PrintAndWriteToLog("using Logistic Regression CV")
    from sklearn.linear_model import LogisticRegressionCV
    classifier = LogisticRegressionCV()
    classifier.fit(X_train, y_train)
    return classifier
def CallPerceptron(X_train, y_train):
    PrintAndWriteToLog("using Perceptron")
    from sklearn.linear_model import Perceptron
    classifier = Perceptron(fit_intercept=False, max_iter=10, tol=None,shuffle = False)
    classifier.fit(X_train, y_train)
    return classifier

def CallCatBoost(X_train, y_train):
    PrintAndWriteToLog("using CatBoost")
    from catboost import CatBoostClassifier
    classifier = CatBoostClassifier(iterations=2,
                           depth=10,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
    classifier.fit(X_train, y_train)
    return classifier

def CallExtraTrees(X_train, y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    PrintAndWriteToLog("using ExtraTreesClassifier")
    classifier = ExtraTreesClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def FeatureScaling(X_train,X_test,DECIMAL_COLUMNS):
     #Feature Scaling - only the numeric values
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    sc = StandardScaler()
    PrintAndWriteToLog("FeatureScaling - StandardScaler",True)
    X_train[:, 1:DECIMAL_COLUMNS] = sc.fit_transform(X_train[:, 1:DECIMAL_COLUMNS])
    X_test[:, 1:DECIMAL_COLUMNS] = sc.transform(X_test[:, 1:DECIMAL_COLUMNS])
    # mms = MinMaxScaler()
    # PrintAndWriteToLog("FeatureScaling - MinMaxScaler",True)
    # X_train[:, 1:DECIMAL_COLUMNS] = mms.fit_transform(X_train[:, 1:DECIMAL_COLUMNS])
    # X_test[:, 1:DECIMAL_COLUMNS] = mms.transform(X_test[:, 1:DECIMAL_COLUMNS])

def PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # y[y<=5] = 0
    # y[y>0] = 1
    resDict['Notes'] = ""

    if (handle_missing_data):
        HandleMissingData(DECIMAL_COLUMNS, X)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # X_train, X_test = FeatureSelectionKBest(X_train, y_train,X_test, k=72)
    #X_train, X_test = FeatureSelectionPCA(X_train, X_test, 60)

    #X_test_best = X_test[:, best_features]
    #FeatureScaling(X_train, X_test, DECIMAL_COLUMNS)
    #X_train, X_test = LASSO(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test
def PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
                                       algoritem_name = XG_BOOST, handle_missing_data = False):
    PrintAndWriteToLog("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
    resDict['File Name'] = file_name
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data)
    if(algoritem_name == XG_BOOST):
        classifier = CallXGBoost(X_train, y_train)
    elif(algoritem_name == CAT_BOOST):
        classifier = CallCatBoost(X_train, y_train)
    elif(algoritem_name == KNN):
        classifier = CallKNN(X_train, y_train)
    elif(algoritem_name == SVM):
        classifier = CallSVM(X_train, y_train)
    elif(algoritem_name == LOGISTIC_REGRESSION):
        classifier = CallLogisticRegression(X_train, y_train)
        #classifier = CallLogisticRegressionCV(X_train, y_train)
    elif(algoritem_name == NEURAL_NETWORK):
        classifier = CallNeuralNetwork(X_train, y_train)
    elif(algoritem_name == EXTRA_TREES):
        classifier = CallExtraTrees(X_train, y_train)
    elif(algoritem_name == LINEAR_SVC):
        classifier = CallLinearSVC(X_train, y_train)
    elif(algoritem_name == PERCEPTRON):
        classifier = CallPerceptron(X_train, y_train)
    else:
        return 0

    PrintAndWriteToLog("Pred train result")
    y_pred_train = classifier.predict(X_train)
    GetAndPrintResult(y_train, y_pred_train)
    # Predicting the Test set results
    PrintAndWriteToLog("Pred test result")
    y_pred = classifier.predict(X_test)
    res = GetAndPrintResult(y_test, y_pred)
    #UsingShap(classifier, 'XG_Boost', X_test)
    RocAndAuc(X_train, X_test, y_train, y_test)
    return res

def PredictUsingAllCalssificationAlgoritems(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names):
    PrintAndWriteToLog("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
    resDict['File Name'] = file_name
    resDict['Date and time'] = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS, True)

    resList = []
    classifier = CallXGBoost(X_train, y_train)
    resDict['XGBoost'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallCatBoost(X_train, y_train)
    resDict['CatBoost'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallKNN(X_train, y_train)
    resDict['KNN'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallNeuralNetwork(X_train, y_train)
    resDict['MLPClassifier'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallExtraTrees(X_train, y_train)
    resDict['ExtraTreesClassifier'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallPerceptron(X_train, y_train)
    resDict['Perceptron'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallSVM(X_train, y_train)
    resDict['SVM'] = GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    # classifier = CallLinearSVC(X_train, y_train)
    # GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    classifier = CallLogisticRegression(X_train, y_train)
    resDict['LogisticRegression'] = GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    max_res = max(resList)
    resDict['Best result'] = max_res
    SaveResultToCSV(resDict)
    PrintAndWriteToLog("The best result for file name: " + file_name + " is: "+ str(max_res))
    RocAndAuc(X_train, X_test, y_train, y_test)

def GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train, KFoldCrossVal = True):
    #classifier = FeatureSelectionRFE(classifier, 35, X_train, y_train, X_test,y_test)
    # if (KFoldCrossVal):
    #     train_best_accuracy, test_accuracy = GetTheBestScoreUsingKFoldCrossValidation(classifier, X_train, y_train,
    #                                                                                  X_test, y_test)
    #   res = test_accuracy
    #else:
        # PrintAndWriteToLog("Pred train result")
        # y_pred_train = classifier.predict(X_train)
        # GetAndPrintResult(y_train, y_pred_train)
        # Predicting the Test set results
    PrintAndWriteToLog("Pred test result")
    y_pred = classifier.predict(X_test)
    res = GetAndPrintResult(y_test, y_pred)
    resList.append(res)
    return res


def GetTheBestScoreUsingKFoldCrossValidation(classifier, X_train, y_train, X_test, y_test):
    best_features, train_best_accuracy = feature_selection_with_cross_validation(X_train, y_train, classifier, num_folds=10,
                                                                           random_state=42)
    PrintAndWriteToLog("GetTheBestScoreUsingKFoldCrossValidation")
    # Select the best features on the training data
    X_train_best = X_train[:, best_features]
    classifier.fit(X_train_best, y_train)
    # Evaluate the classifier on the test data
    X_test_best = X_test[:, best_features]
    test_accuracy = classifier.score(X_test_best, y_test)
    print("Best feature subset:", best_features)
    print("Training accuracy with best features:", train_best_accuracy)
    print("Test accuracy with best features:", test_accuracy)
    return train_best_accuracy, test_accuracy

def FeatureSelectionRFE(classifier, k,X_train,y_train, X_test,y_test):
    PrintAndWriteToLog(f"FeatureSelection - RFE, n_features_to_select={k}",True)
    from sklearn.feature_selection import RFE
    selector = RFE(classifier, n_features_to_select=k, step=1)
    selector.fit_transform(X_train, y_train)
    selector.transform(X_test)
    return classifier
def FeatureSelectionKBest(X_train,y_train,X_test, k):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    PrintAndWriteToLog(f"Feature Selection - SelectKBest(chi2, k={k})",True)
    kbest = SelectKBest(chi2, k=k)
    # X_train_new = X_train_new.fit(X_train)
    X_train_new = kbest.fit_transform(X_train,y_train)
    X_test_new = kbest.transform(X_test)
    X_train_new.shape
    X_test_new.shape
    return X_train_new, X_test_new


def FeatureSelectionPCA(X_train,X_test,n):
    PrintAndWriteToLog(f"FeatureSelection - PCA, n_components = {n}",True)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n, svd_solver='full')
    #print(pca.explained_variance_ratio_)
    X_train_new = pca.fit_transform(X_train)
    X_test_new = pca.transform(X_test)
    return X_train_new, X_test_new

def HandleMissingData(DECIMAL_COLUMNS, X):
       #Hendle missing data
    #change to avg value in the numeric columns (like age, height, weight...)
    # i can change it to median
    from sklearn.impute import SimpleImputer
    PrintAndWriteToLog("HandleMissingData - strategy='mean'",True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(X[:, 1:DECIMAL_COLUMNS])
    X[:, 1:DECIMAL_COLUMNS] = imputer.transform(X[:, 1:DECIMAL_COLUMNS])
    #where_are_NaNs = np.isnan(X)
    # X[where_are_NaNs] = -1  #??


#TODO: need to be in Statistics class
def FeatureImportance(classifier):
    PrintAndWriteToLog("feature importance:")
    from matplotlib import pyplot
    PrintAndWriteToLog(classifier.feature_importances_)
    pyplot.bar(range(len(classifier.feature_importances_)), classifier.feature_importances_)
    pyplot.show()

def UsingShap(classifier, classifier_name, X_test):
    import shap
    explainer = shap.TreeExplainer(classifier)
    # Calculate shapley values for test data
    start_index = 1
    end_index = 2
    shap_values = explainer.shap_values(X_test[start_index:end_index])
    X_test[start_index:end_index]

    # %% Investigating the values (classification problem)
    # class 0 = contribution to class 1
    # class 1 = contribution to class 2
    print(shap_values[0].shape)
    shap_values

    # %% >> Visualize local predictions
    shap.initjs()
    # Force plot
    prediction = classifier.predict(X_test[start_index:end_index])[0]
    PrintAndWriteToLog(f"The {classifier_name} predicted: {prediction}")
    shap.force_plot(explainer.expected_value[1],
                    shap_values[1],
                    X_test[start_index:end_index],
                    feature_names=list_feature_names)  # for values

    # %% >> Visualize global features
    # Feature summary
    shap.summary_plot(shap_values, X_test, feature_names=list_feature_names, plot_type="bar", show=False)
    plt.title(file_name)
    plt.show()

    shap.decision_plot(explainer.expected_value[0], shap_values[0], x_test)
    shap.force_plot(explainer.expected_value[0], shap_values[0], x_test.iloc[0])
def RocAndAucTest(dataset, DECIMAL_COLUMNS, file_name):
    print(file_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    FeatureScaling(X_train, X_test, DECIMAL_COLUMNS)
    RocAndAuc(X_train, X_test, y_train, y_test)

def RocAndAuc(X_train, X_test, y_train, y_test):
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    #classLogisticRegression = CallLogisticRegression(X_train, y_train)
    classKNN = CallKNN(X_train, y_train)
    classSVM = CallSVM(X_train, y_train)

    #y_scoreLogisticRegression = classLogisticRegression.predict_proba(X_test)[:,1]
    y_scoreKNN = classKNN.predict_proba(X_test)[:,1]
    y_scoreSVM = classSVM.predict_proba(X_test)[:, 1]

    #false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_scoreLogisticRegression)
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_scoreKNN)
    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_scoreSVM)

    #print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_scoreLogisticRegression))
    print('roc_auc_score for y_scoreKNN: ', roc_auc_score(y_test, y_scoreKNN))
    print('roc_auc_score for y_scoreSVM: ', roc_auc_score(y_test, y_scoreSVM))

    plt.subplots(1, figsize=(10,10))
    #plt.title('Receiver Operating Characteristic - LogisticRegression')
    plt.title('Receiver Operating Characteristic - KNN')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - SVM')
    plt.plot(false_positive_rate2, true_positive_rate2)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



import csv
from typing import Dict, List
from sklearn.linear_model import LinearRegression

def call_predict_by_feature(dataset,feature_names,output_file):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    results = predict_by_feature(X, y, feature_names)

    # write predicted values to CSV file
    write_results_to_csv(results, output_file)

def write_results_to_csv(results: Dict[str, List[float]], output_file: str):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # write header row
        writer.writerow(['Feature', 'Predicted Values'])

        # write predicted values for each feature
        for feature_name, predictions in results.items():
            writer.writerow([feature_name, predictions])

from typing import Dict, List
from sklearn.linear_model import LinearRegression

def predict_by_feature(X, y, feature_names: List[str]) -> Dict[str, List[float]]:
    # assuming X is your feature matrix and y is your target variable
    n_features = X.shape[1]
    results = {}

    for i in range(n_features):
        # select ith feature
        X_i = X[:, i].reshape(-1, 1)

        # fit a linear regression model
        model = LinearRegression()
        model.fit(X_i, y)

        # predict y using the ith feature
        y_pred = model.predict(X_i)

        # add predicted values to dictionary
        feature_name = feature_names[i]
        results[feature_name] = y_pred.tolist()

    return results

def LASSO(X_train, X_test, y_train, y_test):
    PrintAndWriteToLog("using LASSO")
    from sklearn.linear_model import LassoCV
    from sklearn.feature_selection import SelectFromModel

    # Create a Lasso model with cross-validated alpha selection
    lasso = LassoCV(max_iter=10000, alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)

    # Fit the Lasso model to your data for feature selection
    lasso.fit(X_train, y_train)  # X_train is your training data, y_train is the target labels

    # Use SelectFromModel to select features based on Lasso coefficients
    sfm = SelectFromModel(lasso, prefit=True)

    # Transform your data to keep only the selected features
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)  # If you have a test set

    return X_train_selected,X_test_selected


def feature_selection_with_cross_validation(X, y, classifier, num_folds=10, random_state=42):
    """
    Perform feature selection using k-fold cross-validation and return the best feature subset and its accuracy.

    Parameters:
    - X: Features (feature matrix).
    - y: Target labels.
    - classifier: The classifier used for evaluation (default: Support Vector Classifier with linear kernel).
    - num_folds: Number of folds for cross-validation (default: 10).
    - random_state: Random seed for reproducibility (default: 42).

    Returns:
    - best_features: The best feature subset.
    - best_accuracy: The accuracy of the best feature subset.
    """
    from sklearn.model_selection import cross_val_score, KFold
    from itertools import combinations

    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    feature_accuracies = []

    for num_features in range(1, X.shape[1] + 1):
        feature_combinations = list(combinations(range(X.shape[1]), num_features))

        subset_accuracies = []

        for features in feature_combinations:
            X_subset = X[:, features]
            accuracy_scores = cross_val_score(classifier, X_subset, y, cv=k_fold, scoring='accuracy')
            mean_accuracy = np.mean(accuracy_scores)
            subset_accuracies.append((features, mean_accuracy))

        best_subset = max(subset_accuracies, key=lambda x: x[1])
        feature_accuracies.append(best_subset)

    best_features, best_accuracy = max(feature_accuracies, key=lambda x: x[1])
    return best_features, best_accuracy

#
# # Usage example:
# # Replace X and y with your own data
# X = ...  # Your feature matrix
# y = ...  # Your target labels
# best_features, best_accuracy = feature_selection_with_cross_validation(X, y)
# print("Best feature subset:", best_features)
# print("Best accuracy:", best_accuracy)
