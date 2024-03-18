from Constants import *
from InputAndOutputService import *
#from Statistics import *

def CallXGBoost(X_train, y_train):
    PrintAndWriteToLog("using XGboost")
    from xgboost import XGBClassifier
    classifier = XGBClassifier(
        random_state = 42,tree_method = "approx", booster = "dart", alpha = 1,eta = 0.1, gamma = 1)
    classifier.fit(X_train, y_train)
    return classifier

def CallNeuralNetwork(X_train, y_train):
    PrintAndWriteToLog("using sklearn.neural_network MLPClassifier")
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100,2), random_state=42, solver='sgd', max_iter=10000)
    classifier.fit(X_train, y_train)
    return classifier

def CallSVM(X_train, y_train):
    PrintAndWriteToLog("using SVM")
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'poly', degree = 3, probability=True, C=30)
    #classifier = SVC(kernel='rbf',C=20, gamma = 'auto', probability=True)
    #classifier = SVC(kernel='sigmoid', C=20, gamma = 'auto', probability=True)
    # classifier = SVC(kernel = 'linear',C=10, random_state = 42, max_iter=10000,probability=True)
    classifier.fit(X_train, y_train)
    return classifier

def CallLinearSVC(X_train, y_train):
    PrintAndWriteToLog("using LinearSVC")
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(max_iter=100000)
    classifier.fit(X_train, y_train)
    return classifier


def CallRBFSVC(X_train, y_train):
    PrintAndWriteToLog("using RBF SVC")
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',C=50, gamma = 'scale', probability=True)
    #classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    return classifier

def CallKNN(X_train, y_train):
    k_neighbors = 2
    PrintAndWriteToLog(f"using KNN, k={k_neighbors}")
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
    classifier.fit(X_train, y_train)
    return classifier

def CallLogisticRegression(X_train, y_train):
    PrintAndWriteToLog("using Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter= 10000, solver = 'sag', C=0.1)
    #classifier = LogisticRegression(max_iter= 10000, penalty = 'l2', solver='saga', C=0.1)
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
                           depth=2,
                           learning_rate=0.1,
                           loss_function='Logloss',
                           verbose=True)
    classifier.fit(X_train, y_train)
    return classifier

def CallDecisionTree(X_train, y_train):
    PrintAndWriteToLog("using Decision Tree")
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def CallExtraTrees(X_train, y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    PrintAndWriteToLog("using ExtraTreesClassifier")
    classifier = ExtraTreesClassifier(n_estimators = 100,criterion = "gini")
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

def PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data,feature_names):
    X = dataset.iloc[:, :-1].values
    # X = dataset[['Bacteroides_uniformis_1','Bacteroides_uniformis_2','Bacteroides_uniformis_3']]
    # X = dataset[['Bacteroides_uniformis_1','Bacteroides_uniformis_2','Bacteroides_uniformis_3']]
    # X = dataset[['Prevotella_copri_1','Prevotella_12', 'Bacteroides_uniformis_1']]
    #                  'Bacteroides_dorei_1','Bacteroides_MS_3', 'Bacteroides_vulgatus_2','Parabacteroides_merdae_3',
    #                  'Alloprevotella_1','Alistipes_finegoldii_2','Bacteroides_3','Prevotella_4','Parabacteroides_merdae_1',
    #              'Prevotella_5','Firmicutes_MG_4', 'Ruminococcaceae_MG_1']].values
    # the 6 OTUs < 0.05 contained within the 15 best OTUs
    # X = dataset[['Prevotella_copri_1', 'Prevotella_12', 'Bacteroides_uniformis_1',
    #                  'Bacteroides_dorei_1','Bacteroides_MS_3', 'Bacteroides_vulgatus_2']]
    # the 5 OTUs < 0.05 contained within the 12 best OTUs
    #X = dataset[['Prevotella_copri_1','Prevotella_12','Bacteroides_uniformis_1','Parabacteroides_merdae_3','Akkermansia_muciniphila_1']]
    # X = dataset[['Bacteroides_dorei_1','Bacteroides_MS_3','Bacteroides_vulgatus_2','Alloprevotella_1',
    #              'Alistipes_finegoldii_2','Bacteroides_3','Prevotella_4','Prevotella_5','Ruminococcaceae_MG_1']]
    y = dataset.iloc[:, -1].values

    # y[y<=5] = 0
    # y[y>0] = 1
    resDict['Notes'] = ""

    # if (handle_missing_data):
    #     HandleMissingData(DECIMAL_COLUMNS, X)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    k=40
    X_train, X_test = FeatureSelectionKBest(X_train, y_train, X_test, k,feature_names)
    # X_train, X_test = FeatureSelectionKBest(X_train, y_train,X_test, k=72)
    # X_train, X_test = FeatureSelectionPCA(X_train, X_test, 86)

    #X_test_best = X_test[:, best_features]
    # FeatureScaling(X_train, X_test, DECIMAL_COLUMNS)
    #X_train, X_test = LASSO(X_train, X_test, y_train, y_test)
    # plot_tsne(X_train, y_train)
    # plot_PCA(X_train, y_train)
    return X_train, X_test, y_train, y_test
def PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
                                       algoritem_name = XG_BOOST, handle_missing_data = False):
    PrintAndWriteToLog("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
    resDict['File Name'] = file_name
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data,list_feature_names)
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

    #train_best_accuracy, test_accuracy = GetTheBestScoreUsingKFoldCrossValidation(classifier, X_train, y_train,
    #                                                                             X_test, y_test)
    #res = test_accuracy
    # PrintAndWriteToLog("Pred train result")
    # y_pred_train = classifier.predict(X_train)
    #
    # GetAndPrintResult(y_train, y_pred_train)
    # # Predicting the Test set results
    # PrintAndWriteToLog("Pred test result")
    y_pred = classifier.predict(X_test)
    res = GetAndPrintResult(y_test, y_pred)
    #UsingShap(classifier, 'XG_Boost', X_test)
    RocAndAuc(X_train, X_test, y_train, y_test)
    return res

def PredictUsingAllCalssificationAlgoritems(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names):
    PrintAndWriteToLog("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
    resDict['File Name'] = file_name
    resDict['Date and time'] = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS, True,list_feature_names)

    resList = []
    classifier = CallExtraTrees(X_train, y_train)
    resDict['ExtraTreesClassifier'] = GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    classifier = CallXGBoost(X_train, y_train)
    resDict['XGBoost'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallCatBoost(X_train, y_train)
    resDict['CatBoost'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    # classifier = CallKNN(X_train, y_train)
    # resDict['KNN'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    resDict['KNN'] = call_knn_with_treshold(X_train,X_test, y_train,y_test,resList,0.8)
    # classifier = CallNeuralNetwork(X_train, y_train)
    # resDict['MLPClassifier'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    # classifier = CallPerceptron(X_train, y_train)
    # resDict['Perceptron'] = GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallDecisionTree(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    classifier = CallSVM(X_train, y_train)
    resDict['SVM'] = GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    classifier = CallRBFSVC(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    call_SVM_RBF_with_treshold(X_train, X_test, y_train, y_test, resList, threshold=0.59368)
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
    #     res = test_accuracy
    # else:
        # PrintAndWriteToLog("Pred train result")
        # y_pred_train = classifier.predict(X_train)
        # GetAndPrintResult(y_train, y_pred_train)
        # Predicting the Test set results
    # PrintAndWriteToLog("Pred test result")
    y_pred = classifier.predict(X_test)
    res = GetAndPrintResult(y_test, y_pred)
    # sensitivity_and_specificity(y_test, y_pred)
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
def FeatureSelectionKBest(X_train,y_train,X_test, k,feature_names):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
    PrintAndWriteToLog(f"Feature Selection - SelectKBest(chi2, k={k})",True)
    kbest = SelectKBest(chi2, k=k)
    X_train_new = kbest.fit_transform(X_train,y_train)
    X_test_new = kbest.transform(X_test)
    X_train_new.shape
    X_test_new.shape
    print_k_best_features(kbest,k,feature_names)
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

    classKNN = CallKNN(X_train, y_train)
    classSVM = CallRBFSVC(X_train, y_train)
    #classSVM = CallSVM(X_train, y_train)
    classLogisticRegression = CallLogisticRegression(X_train, y_train)
    classExtraTrees = CallExtraTrees(X_train, y_train)

    y_scoreKNN = classKNN.predict_proba(X_test)[:,1]
    y_scoreSVM = classSVM.predict_proba(X_test)[:, 1]
    y_scoreLogisticRegression = classLogisticRegression.predict_proba(X_test)[:,1]
    y_scoreExtraTrees = classExtraTrees.predict_proba(X_test)[:, 1]

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_scoreKNN)
    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_scoreSVM)
    false_positive_rate3, true_positive_rate3, threshold3 = roc_curve(y_test, y_scoreLogisticRegression)
    false_positive_rate4, true_positive_rate4, threshold4 = roc_curve(y_test, y_scoreExtraTrees)

    print('roc_auc_score for y_scoreKNN: ', roc_auc_score(y_test, y_scoreKNN))
    print('roc_auc_score for y_scoreSVM: ', roc_auc_score(y_test, y_scoreSVM))
    print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_scoreLogisticRegression))
    print('roc_auc_score for ExtraTrees: ', roc_auc_score(y_test, y_scoreExtraTrees))

    # plt.subplots(1, figsize=(10,10))
    # #plt.title('Receiver Operating Characteristic - LogisticRegression')
    # plt.title('Receiver Operating Characteristic - KNN')
    # plt.plot(false_positive_rate1, true_positive_rate1)
    # plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
    #
    # plt.subplots(1, figsize=(10,10))
    # plt.title('Receiver Operating Characteristic - SVM')
    # plt.plot(false_positive_rate2, true_positive_rate2)
    # plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(false_positive_rate1, true_positive_rate1, label='Model KNN (AUC = {:.2f})'.format(roc_auc_score(y_test, y_scoreKNN)))
    plt.plot(false_positive_rate2, true_positive_rate2, label='Model SVM(RBF) (AUC = {:.2f})'.format(roc_auc_score(y_test, y_scoreSVM)))
    plt.plot(false_positive_rate3, true_positive_rate3, label='Model Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_scoreLogisticRegression)))
    plt.plot(false_positive_rate4, true_positive_rate4, label='Model Extra Trees (AUC = {:.2f})'.format(roc_auc_score(y_test, y_scoreExtraTrees)))

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC curve - Bacteroides_uniformis_1 and Bacteroides_uniformis_3')
    plt.legend()
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

def sensitivity_and_specificity(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Extract values from the confusion matrix
    true_negatives = conf_matrix[0, 0]
    false_negatives = conf_matrix[1, 0]
    true_positives = conf_matrix[1, 1]
    false_positives = conf_matrix[0, 1]

    # Calculate sensitivity (True Positive Rate)
    sensitivity = true_positives / (true_positives + false_negatives)
    print("Sensitivity (True Positive Rate): {:.2f}".format(sensitivity))

    # Calculate specificity (True Negative Rate)
    specificity = true_negatives / (true_negatives + false_positives)
    print("Specificity (True Negative Rate): {:.2f}".format(specificity))

def use_hyperparameter_search_with_LogisticRegression(dataset, DECIMAL_COLUMNS,handle_missing_data,list_feature_names):
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data,list_feature_names)
    y_pred = hyperparameter_search_with_LogisticRegression(X_train, X_test, y_train, y_test)
    res = GetAndPrintResult(y_test, y_pred)
    sensitivity_and_specificity(y_test, y_pred)
def use_hyperparameter_search_with_SVM(dataset, DECIMAL_COLUMNS,handle_missing_data,list_feature_names):
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data,list_feature_names)
    y_pred = hyperparameter_search_with_SVM(X_train, X_test, y_train, y_test)
    res = GetAndPrintResult(y_test, y_pred)
    sensitivity_and_specificity(y_test, y_pred)
def hyperparameter_search_with_SVM(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    # Define the SVM model
    svm_model = SVC()

    # Define the hyperparameter grid to search
    param_grid = {
        'C': [0.1, 1, 10,20,30,100],  # Regularization parameter
        'kernel': ['rbf','sigmoid'],  # Kernel type
        'gamma': ['scale', 'auto'],  # Kernel coefficient
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the model to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Predict on the test set using the best model
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return y_pred

def hyperparameter_search_with_LogisticRegression(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Define the SVM model
    svm_model = LogisticRegression()

    # Define the hyperparameter grid to search
    param_grid = {
        'C': [0.5, 1, 10,20,30,100],  # Regularization parameter
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the model to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Predict on the test set using the best model
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return y_pred
def call_knn_with_treshold(X_train,X_test, y_train,y_test,resList,threshold = 0.5):
    knn = CallKNN(X_train, y_train)
    probs = knn.predict_proba(X_test)
    probs_train = knn.predict_proba(X_train)
    # Set a custom threshold (e.g., 0.5) for binary classification
    y_pred = (probs[:, 1] >= threshold).astype(int)
    y_pred_train = (probs_train[:, 1] >= threshold).astype(int)

    # plot_tsne(X_train, y_train)

    print(f"threshold={threshold}")
    # print("Train")
    # GetAndPrintResult(y_train, y_pred_train)
    # print("Test")
    res = GetAndPrintResult(y_test, y_pred)
    resList.append(res)

    return res

def call_SVM_RBF_with_treshold(X_train,X_test, y_train,y_test,resList,threshold):
    print("call_SVM_RBF_with_treshold")
    svm_rbf = CallRBFSVC(X_train, y_train)
    probs = svm_rbf.predict_proba(X_test)
    probs_train = svm_rbf.predict_proba(X_train)
    # Set a custom threshold (e.g., 0.5) for binary classification
    y_pred = (probs[:, 1] >= threshold).astype(int)
    y_pred_train = (probs_train[:, 1] >= threshold).astype(int)

    # plot_tsne(X_train, y_train)

    print(f"threshold={threshold}")
    # print("Train")
    # GetAndPrintResult(y_train, y_pred_train)
    # print("Test")
    res = GetAndPrintResult(y_test, y_pred)
    resList.append(res)

    return res

def predict_with_knn_with_treshold(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names, threshold = 0.5):
    PrintAndWriteToLog("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
    resDict['File Name'] = file_name
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS, True,
                                                                  list_feature_names)
    knn = CallKNN(X_train, y_train)

    # Make predictions using predict_proba
    probs = knn.predict_proba(X_test)
    probs_train = knn.predict_proba(X_train)

    # Set a custom threshold (e.g., 0.5) for binary classification
    y_pred = (probs[:, 1] >= threshold).astype(int)
    y_pred_train = (probs_train[:, 1] >= threshold).astype(int)

    plot_tsne(X_train, y_train)

    print(f"threshold={threshold}")
    print("Train")
    GetAndPrintResult(y_train, y_pred_train)
    print("Test")
    res = GetAndPrintResult(y_test, y_pred)

    return res

def KNN_with_leaveoneout(dataset,feature_names):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score
    import numpy as np

    resDict['Notes'] = ""
    # Create your K-nearest neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=2)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # k = 15
    # X_train, X_test = FeatureSelectionKBest(X_train, y_train, X_test, k, feature_names)
    # #
    # Initialize LeaveOneOut cross-validation
    loo = LeaveOneOut()

    accuracy_scores = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        k = 15
        X_train, X_test = FeatureSelectionKBest(X_train, y_train, X_test, k, feature_names)

        knn.fit(X_train, y_train)

        # y_pred = knn.predict(X_test)
        probs = knn.predict_proba(X_test)
        y_pred = (probs[:, 1] >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    # Calculate the mean accuracy over all folds
    mean_accuracy = np.mean(accuracy_scores)

    print(f'Mean Accuracy with LOOCV: {mean_accuracy:.2f}')

def KNN_Kfoldcrossvalidation(dataset,feature_names):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    import numpy as np

    # Create your K-nearest neighbors classifier
    # knn = KNeighborsClassifier(n_neighbors=2)
    from sklearn.svm import SVC
    knn = SVC(kernel='rbf', C=20, gamma='scale', probability=True)

    # Assuming X is your feature matrix and y is your target vector
    # X = dataset[['Prevotella_copri_1', 'Prevotella_12', 'Bacteroides_uniformis_1',
    #                  'Bacteroides_dorei_1','Bacteroides_MS_3', 'Bacteroides_vulgatus_2','Parabacteroides_merdae_3',
    #                  'Alloprevotella_1','Alistipes_finegoldii_2','Bacteroides_3','Prevotella_4','Parabacteroides_merdae_1',
    #              'Prevotella_5','Firmicutes_MG_4', 'Ruminococcaceae_MG_1']].values
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Initialize 10-fold cross-validation
    kf = KFold(n_splits=12)

    accuracy_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # k = 15
        # X_train, X_test = FeatureSelectionKBest(X_train, y_train, X_test, k, feature_names)

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # probs = knn.predict_proba(X_test)
        # y_pred = (probs[:, 1] >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    # Calculate the mean accuracy over all folds
    mean_accuracy = np.mean(accuracy_scores)

    print(f'Mean Accuracy with 10-Fold Cross-Validation: {mean_accuracy:.2f}')

def plot_tsne(X,y):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Apply t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2,  random_state=42, init = "pca", learning_rate = 1000.0)
    X_tsne = tsne.fit_transform(X)

    # Separate the data points for each class
    class_0_indices = (y == 0)
    class_1_indices = (y == 1)

    # Create a scatter plot of the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[class_0_indices, 0], X_tsne[class_0_indices, 1], label="Class 0", c='blue')
    plt.scatter(X_tsne[class_1_indices, 0], X_tsne[class_1_indices, 1], label="Class 1", c='red')
    plt.title("t-SNE Plot of the Dataset,  init = pca, learning_rate = 1000.00")
    plt.legend()
    plt.show()

def plot_PCA(X, y):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Perform PCA to reduce the dimensionality to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a scatter plot of the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0][:, 0], X_pca[y == 0][:, 1], label="Class 0", c='blue')
    plt.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], label="Class 1", c='red')
    plt.title("PCA Projection Plot of the Dataset")
    plt.legend()
    plt.show()

def plot_everything(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    from sklearn.datasets import make_circles, make_classification, make_moons
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    # X, y = make_classification(
    #     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    # )
    # rng = np.random.RandomState(2)
    # X += 2 * rng.uniform(size=X.shape)
    # linearly_separable = (X, y)

    datasets = [
        make_moons(noise=0.3, random_state=0),
        # make_circles(noise=0.2, factor=0.5, random_state=1),
        # linearly_separable,
    ]

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        X = dataset[['Prevotella_copri_1', 'Prevotella_12', 'Bacteroides_uniformis_1',
                         'Bacteroides_dorei_1','Bacteroides_MS_3', 'Bacteroides_vulgatus_2','Parabacteroides_merdae_3',
                         'Alloprevotella_1','Alistipes_finegoldii_2','Bacteroides_3','Prevotella_4','Parabacteroides_merdae_1',
                     'Prevotella_5','Firmicutes_MG_4', 'Ruminococcaceae_MG_1']].values
        y = dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            clf = CallSVM(X_train,y_train)
            # clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    plt.tight_layout()
    plt.show()


def plot_dataset_by_classifier(dataset,list_feature_names):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, 15, False, list_feature_names)

    # Train a KNN classifier with k=2
    knn_classifier = KNeighborsClassifier(n_neighbors=2)
    knn_classifier.fit(X_train, y_train)

    # Create a meshgrid to plot the decision boundary using the first two features
    feature1_index = 0
    feature2_index = 1

    # Set the plot boundaries based on the dataset
    x_min, x_max = X[:, feature1_index].min() - 1, X[:, feature1_index].max() + 1
    y_min, y_max = X[:, feature2_index].min() - 1, X[:, feature2_index].max() + 1

    # Generate points for the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Make predictions on the meshgrid
    Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Apply the threshold of 0.5 for visualization
    above_threshold_train = knn_classifier.predict_proba(X_train)[:, 1] >= 0.5
    below_threshold_train = ~above_threshold_train

    above_threshold_test = knn_classifier.predict_proba(X_test)[:, 1] >= 0.5
    below_threshold_test = ~above_threshold_test

    # Plot the decision boundary
    plt.figure(figsize=(12, 8))

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)

    # Plot points for the true labels in the training set
    plt.scatter(X_train[y_train == 1, feature1_index], X_train[y_train == 1, feature2_index], c='blue', edgecolors='k',
                s=80, label='Train - Label 1')
    plt.scatter(X_train[y_train == 0, feature1_index], X_train[y_train == 0, feature2_index], c='green', edgecolors='k',
                s=80, label='Train - Label 0')

    # Plot points above the threshold in the training set
    plt.scatter(X_train[above_threshold_train, feature1_index], X_train[above_threshold_train, feature2_index], c='red',
                marker='x', s=80, label='Train - Above Threshold')

    # Plot points below the threshold in the training set
    plt.scatter(X_train[below_threshold_train, feature1_index], X_train[below_threshold_train, feature2_index],
                c='orange', marker='x', s=80, label='Train - Below Threshold')

    # Plot points for the true labels in the test set
    plt.scatter(X_test[y_test == 1, feature1_index], X_test[y_test == 1, feature2_index], c='cyan', edgecolors='k',
                s=80, label='Test - Label 1')
    plt.scatter(X_test[y_test == 0, feature1_index], X_test[y_test == 0, feature2_index], c='magenta', edgecolors='k',
                s=80, label='Test - Label 0')

    # Plot points above the threshold in the test set
    plt.scatter(X_test[above_threshold_test, feature1_index], X_test[above_threshold_test, feature2_index], c='purple',
                marker='x', s=80, label='Test - Above Threshold')

    # Plot points below the threshold in the test set
    plt.scatter(X_test[below_threshold_test, feature1_index], X_test[below_threshold_test, feature2_index], c='yellow',
                marker='x', s=80, label='Test - Below Threshold')

    plt.title('Classification Results with Decision Boundary (KNN)')
    plt.xlabel(f'Feature {feature1_index + 1}')
    plt.ylabel(f'Feature {feature2_index + 1}')
    plt.legend()
    plt.show()

    # X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, 15, False, list_feature_names)

    # # Apply PCA for dimensionality reduction to 2D
    # # Apply PCA for dimensionality reduction to 2D
    # pca = PCA(n_components=2)
    # X_test_pca = pca.fit_transform(X_test)
    #
    # # Train a classifier (KNN for example) with a threshold of 0.5
    # # knn_classifier = KNeighborsClassifier(n_neighbors=2)
    # # knn_classifier.fit(X_train, y_train)
    # knn = CallKNN(X_train, y_train)
    # probs = knn.predict_proba(X_test)
    #
    # # Create a meshgrid to plot the decision boundary
    # h = 100  # step size in the mesh
    # x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    # y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # y_pred = (probs[:, 1] >= 0.5).astype(int)
    #
    # # Make predictions on the meshgrid
    # Z = knn.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    # Z = Z.reshape(xx.shape)
    #
    # # Apply the threshold of 0.5
    # Z = Z >= 0.5
    #
    # # Plot the test set and the decision boundary
    # plt.figure(figsize=(8, 6))
    #
    # # Plot decision boundary
    # plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
    #
    # # Plot points for the true labels in the test set
    # plt.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], c='red', edgecolors='k', s=80,
    #             label='True Positives (Test)')
    # plt.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], c='blue', edgecolors='k', s=80,
    #             label='True Negatives (Test)')
    #
    # # Plot points for the false labels in the test set
    # incorrectly_classified = y_test != y_pred
    # # plt.scatter(X_test_pca[incorrectly_classified, 0], X_test_pca[incorrectly_classified, 1], c='black', marker='x', s=80,
    # #             label='False Labels (Test)')
    #
    # plt.title('Classification Results with Decision Boundary (KNN) - Test Set Only')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.legend()
    # plt.show()
    #
    # GetAndPrintResult(y_test, y_pred)

