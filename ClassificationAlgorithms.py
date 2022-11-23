from Constants import *
from InputAndOutputService import *
#from Statistics import *


def CallXGBoost(X_train, y_train):
    print("using XGboost")
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def CallNeuralNetwork(X_train, y_train):
    print("using sklearn.neural_network MLPClassifier")
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(100,2), random_state=1, solver='lbfgs')
    classifier.fit(X_train, y_train)
    return classifier

def CallSVM(X_train, y_train):
    print("using SVM")
    from sklearn.svm import SVC
    #classifier = SVC(kernel = 'linear',degree = 5, random_state = 10, max_iter=10)
    classifier = SVC(kernel = 'poly', degree = 3)
    classifier.fit(X_train, y_train)
    return classifier

def CallLinearSVC(X_train, y_train):
    print("using LinearSVC")
    from sklearn.svm import LinearSVC
    #classifier = LinearSVC(max_iter=100000)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    return classifier

def CallKNN(X_train, y_train):
    print("using KNN")
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    return classifier

def CallLogisticRegression(X_train, y_train):
    print("using Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(solver = 'lbfgs')
    classifier.fit(X_train, y_train)
    return classifier

def CallLogisticRegressionCV(X_train, y_train):
    print("using Logistic Regression CV")
    from sklearn.linear_model import LogisticRegressionCV
    classifier = LogisticRegressionCV()
    classifier.fit(X_train, y_train)
    return classifier
def CallPerceptron(X_train, y_train):
    print("using Perceptron")
    from sklearn.linear_model import Perceptron
    classifier = Perceptron(fit_intercept=False, max_iter=10, tol=None,shuffle = False)
    classifier.fit(X_train, y_train)
    return classifier

def CallCatBoost(X_train, y_train):
    print("using CatBoost")
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
    print("using ExtraTreesClassifier")
    classifier = ExtraTreesClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def FeatureScaling(X_train,X_test,DECIMAL_COLUMNS):
     #Feature Scaling - only the numeric values
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    sc = StandardScaler()
    print("FeatureScaling - StandardScaler")
    X_train[:, 1:DECIMAL_COLUMNS] = sc.fit_transform(X_train[:, 1:DECIMAL_COLUMNS])
    X_test[:, 1:DECIMAL_COLUMNS] = sc.transform(X_test[:, 1:DECIMAL_COLUMNS])
    mms = MinMaxScaler()
    print("FeatureScaling - MinMaxScaler")
    X_train[:, 1:DECIMAL_COLUMNS] = mms.fit_transform(X_train[:, 1:DECIMAL_COLUMNS])
    X_test[:, 1:DECIMAL_COLUMNS] = mms.transform(X_test[:, 1:DECIMAL_COLUMNS])

def PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS,handle_missing_data):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X = FeatureSelectionKBest(X,y,k=15)
    #X = FeatureSelectionPCA(X)

    if (handle_missing_data):
        HandleMissingData(DECIMAL_COLUMNS, X)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    FeatureScaling(X_train, X_test, DECIMAL_COLUMNS)
    return X_train, X_test, y_train, y_test
def PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
                                       algoritem_name = XG_BOOST, handle_missing_data = False):
    print("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
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

    print("Pred train result")
    y_pred_train = classifier.predict(X_train)
    GetAndPrintResult(y_train, y_pred_train)
    # Predicting the Test set results
    print("Pred test result")
    y_pred = classifier.predict(X_test)
    res = GetAndPrintResult(y_test, y_pred)
    #UsingShap(classifier, 'XG_Boost', X_test)
    return res

def PredictUsingAllCalssificationAlgoritems(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names):
    print("File name is: '" + file_name + "', Sheet name is: '" + sheet_name + "'")
    X_train, X_test, y_train, y_test = PreperDataBeforePrediction(dataset, DECIMAL_COLUMNS, True)
    resList = []
    classifier = CallXGBoost(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallCatBoost(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallKNN(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallNeuralNetwork(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallExtraTrees(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallPerceptron(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train)
    classifier = CallSVM(X_train, y_train)
    GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    # classifier = CallLinearSVC(X_train, y_train)
    # GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    # classifier = CallLogisticRegression(X_train, y_train)
    # GetPredictionAndResult(X_test, classifier, resList, y_test, X_train, y_train)
    max_res = max(resList)
    print("The best result for file name: " + file_name + " is: "+ str(max_res))


def GetPredictionAndResult(X_test, classifier, resList, y_test,X_train,y_train):
    #lassifier = FeatureSelectionRFE(classifier, 35, X_train, y_train, X_test,y_test)

    print("Pred train result")
    y_pred_train = classifier.predict(X_train)
    GetAndPrintResult(y_train, y_pred_train)
    # Predicting the Test set results
    print("Pred test result")
    y_pred = classifier.predict(X_test)
    res = GetAndPrintResult(y_test, y_pred)
    resList.append(res)

def FeatureSelectionRFE(classifier, k,X_train,y_train, X_test,y_test):
    print(f"FeatureSelection - RFE, n_features_to_select={k}")
    from sklearn.feature_selection import RFE
    selector = RFE(classifier, n_features_to_select=k, step=1)
    selector.fit_transform(X_train, y_train)
    selector.transform(X_test)
    return classifier
def FeatureSelectionKBest(X,y,k):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    print(f"Feature Selection - SelectKBest(chi2, k={k})")
    X_new = SelectKBest(chi2, k=k).fit_transform(X, y)
    X_new.shape
    return X_new

def FeatureSelectionPCA(X):
    print("FeatureSelection - PCA")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, svd_solver='full')
    X_new = pca.fit_transform(X)
    return X_new

def HandleMissingData(DECIMAL_COLUMNS, X):
       #Hendle missing data
    #change to avg value in the numeric columns (like age, height, weight...) 
    # i can change it to median
    from sklearn.impute import SimpleImputer
    print("HandleMissingData - strategy='mean'")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(X[:, 1:DECIMAL_COLUMNS])
    X[:, 1:DECIMAL_COLUMNS] = imputer.transform(X[:, 1:DECIMAL_COLUMNS])
    # where_are_NaNs = np.isnan(X)
    # X[where_are_NaNs] = -1  #??


#TODO: need to be in Statistics class
def FeatureImportance(classifier):
    print("feature importance:")
    from matplotlib import pyplot
    print(classifier.feature_importances_)
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
    print(f"The {classifier_name} predicted: {prediction}")
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
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    classLogisticRegression = CallLogisticRegression(X_train, y_train)
    classKNN = CallKNN(X_train, y_train)

    y_scoreLogisticRegression = classLogisticRegression.predict_proba(X_test)[:,1]
    y_scoreKNN = classKNN.predict_proba(X_test)[:,1]

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_scoreLogisticRegression)
    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_scoreKNN)

    print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_scoreLogisticRegression))
    print('roc_auc_score for y_scoreKNN: ', roc_auc_score(y_test, y_scoreKNN))

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - LogisticRegression')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - KNN')
    plt.plot(false_positive_rate2, true_positive_rate2)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

