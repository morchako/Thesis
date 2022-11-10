from Constants import *
from InputAndOutputService import *

def CallXGBoost(X_train, y_train):
    print("using XGboost")
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def CallNeuralNetwork(X_train, y_train):
    print("using sklearn.neural_network MLPClassifier")
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(100,100), random_state=1, solver='lbfgs')
    classifier.fit(X_train, y_train)
    return classifier

def CallSVM(X_train, y_train):
    print("using SVM")
    from sklearn.svm import SVC
    #classifier = SVC(kernel = 'linear', random_state = 0)
    classifier = SVC(kernel = 'poly')
    classifier.fit(X_train, y_train)
    return classifier

def CallKNN(X_train, y_train):
    print("using KNN")
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    return classifier

def CallLogisticRegression(X_train, y_train):
    print("using Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    return classifier

def CallCatBoost(X_train, y_train):
    print("using CatBoost")
    from catboost import CatBoostClassifier
    classifier = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
    classifier.fit(X_train, y_train)
    return classifier


def FeatureScaling(X_train,X_test,DECIMAL_COLUMNS):
     #Feature Scaling - only the numeric values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[:, 1:DECIMAL_COLUMNS] = sc.fit_transform(X_train[:, 1:DECIMAL_COLUMNS])
    X_test[:, 1:DECIMAL_COLUMNS] = sc.transform(X_test[:, 1:DECIMAL_COLUMNS])

def PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
                                       algoritem_name = XG_BOOST, handle_missing_data = False):
    X = dataset.iloc[:, :-1].values        
    y = dataset.iloc[:, -1].values
    
    if(handle_missing_data):
        HandleMissingData(DECIMAL_COLUMNS, X)
    
    #Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    FeatureScaling(X_train,X_test,DECIMAL_COLUMNS)
    
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
    elif(algoritem_name == NEURAL_NETWORK):
        classifier = CallNeuralNetwork(X_train, y_train)

    #Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("File name is: '"+ file_name+"', Sheet name is: '"+sheet_name+"'")
    res = GetAndPrintResult(y_test, y_pred)
    return res

def HandleMissingData(DECIMAL_COLUMNS, X):
       #Hendle missing data
    #change to avg value in the numeric columns (like age, height, weight...) 
    # i can change it to median
    from sklearn.impute import SimpleImputer
    print(X)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:DECIMAL_COLUMNS])
    X[:, 1:DECIMAL_COLUMNS] = imputer.transform(X[:, 1:DECIMAL_COLUMNS])

    where_are_NaNs = np.isnan(X)
    X[where_are_NaNs] = -1  #??
    print(X[:,:])

