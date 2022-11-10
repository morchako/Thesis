import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ClassificationAlgorithms import *
from Constants import *
from Statistics import *



def main():
#def CardiacPrediction_XGBoost_main():
    #Get Dataset
    #GetAllLabeventPerICD9('4019')
    #GetInnerJoinDatasets()
    DECIMAL_COLUMNS, dataset, file_name, sheet_name, list_feature_names, pred_col, cond_col = GetDataset()
    #GetStatisticsForRandomBalancedDatabase(dataset, DECIMAL_COLUMNS, file_name, sheet_name,list_feature_names,
    #                                   pred_col, cond_col, algoritem_name = SVM, handle_missing_data = False)
    res = PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
                                           algoritem_name = NEURAL_NETWORK, handle_missing_data = False)    
    print(res)
    #GenderPerICD(dataset)
    #SortAndCountGenderByICD9Diagnoses(dataset)


if __name__ == '__main__':
    main()
