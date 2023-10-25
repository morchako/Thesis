import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ClassificationAlgorithms import *
from Constants import *
from Statistics import *
from DataProcessingHelper import *
from ReproducingResultsService import *

def main():

    #CreateSumAndCountFeatures()
    #CreateFamilyDataset()
    #GetInnerJoinDatasets()
    # path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\'
    DECIMAL_COLUMNS, dataset, file_name, sheet_name, list_feature_names, pred_col, cond_col = GetDataset()
    #use_hyperparameter_search_with_SVM(dataset, DECIMAL_COLUMNS, True,list_feature_names)
    # create_new_dataframe_by_avg_diffrence(dataset)
    #PredictUsingAllCalssificationAlgoritems(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names)
    res = predict_with_knn_with_treshold(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names, threshold = 0.5)
    #ReproducingResult_LASSO_SVM(dataset, pred_col)
    # call_predict_by_feature(dataset,list_feature_names,path+file_name+"_1by1_output")
    #ReproducingResult_LASSO_SVM_kfold(dataset, pred_col)
    #get_inner_multiplication_matrix(dataset)
    # GetStatisticsForRandomBalancedDatabase(dataset, DECIMAL_COLUMNS, file_name, sheet_name,list_feature_names,
    #                                    pred_col, cond_col, algoritem_name = SVM, handle_missing_data = False)
    # res = PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
    #                                          algoritem_name = SVM, handle_missing_data = True)
    print(res)
    #GenderPerICD(dataset)
    #SortAndCountGenderByICD9Diagnoses(dataset)


if __name__ == '__main__':
    main()
