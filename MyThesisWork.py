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
    # top_features_name_list = ['Prevotella_copri_1', 'Prevotella_12', 'Bacteroides_uniformis_1',
    #                  'Bacteroides_dorei_1','Bacteroides_MS_3', 'Bacteroides_vulgatus_2','Parabacteroides_merdae_3',
    #                  'Alloprevotella_1','Alistipes_finegoldii_2','Bacteroides_3','Prevotella_4','Parabacteroides_merdae_1',
    #              'Prevotella_5','Firmicutes_MG_4', 'Ruminococcaceae_MG_1'] #15 top

    # list_feature_names = ['Prevotella_copri_1','Prevotella_12','Bacteroides_uniformis_1','Bacteroides_dorei_1','Alistipes_finegoldii_2',
    #                           'Bacteroides_MS_3','Parabacteroides_merdae_3','Ruminococcaceae_MG_1','Alloprevotella_1',
    #                           'Bacteroides_3','Prevotella_4','Akkermansia_muciniphila_1']#12 top
    # # plot_dataset_by_classifier(dataset,list_feature_names)

    # list_feature_names = ['Bacteroides_vulgatus_1','Prevotella_copri_1','Prevotella_12','Bacteroides_uniformis_3']
    # create_bar_graph_of_a_statistical_test_for_feature_for_different_labels(dataset, list_feature_names)
    PredictUsingAllCalssificationAlgoritems(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names)
    # manova_interacion_test_to_every_pair(dataset, top_features_name_list)
    # correlation_test(dataset, top_features_name_list)
    # plot_everything(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names)
    # use_hyperparameter_search_with_SVM(dataset, DECIMAL_COLUMNS, True,list_feature_names)
    #use_hyperparameter_search_with_LogisticRegression(dataset, DECIMAL_COLUMNS, True,list_feature_names)
    # create_new_dataframe_by_avg_diffrence(dataset)
    #KNN_with_leaveoneout(dataset, list_feature_names)
    #KNN_Kfoldcrossvalidation(dataset,list_feature_names)

    # res = predict_with_knn_with_treshold(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names, threshold = 0.5)
    # #ReproducingResult_LASSO_SVM(dataset, pred_col)
    # call_predict_by_feature(dataset,list_feature_names,path+file_name+"_1by1_output")
    #ReproducingResult_LASSO_SVM_kfold(dataset, pred_col)
    #get_inner_multiplication_matrix(dataset)
    # GetStatisticsForRandomBalancedDatabase(dataset, DECIMAL_COLUMNS, file_name, sheet_name,list_feature_names,
    #                                    pred_col, cond_col, algoritem_name = SVM, handle_missing_data = False)
    # res = PredictUsingCalssificationAlgoritem(dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
    #                                          algoritem_name = SVM, handle_missing_data = True)
    # print(res)
    #GenderPerICD(dataset)
    #SortAndCountGenderByICD9Diagnoses(dataset)


if __name__ == '__main__':
    main()
