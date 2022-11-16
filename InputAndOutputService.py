import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def GetDataset():
    DECIMAL_COLUMNS = 30
    file_name = "Processed Wisconsin Diagnostic Breast Cancer"
    sheet_name = "Processed Wisconsin Diagnostic"
    predict_column = 'diagnosis'
    cond_col = "None"

    #DECIMAL_COLUMNS = 1686
    #file_name = "FM_dataset - Processed"

    #DECIMAL_COLUMNS = 1675
    #file_name = "FM_dataset – objective values and bacterias"

    #DECIMAL_COLUMNS = 1620
    #file_name = "FM_dataset - onlyMicro"

    # sheet_name = "None"
    # predict_column = 'diagnosis'
    # cond_col = "None"

    #path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\';
    #path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\MIT\\files\\mimic-iii-clinical-database-1.4\\';
    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Breast Cancer\\';

    print("File name is: '"+ file_name+"', Sheet name is: '"+sheet_name+"'")
    #dataset = pd.read_excel(path+file_name+'.xlsx',sheet_name=sheet_name, engine='openpyxl')
    dataset = pd.read_csv(path+file_name+'.csv')
    list_feature_names = dataset.columns.values 

    return DECIMAL_COLUMNS, dataset, file_name, sheet_name, list_feature_names, predict_column, cond_col

def GetAndPrintResult(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)*100
    print("Accuracy: {:.2f} %".format(accuracy))
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Accuracy: {:.2f} %".format(accuracy))

    #print(classification_report(y_test, y_pred))
    return accuracy


def SplitTheDatasetTo2(full_dataset):
    dataset1 = full_dataset.sample(frac=0.5)
    columns_names = list(dataset1)
    dataset2 = pd.DataFrame(columns=columns_names)
    #dataset2 = dataset2.reset_index()  # make sure indexes pair with number of rows
    for index, row in full_dataset.iterrows():
        if not((dataset1['SEQN'] ==row['SEQN']).any()):
            dataset2 = dataset2.append(row, ignore_index=True)
 
    return dataset1, dataset2

def CreateRandomlyBalancedDatabaseByCondition(dataset,rows_number_per_cond, cond_value ,pred_col, cond_col):
    optionADataset = dataset[dataset[pred_col] == 0] #for exsample - all the sick
    optionBDataset = dataset[dataset[predcol] == 1] #for exsample - all the helthy

    balancedDataset = optionADataset[optionADataset[cond_col == cond_value]].sample(rows_number_per_cond)
    balancedDataset = balancedDataset.append(optionBDataset[optionBDataset[cond_col == cond_value]].sample(rows_number_per_cond))
    return balancedDataset

def CreateRandomlyBalancedDatabase(dataset,rows_number_per_cond,pred_col, cond_col):
    optionADataset = dataset[dataset[pred_col] == 0] #for example - all the sick
    optionBDataset = dataset[dataset[pred_col] == 1] #for example - all the helthy

    #for example - male
    balancedDataset = optionADataset[optionADataset[cond_col] == 1].sample(rows_number_per_cond)
    balancedDataset = balancedDataset.append(optionBDataset[optionBDataset[cond_col] == 1].sample(rows_number_per_cond))

    #for example - female
    balancedDataset = balancedDataset.append(optionADataset[optionADataset[cond_col] == 2].sample(rows_number_per_cond))
    balancedDataset = balancedDataset.append(optionBDataset[optionBDataset[cond_col] == 2].sample(rows_number_per_cond))

    return balancedDataset
    
#def GetInnerJoinDatasets(file_name1,file_name2,lefton, reghtin, new_file_name):
def GetInnerJoinDatasets():
    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\'
    file_name = 'FM_dataset - temp'
    sheet1 = 'design'
    sheet2 = 'data_T'
    #file_name1 = 'GenderPerICDDIAGNOSES'
    #file_name2 = 'D_ICD_DIAGNOSES'
    
    dataset1 = pd.read_excel(path+file_name+'.xlsx',sheet_name=sheet1, engine='openpyxl')
    dataset2 = pd.read_excel(path+file_name+'.xlsx',sheet_name=sheet2, engine='openpyxl')
    result = pd.merge(dataset1, dataset2, left_on='sample', right_on='OUT')
    result.to_csv(path + 'FM_dataset - Processed.csv')

def GetAllLabeventPerICD9(icd9):
    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\MIT\\files\\mimic-iii-clinical-database-1.4\\'
    file_name = 'DIAGNOSES_ICD'
    dataset1 = pd.read_csv(path+file_name+'.csv')
    subjectsList = []
    for index, row in dataset1.iterrows():
        if(row['ICD9_CODE'] == icd9):
            subjectsList.append(row['SUBJECT_ID'])
    file_name = 'LABEVENTS'
    dataset2 = pd.read_csv(path+file_name+'.csv')
    ItemidSet = set()
    for index,row in dataset2.iterrows():
        if(row['SUBJECT_ID'] in subjectsList):
            ItemidSet.add(row['ITEMID'])
    return ItemidSet
     


