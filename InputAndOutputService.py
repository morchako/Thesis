import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

resDict = {}

def GetDataset():
    # DECIMAL_COLUMNS = 30
    # file_name = "Processed Wisconsin Diagnostic Breast Cancer"
    # sheet_name = "Processed Wisconsin Diagnostic"
    # predict_column = 'diagnosis'
    # cond_col = "None"

    #DECIMAL_COLUMNS = 1686
    #file_name = "FM_dataset - Processed"

    #DECIMAL_COLUMNS = 1675
    #file_name = "FM_dataset – objective values and bacterias"

    # DECIMAL_COLUMNS = 1620
    # file_name = "FM_dataset - onlyMicro"

    #DECIMAL_COLUMNS = 62
    #file_name = 'FM_dataset - SumFamily'

    DECIMAL_COLUMNS = 152
    file_name = 'FM_dataset - Sum Family and count'

    # DECIMAL_COLUMNS = 308
    # file_name = 'FM_dataset - Sum Genus and count'

    # DECIMAL_COLUMNS = 317
    # file_name = 'FM_dataset - SumSpecies and diet'

    # DECIMAL_COLUMNS = 604
    # file_name = 'FM_dataset - Sum Species and count'

    # DECIMAL_COLUMNS = 288
    # file_name = 'FM_dataset - SumSpecies'

    sheet_name = "None"
    predict_column = 'diagnosis'
    cond_col = "None"

    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\';
    #path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Breast Cancer\\';

    print("File name is: '"+ file_name+"', Sheet name is: '"+sheet_name+"'")
    #dataset = pd.read_excel(path+file_name+'.xlsx',sheet_name=sheet_name, engine='openpyxl')
    dataset = pd.read_csv(path+file_name+'.csv')
    list_feature_names = dataset.columns.values 

    return DECIMAL_COLUMNS, dataset, file_name, sheet_name, list_feature_names, predict_column, cond_col

def GetAndPrintResult(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    cm = confusion_matrix(y_test, y_pred)
    PrintAndWriteToLog(str(cm))
    accuracy = accuracy_score(y_test, y_pred)*100
    PrintAndWriteToLog("Accuracy: {:.2f} %".format(accuracy))
    #print(classification_report(y_test, y_pred))
    return accuracy

def PrintAndWriteToLog(logMsg,AddToTable = False):
    print(logMsg)
    with open('ThesisWorkLog.txt', 'a') as f:
        f.writelines(str(datetime.now())+"  "+ logMsg+'\n')
        f.close()
    if(AddToTable):
        resDict['Notes'] = resDict['Notes'] + logMsg +'\n'

def SaveResultToCSV(resDict):
    from csv import DictWriter
    field_names = ['Date and time','File Name','Notes','XGBoost','CatBoost',
                   'SVM','KNN','MLPClassifier','ExtraTreesClassifier',
                   'Perceptron','Best result']
    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\'
    with open(path+'Thesis Documentation.csv', 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(resDict)
        f_object.close()

def get_inner_hem(df):
  inner_hem = []
  for i in range(1, df.shape[0] - 1):
    for j in range(1, df.shape[1] - 1):
      inner_hem.append([df.iloc[i, j], df.iloc[i, j-1], df.iloc[i, j+1], df.iloc[i-1, j], df.iloc[i+1, j]])
  return pd.DataFrame(inner_hem, columns=['center', 'left', 'right', 'top', 'bottom'])


def CreateSumAndCountFeatures():
    # df = pd.DataFrame([[0, 0, 3], [4, 0, 6], [7, 0, 0], [0, 1, 2], [1, 1, 1], [0, 1, 0]], columns=['C1', 'C2', 'C3'],
    #                   index=['A', 'A', 'B', 'B', 'B', 'C'])
    file_name = "FM_dataset - Species and pt"
    print("File name is: '" + file_name)
    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\'
    df = pd.read_csv(path + file_name + '.csv')
    df = df.set_index('Species')
    df_sum = df.copy()
    df_sum.index = pd.Series([x + '-sum' for x in df.index])
    df_sum
    for k in df_sum.columns:
        df_sum[k] = df_sum[k].where(df_sum[k] == 0, 1)
    df = pd.concat([df, df_sum])
    appearances = df.index.value_counts()
    res = []
    for k in appearances.index:
        if appearances[k] > 1:
            res.append(df.loc[k].values.sum(axis=0))
        else:
            res.append(df.loc[k])
    A = np.array(res)
    df = pd.DataFrame(A, columns=df.columns, index=appearances.index)
    df = df.transpose()
    print(df)
    df.to_csv(path + 'FM_dataset - Sum Species and count.csv')
def CreateFamilyDataset():
    #file_name = "FM_dataset - family and pt"
    file_name = "FM_dataset - Family and pt"
    print("File name is: '"+ file_name)
    path = 'C:\\Users\\Mor\\OneDrive\\Documents\\Thesis\\DS for training\\Fibro\\'
    df = pd.read_csv(path+file_name+'.csv')
    df.set_index('Family')
    dict = {}
    # for k in df.columns:
    #     df[k + '-FC'] = df[k].where(df[k] == 0, 1)
    #
    df2 = df
    for index, row in df.iterrows():
        family_index = row["Family"];
        sum_index = f"{family_index}_sum"
        df2[sum_index] = df[index].where(df[index] == 0, 1)

    for index, row in df.iterrows():
         family_index = row["Family"];
         #sum_index = f"{family_index}_sum"
         if family_index not in dict:
             dict[family_index] = row;
         else:
             dict[family_index] = dict[family_index] + row
             dict[family_index]["Family"] = family_index
    for index, row in df2.iterrows():
         family_index = row["Family"];
         #sum_index = f"{family_index}_sum"
         if family_index not in dict:
             dict[family_index] = row;
         else:
             dict[family_index] = dict[family_index] + row
             dict[family_index]["Family"] = family_index
    print(dict)
    df1 = pd.DataFrame(data=dict)
    #df1 = pd.DataFrame(df, columns=df.columns, index=appearances.index)
    df1.to_csv(path + 'FM_dataset - Sum Family and count try1.csv')
    return df1

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
     


