from InputAndOutputService import *
from ClassificationAlgorithms import *
import statistics

def StatisticsPerColumn(full_dataset, isCondition= False, theConditionColumn = None, conditionValue=None):
    if(isCondition):
       dataset = full_dataset[full_dataset['CoronaryHeartDisease'] == 1]
    else:
       dataset= full_dataset

    data_top = dataset.head()    
    
    with open(file_name+'_'+sheet_name+'_WithCoronaryHeartDisease_Statistics.txt', 'w') as f:
        #f.write(dataset.describe())
        f.write("File name is: '"+ file_name+"', Sheet name is: '"+sheet_name+"' with CoronaryHeartDisease"+'\n')
        for col in data_top:
            f.write(col +': variance: ' +str(statistics.variance(dataset[col]))
                     +' mean: '+ str(statistics.mean(dataset[col]))
                     +' stdev: '+ str(statistics.stdev(dataset[col])) +'\n')

def GetStatisticsForRandomBalancedDatabase(dataset, DECIMAL_COLUMNS, file_name, sheet_name,list_feature_names,
                                       pred_col, cond_col, algoritem_name = XG_BOOST, handle_missing_data = False):
    res_list = []
    for x in range(10):
        balanced_dataset = CreateRandomlyBalancedDatabase(dataset,480,pred_col, cond_col)
        #balanced_dataset = CreateRandomlyBalancedDatabaseByCondition(dataset,480,0,pred_col, cond_col)
        res = PredictUsingCalssificationAlgoritem(balanced_dataset, DECIMAL_COLUMNS, file_name, sheet_name, list_feature_names,
                                           algoritem_name, handle_missing_data)
        res_list.append(res)
    print(res_list)
    print('variance: ' +str(statistics.variance(res_list))
                     +' mean: '+ str(statistics.mean(res_list))
                     +' stdev: '+ str(statistics.stdev(res_list)) +'\n')

def CalculationOfStandardDeviationOfTheDifferenceBetweenComplementaryHalves(DECIMAL_COLUMNS, dataset, file_name, sheet_name):
    difference_list = []
    
    for x in range(30):
        dataset1, dataset2 = SplitTheDatasetTo2(dataset)
        res1 = PredictUsingCalssificationAlgoritem(dataset1, DECIMAL_COLUMNS, file_name, sheet_name,
                                           algoritem_name = XG_BOOST, handle_missing_data = False)
        res2 = PredictUsingCalssificationAlgoritem(dataset2, DECIMAL_COLUMNS, file_name, sheet_name,
                                           algoritem_name = XG_BOOST, handle_missing_data = False)
        absolute_value_difference = abs(res1-res2)
        difference_list.append(absolute_value_difference)     
        print(str(res1) + " - " + str(res2) + " = " + str(res1-res2))
    
    print(difference_list);

def GenderPerICD(dataset):
    dict = {}
    for index, row in dataset.iterrows():
         icd = row["ICD9_CODE"];
         if icd not in dict:
             #[0] = Female, [1] = Male, [2] = count
             dict[icd] = [0,0,0];
         if(row["MALE"] == 0):
             dict[icd][0]= dict[icd][0]+1
         else:
             dict[icd][1] = dict[icd][1]+1
         dict[icd][2] = dict[icd][2]+1
    print(dict)
    df = pd.DataFrame(data=dict)
    df = (df.T)
    print (df)
    df.to_excel('GenderPerICD.xlsx')
          
def GetIDC9Key(icd):
    if(icd.startswith(('0','10','11','12','13'))): #001-139
        key = "Infectious and Parasitic Diseases"
    elif(icd.startswith(('1','20','21' , '22' , '23'))):#140-239
        key = "Neoplasms (140-239)"
    elif(icd.startswith(('24' , '25' , '26' , '27'))):#240-279
        key = "Endocrine, Nutritional and Metabolic Diseases, and Immunity Dis,ders (240-279)"
    elif(icd.startswith('28')):#280-289
        key = "Diseases of the Blood and Blood-forming Organs (280-289)"
    elif(icd.startswith(('29' , '30' , '31'))):#290-319
        key = "	Mental Disorders (290-319)"
    elif(icd.startswith(('39' , '40' , '41' , '42', '43','45'))):#390-459
        key = "Diseases of the Circulatory System (390-459)"
    elif(icd.startswith(('3'))): #320-389
        key = "Diseases of the Nervous System and Sense Organs(320-389)"
    elif(icd.startswith(('4', '50', '51'))): #460-519
        key = "Diseases of the Respiratory System (460-519)"
    elif(icd.startswith(('58', '59', '60', '61', '62'))): #580-629
        key = "Diseases of the Genitourinary Systemm (580-629)"
    elif(icd.startswith('5')): #520-579
        key = "Diseases of the Digestive System (520-579)"
    elif(icd.startswith(('63', '64', '65', '66', '67'))): #630-679
        key = "Complications of Pregnancy, Childbirth, and the Puerperium (630-679)"
    elif(icd.startswith(('6', '70'))): #680–709
        key = "Diseases of the Skin and Subcutaneous Tissue (680–709)"
    elif(icd.startswith(('71', '72', '73'))): #710–739
        key = "Diseases of the Musculoskeletal System and Connective Tissue (710–739)"
    elif(icd.startswith(('74', '75'))): #740–759
        key = "Congenital Anomalies (740–759)"
    elif(icd.startswith(('76', '77'))): #760–779
        key = "Certain Conditions originating in the Perinatal Period (760–779)"
    elif(icd.startswith('7')): #780–799
        key = "Symptoms, Signs and Ill-defined Conditions (780–799)"
    elif(icd.startswith(('8', '9'))): #800–999
        key = "Injury and Poisoning (800–999)"
    elif(icd.startswith('E')): #E800–E999
        key = "	Supplementary Classification of External Causes of Injury and Poisoning(E800–E999)"
    elif(icd.startswith('V')): #V01–V82
        key = "	Supplementary Classification of Factors influencing Health Status and Contact with Health Services (V01–V82)"







    return key

def SortAndCountGenderByICD9Diagnoses(dataset): #file_name = "GenderPerICDDIAGNOSES" 
    dict = {}
    for index, row in dataset.iterrows():
         icd = row["ICD9_DIAGNOSES"];
         #if(icd.startswith('10') or icd.startswith('11') or icd.startswith('12') or icd.startswith('13')):
         key = GetIDC9Key(icd)
         if key not in dict:
            dict[key] = [0,0,0];
         dict[key][0] =  dict[key][0]+row["female"]
         dict[key][1] =  dict[key][1]+row["male"]
         dict[key][2] =  dict[key][2]+row["count"]

    print(dict)
    df = pd.DataFrame(data=dict)
    df = (df.T)
    print (df)
    df.to_excel('SortAndCountGenderByICD9Diagnoses.xlsx')
          
