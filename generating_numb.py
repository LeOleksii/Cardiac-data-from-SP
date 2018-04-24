import csv
import os
import numpy as np
import pandas as pd
import sklearn

dirr = "E:\cardiac_myopathy\data"
radio_path = dirr + "\\" + "feature-names.csv"
df = pd.read_csv(radio_path)
df1 = df.copy()
for x in range( df.shape[0] ):
    string2parce = df.iloc[x] 
    result_check = string2parce[0].split("_")
    if len(result_check) > 1 :        
        if (result_check[1] == 'glcm' or result_check[1] == 'glrlm' or result_check[1] == 'glszm' or result_check[1] == 'ngtdm' or result_check[1] == 'gldm') :
            result1 = result_check[0] + "_"+result_check[1]
        else:
            result1 = result_check[0]            
        df.iloc[x] = result1
    
lst = df['patient'].tolist() 
myset = set(lst)
unique = list(myset)

occur = [None] * len(unique)
new_names = [None] * len(unique)

for l in range( len(unique) ) :
    numb = df[df['patient'] ==  unique[l] ] 
    #numb = df[df['patient'].str.match(unique[ l ])]
    occur[l] = numb.index.values 
    test1 = df1.iloc[occur[l]].values.tolist()
    test2= ','.join(str(x) for x in test1)
    new_names[l] = test2.replace("[", "").replace("]", "").replace("'", " ")
    
#now let´s create a dataframe
occur2 = [x+2 for x in occur]
#uniqueDF = pd.DataFrame( { 'Radiomics_output_excel_number': occur, 'Names': unique, 'Variable_names': new_names})
uniqueDF = pd.DataFrame( { 'Radiomics_output_excel_number1': occur2, 'Names': unique, 'Variable_names': new_names})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('variable_index.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
uniqueDF.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
    
    
