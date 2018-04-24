import csv
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from pylab import rcParams
import seaborn as sb




def reading_files(directory):    
    #create a mask of patients who have diabetes
    patient_ID_diabetes = np.zeros((5065,), dtype = bool)
    #patient_ID_cardiomyopathy = np.zeros((5065,), dtype = bool)
    
    #run through all the files and select patient ID with diabetes and add this id to the mask
    for filename in os.listdir(directory):
        if 'csv' not in filename :
            continue
        location = directory +"\\" + filename
        #location = "E:\cardiac_myopathy\data\\" 
        df = pd.read_csv(location)
        
        binary_mask = df[df == "diabetes"]
        #binary_mask_c = df[df == "stroke"]
    
        for x in range(binary_mask.shape[0]):
            test = binary_mask[x:x+1]
            #test_c = binary_mask_c[x:x+1]
            if( test.isnull().sum().sum() != test.size):
                patient_ID_diabetes[x] = "True"
            #if( test_c.isnull().sum().sum() != test_c.size):
                #patient_ID_cardiomyopathy[x] = "True"    
    return patient_ID_diabetes; 

def normalize_values(dataframe_ex):
    df = dataframe_ex.copy()
    output_df = df[["Unnamed: 0", "patient"]].copy()
    cols = df.columns.tolist() #name of all the columns
    sizes = df.shape[1]
    for feat in range(2,sizes):
        column = np.array( df.iloc[:,feat:feat+1] )
        minimum = column.min()
        maximum = column.max()
        column = pd.DataFrame( [(x - minimum + 0.0001) / (maximum-minimum) for x in column] )
        column.index=output_df.index
        output_df.insert(feat,cols[feat], column)
    return output_df;

def normalize_values2(radiomics_IDD):
    df = radiomics_IDD.copy()
    cols = df.columns.tolist() #name of all the columns
    val = df.values[:,2:df.shape[1]]
    first2 = df.values[:,0:2]
    #now need to scale just starting from column 2
    scaled_features = preprocessing.StandardScaler().fit_transform(val)
    conc= np.concatenate((first2,scaled_features), axis=1)
    output = pd.DataFrame(data = conc,    # values
                          index = df.index,    # 1st column as index
                          columns = cols)  # 1st row as the column names
    return output;
    
    
    
def discard_error_features(radiomics_all , diabetes_ID_series):
    radiomics_IDD1 = radiomics_all[diabetes_ID_series.values]
    #now let´s discard patients whose features are all zeros(algorithm error)
    end_ind = radiomics_IDD1.shape[1]
    cut_mat = radiomics_IDD1.iloc[:,3:end_ind]#matrix without patient ID
    check = cut_mat.isnull().sum(axis=1).nonzero()#inex of zero rows
    radiomics_IDD1 = radiomics_IDD1.drop(radiomics_IDD1.index[check])#updated matrix
    return radiomics_IDD1;
      
###############################################################################    
dirr = "E:\cardiac_myopathy\data"
IDD = reading_files(dirr)

#now that we have patient IDs let´s pull the radiomics data of certain patients
radio_path = dirr + "\\" + "ARadiomics_ukbiobank_last.csv"
radiomics_all = pd.read_csv(radio_path)
diabetes_ID_series = pd.Series(IDD)
radiomics_IDD = discard_error_features(radiomics_all , diabetes_ID_series)
radiomics_norm = normalize_values2(radiomics_IDD)
####NOW WE CAN PERFORM PCA 
only_val = radiomics_norm.values[:,2:radiomics_norm.shape[1]]
scaler = preprocessing.StandardScaler().fit(only_val)
rad4PCA = scaler.transform(only_val)
# Make an instance of the Model
pca = PCA(.92)
pca.fit(rad4PCA)
numb_of_left_feat = pca.n_components_
new_feat = pca.transform(rad4PCA)


cols = radiomics_norm.columns.tolist()[2:] #name of all the columns
#SSS = pd.DataFrame(pca.components_, columns = cols,index = ['PC-1','PC-2'])

####3 or alternative for visualising

pca = PCA(2)
projected = pca.fit_transform(rad4PCA)
plt.scatter(projected[:, 0], projected[:, 1],
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))

#using hierarchical clustering on data processed with pca

z = linkage(new_feat,'ward')
#printing

dendrogram(z, truncate_mode='lastp',p=10,leaf_rotation=45,leaf_font_size=15, show_leaf_counts=True, show_contracted=True)
plt.title('clustering dendrogram')
plt.xlabel('cluster size')
plt.ylabel('Distance')
plt.show
plt.savefig(dirr + "\\" + "diabetes"+'.png')


max_d = 200
clusters = fcluster(z, max_d, criterion='distance')
#for this particular case let's plot the pca
indices1 = [i for i,v in enumerate(clusters < 2) if v]
indices2 = [i for i,v in enumerate(clusters > 1) if v]

class1 = projected[indices1,:]
class2 = projected[indices2,:]

plt.hold(True)
plt.scatter(class1[:, 0], class1[:, 1],
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10),c='b')

plt.scatter(class2[:, 0], class2[:, 1],
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10),c='b')

plt.savefig(dirr + "\\" + "diabetes_PCA_2_classes1"+'.png')















