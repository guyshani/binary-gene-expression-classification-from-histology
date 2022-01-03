'''
this script will search for the rows in the countmatrix of the genes in the cancer gene list from cosmic

'''


import pandas as pd
import numpy as np

#with open("/home/Documents/gene_lists_forcountsrun/Census_allSun Dec_5_14_46_03_2021.csv", "r") as oncogenes:
oncogenes = pd.read_csv("/home/guysh/Documents/gene_lists_forcountsrun/Census_allSun Dec_5_14_46_03_2021.csv", usecols=["Gene Symbol"])
oncogenes = oncogenes['Gene Symbol'].tolist()

countmatrix = pd.read_csv("/home/guysh/Documents/gene_lists_forcountsrun/countmatrix_normalized_10000_symbols.csv")
countmatrix = countmatrix.drop(['Unnamed: 0'], axis=1)
print(countmatrix)

countmarix_indecies =[]

for symbol in oncogenes:
    if countmatrix.loc[countmatrix.symbols == symbol].index.values:
        countmarix_indecies.append(np.int(countmatrix.loc[countmatrix.symbols == symbol].index.values))

oncomatrix = countmatrix.iloc[countmarix_indecies]
oncomatrix.to_csv("/home/guysh/Documents/gene_lists_forcountsrun/oncogenes_countmatrix.csv",header=True, index=True)
