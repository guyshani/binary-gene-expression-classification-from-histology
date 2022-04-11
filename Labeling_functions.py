import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, roc_auc_score, balanced_accuracy_score
import numpy as np
from plotnine import *


def expression_file(gene):
     #use this function with FPKM/FPKM-UQ files.

    '''
    create an expression file         ***FPKM-UQ

    this part is going to create a id_list_{gene},csv that will contain the FPKM name, TCGA name and the expression for that gene,
    for that patient.
    '''

    # id_list.csv - a filr that containes the name of the FPKM file on the first column,
    # and the patient tag (TCGA-3456-01A) on the second column
    with open(FPKM_files+"id_list_FPKM_UQ_CRCdataset.csv", "r") as id_lst:
        with open(csv_location+f"id_list_UQ_{gene}.csv", "w") as new_lst:
            for name in id_lst:
                filename = name.split(',')[1]
                tcga = name.split(',')[0]
                filename = filename.strip()
                fpkm = open(FPKM_files+f"{filename}", "r")
                for line in fpkm:
                    if re.match(f'{ensembl}', line):
                        line = line.split("\t")[1]
                        line = line.strip()
                        new_lst.write(tcga + ',' + line + '\n')
    fpkm.close()
    df = pd.read_csv(csv_location+f'id_list_UQ_{gene}.csv', header =None)
    df.columns=['tcga_name','expression']

    return df


def expression_dataframe(gene):

    '''
    this function recieves a counts matrix with the format: "gene symbols, gene ensembls, sample1, sample2, ..."
    create a dataframe df with TCGA names and expression for the gene.
    '''

    #data = pd.read_csv("/home/guysh/Downloads/counts/countmatrix_normalized_10000_symbols.csv")
    data = pd.read_csv("countmatrix_normalized_10000_symbols.csv")
    df = data.loc[data['symbols'] == gene]
    df=df.drop(["symbols","Row.names"], axis = 1).T
    df = df.reset_index()
    df.columns = ["tcga_name", "expression"]
    df = df.drop(index = 0)
    df = df.reset_index(drop = True)

    return df


def fit_GM(df, gene, csv_location, output_files):

    '''
    this function trys to fit a gaussian mixture model with 2 disterbutions and split samples into classes accordinly
    if one class has less then 10 samples, the classes are decided by the average expression.
    the class labels are then stored in a new column (cluster) in df.
    '''

    GM = GaussianMixture(n_components=2).fit(df[['expression']])
    #GM.means_
    cluster = GM.predict(df[['expression']])
    params=GM.get_params()

    df["cluster"] = cluster
    #df["expression"]
    if np.mean(df['expression'].loc[df.query(f'(cluster == "0")').index.values].values) > np.mean(df['expression'].loc[df.query(f'(cluster == "1")').index.values].values):
        df['cluster'] = df['cluster'].replace(1,'low')
        df['cluster'] = df['cluster'].replace(0,'high')
    else:
        df['cluster'] = df['cluster'].replace(0,'low')
        df['cluster'] = df['cluster'].replace(1,'high')

    # checks if one of the groups is not too small
    if len(np.where(df['cluster'] == 'high')[0]) < 20 or len(np.where(df['cluster'] == 'low')[0]) <10:

        # if one of the classes have less then 20 patients then split the class by the average expression.

        mean = np.nanmean(df['expression'])
        df["cluster"].loc[df.query(f'(expression <= {mean})').index.values] = 'low'
        df["cluster"].loc[df.query(f'(expression > {mean})').index.values] = 'high'

        print("^^^^^^^" +gene+"  is using average for class split")



    #seperation between the groups (1 is best and -1 is worse)
    print("silhouette_score:  "+str(silhouette_score(df[['expression']], cluster)))
    #df.to_csv(output_files+f"{gene}_crossval.csv", columns= ["tcga_name", "cluster"],header=False, index=False)

    print("number of patients in \"high\" group: "+ str(len(np.where(df['cluster'] == 'high')[0])))
    print("number of patients in \"low\" group: "+ str(len(np.where(df['cluster'] == 'low')[0])))

    # save a plot of the samples expression disterbution, colored by classes.
    p = (
    ggplot(df)
    +geom_freqpoly(aes(x = 'expression', group = 'cluster', color = 'cluster'))
    +geom_histogram(aes(x = 'expression', fill = 'cluster'), alpha = 0.5)
    +labs(title = f'{gene}')
    )
    ggsave(plot = p, filename = f"{gene}_crossval.pdf", path = output_files)

    return df



def stats_and_weights(train_csv, test_csv, csv_location):

    '''
        parameters:
            train_csv: a csv file containing the image names (tiles) of all the patients of the trainning set with labels
            test_csv: a csv file containing the image names (tiles) of all the patients of the test set with labels
            csv_location: path to the directory that contains the csv files

        returns:
            weight: the ratio of the images labeled "low" vs labeled "high", to be used in the loss function during trainning

    '''

    image_num = 0
    train = []
    test = []
    for data in [train_csv, test_csv]:
        # create a list of class labels for train set and for test set
        with open(csv_location+f"{data}", "r") as datafile:
            for line in datafile:
                image_num += 1
                if data == train_csv:
                    train.append(line.split(",")[1].strip())
                elif data == test_csv:
                    test.append(line.split(",")[1].strip())

    print("total image number in train: "+str(len(train)))
    print("low: "+str(train.count("low")))
    print("high: "+str(train.count("high")))
    print("total image number in test: "+str(len(test)))
    print("low: "+str(test.count("low")))
    print("high: "+str(test.count("high")))


    # set the weight ration for the loss function

    try:
        weight = test.count("low")/test.count("high")
        print("wieght (low/high): "+str(weight))
    except ZeroDivisionError:
        print(ZeroDivisionError)

    return weight
