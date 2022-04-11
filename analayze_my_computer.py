
#  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    artificial
# balanced accuracy and percision recall

import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, auc, cohen_kappa_score
import numpy as np

gene_list = []
accuracy_list = []
kappa_list = []
auc_list = []
ratios = []
pos_ratio_list = []

filelist = os.listdir("/home/guysh/Documents/servers_data/dgx/artificiale_data/patients_probs")

for file in filelist:

    gene = file.split("_")[0]
    try:
        df = pd.read_csv(f"/home/guysh/Documents/servers_data/dgx/artificiale_data/patients_probs/{file}")
    except FileNotFoundError:
        print("No such file")
        continue
    except NotADirectoryError:
        break
    # get probabilitys (class low/0)
    model_probs = np.array(df["average"])

    high = 0
    low = 0
    t_labels = df["label"]
    t_labels = t_labels.replace("low", 0)
    t_labels = t_labels.replace("high", 1)
    # set the positive class (minority) to be "low"
    pos_label = 0

    # check which one is the minority class
    for i in range(len(t_labels)):
        if t_labels[i] == 1:
            high += 1
        else:
            low += 1

    # if 1 is the minority class, change the probs accordingly
    if high/low < 1:
        model_probs = 1- model_probs
        pos_label = 1

        pos_ratio = high/(high+low)
    else:
        pos_ratio = low/(high+low)


    # balanced accuracy score and cohens kappa

    if high/low < 1:
        new_list = [1 if model_probs[j] > 0.5 else 0 for j in list(range(len(model_probs)))]
    else:
        new_list = [0 if model_probs[j] > 0.5 else 1 for j in list(range(len(model_probs)))]
    BA = balanced_accuracy_score(t_labels, new_list)
    kappa = cohen_kappa_score(t_labels, new_list)



    # model probs = 1-probs that i got
    precision, recall, _ = precision_recall_curve(t_labels, model_probs, pos_label)
    auc_score = auc(recall, precision)


    gene_list.append(gene)
    accuracy_list.append(BA)
    kappa_list.append(kappa)
    auc_list.append(auc_score)
    ratios.append(high/low)
    pos_ratio_list.append(pos_ratio)

d = {'hugo_symbol': gene_list, 'balanced_accuracy': accuracy_list, 'precision_recall_auc': auc_list, 'cohens_kappa': kappa_list, 'ratio_hightolow': ratios, 'pos_ratio': pos_ratio_list}
df = pd.DataFrame(data=d)
df.to_csv(f"/home/guysh/Documents/servers_data/dgx/artificiale_output.csv")


#  p-values and fold schange

import os
import pandas as pd
import numpy as np
from scipy import stats

gene_list = []
average_ttest = []
average_ranked = []
logfoldchange = []

filelist = os.listdir("/home/guysh/Documents/servers_data/dgx/artificiale_data/patients_probs")

for file in filelist:

    gene = file.split("_")[0]
    if gene[0] != "g":
        continue

    try:
        df = pd.read_csv(f"/home/guysh/Documents/servers_data/dgx/artificiale_data/patients_probs/{file}")
    except FileNotFoundError:
        print("No such file")
        continue
    except NotADirectoryError:
        break

    average_probs_low = np.array(df['average'].loc[df['label'] == "low"])
    average_probs_high = np.array(df['average'].loc[df['label'] == "high"])


    average_ttest.append(stats.ttest_ind(average_probs_high, average_probs_low, equal_var=True, alternative='less')[1])
    average_ranked.append(stats.mannwhitneyu(average_probs_high, average_probs_low, alternative='less')[1])
    gene_list.append(gene)

    # calculate fold change

    logfoldchange.append(np.log2(np.mean(average_probs_low)/np.mean(average_probs_high)))

d = {'hugo_symbol': gene_list, 'average_t_test': average_ttest, 'average_ranked': average_ranked, 'log2foldchange': logfoldchange}
df = pd.DataFrame(data=d)
df.to_csv(f"/home/guysh/Documents/servers_data/dgx/artificiale_data/artificiale_pvals_foldchange.csv")

#  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4  oncogenes
# balanced accuracy and percision recall general

gene_list = []
kappa_list = []
accuracy_list = []
auc_list = []
ratios = []
pos_ratio_list = []

filelist = os.listdir("/home/guysh/Documents/servers_data/dgx/oncogenes/other/patients_probs")

for file in filelist:

    gene = file.split("_")[0]
    try:
        df = pd.read_csv(f"/home/guysh/Documents/servers_data/dgx/oncogenes/other/patients_probs/{file}")
    except FileNotFoundError:
        print("No such file")
        continue
    except NotADirectoryError:
        break
    # get probabilitys (class low/0)
    model_probs = np.array(df["average"])

    high = 0
    low = 0
    t_labels = df["label"]
    t_labels = t_labels.replace("low", 0)
    t_labels = t_labels.replace("high", 1)
    # set the positive class (minority) to be "low"
    pos_label = 0

    # check which one is the minority class
    for i in range(len(t_labels)):
        if t_labels[i] == 1:
            high += 1
        else:
            low += 1

    # if 1 is the minority class, change the probs accordingly
    if high/low < 1:
        model_probs = 1- model_probs
        pos_label = 1

        pos_ratio = high/(high+low)
    else:
        pos_ratio = low/(high+low)


    #  balanced accuracy score and cohens kappa
    if high/low < 1:
        new_list = [1 if model_probs[j] > 0.5 else 0 for j in list(range(len(model_probs)))]
    else:
        new_list = [0 if model_probs[j] > 0.5 else 1 for j in list(range(len(model_probs)))]
    BA = balanced_accuracy_score(t_labels, new_list)
    kappa = cohen_kappa_score(t_labels, new_list)



    # model probs = 1-probs that i got
    precision, recall, _ = precision_recall_curve(t_labels, model_probs, pos_label)
    auc_score = auc(recall, precision)


    gene_list.append(gene)
    kappa_list.append(kappa)
    accuracy_list.append(BA)
    auc_list.append(auc_score)
    ratios.append(high/low)
    pos_ratio_list.append(pos_ratio)

d = {'hugo_symbol': gene_list, 'balanced_accuracy': accuracy_list, 'precision_recall_auc': auc_list, 'cohens_kappa': kappa_list, 'ratio_hightolow': ratios, 'pos_ratio': pos_ratio_list}
df = pd.DataFrame(data=d)
df.to_csv(f"/home/guysh/Documents/servers_data/dgx/oncogenes_output.csv")

#  p-values and fold schange

import os
import pandas as pd
import numpy as np
from scipy import stats

gene_list = []
average_ttest = []
average_ranked = []
logfoldchange = []

filelist = os.listdir("/home/guysh/Documents/servers_data/dgx/oncogenes/other/patients_probs")

for file in filelist:

    gene = file.split("_")[0]
    #if gene[0] != "g":
    #    continue

    try:
        df = pd.read_csv(f"/home/guysh/Documents/servers_data/dgx/oncogenes/other/patients_probs/{file}")
    except FileNotFoundError:
        print("No such file")
        continue
    except NotADirectoryError:
        break

    average_probs_low = np.array(df['average'].loc[df['label'] == "low"])
    average_probs_high = np.array(df['average'].loc[df['label'] == "high"])


    average_ttest.append(stats.ttest_ind(average_probs_high, average_probs_low, equal_var=True, alternative='less')[1])
    average_ranked.append(stats.mannwhitneyu(average_probs_high, average_probs_low, alternative='less')[1])
    gene_list.append(gene)

    # calculate fold change

    logfoldchange.append(np.log2(np.mean(average_probs_low)/np.mean(average_probs_high)))

d = {'hugo_symbol': gene_list, 'average_t_test': average_ttest, 'average_ranked': average_ranked, 'log2foldchange': logfoldchange}
df = pd.DataFrame(data=d)
df.to_csv(f"/home/guysh/Documents/servers_data/dgx/oncogenes_pvals_foldchange.csv")



# $$$$$$$$$$$$$$$$$$$$$$$$$$$ correlations


filelist = os.listdir("/home/guysh/Documents/servers_data/dgx/oncogenes/other/patients_probs")

for file in filelist:

    gene = file.split("_")[0]
    try:
        df = pd.read_csv(f"/home/guysh/Documents/servers_data/dgx/oncogenes/other/patients_probs/{file}")
    except FileNotFoundError:
        print("No such file")
        continue
    except NotADirectoryError:
        break
    # get probabilitys (class low/0)
    model_probs = np.array(df["average"])

    high = 0
    low = 0
    t_labels = df["label"]
    t_labels = t_labels.replace("low", 0)
    t_labels = t_labels.replace("high", 1)
    # set the positive class (minority) to be "low"
    pos_label = 0

    # check which one is the minority class
    for i in range(len(t_labels)):
        if t_labels[i] == 1:
            high += 1
        else:
            low += 1

    # if 1 is the minority class, change the probs accordingly
    if high/low < 1:
        model_probs = 1- model_probs
        pos_label = 1

    BA_ref = 0
    # find the cutoff that gives the best balanced accuracy score
    for i in list(np.arange(1,10)/10):
        if high/low < 1:
            new_list = [1 if model_probs[j] > i else 0 for j in list(range(len(model_probs)))]
        else:
            new_list = [0 if model_probs[j] > i else 1 for j in list(range(len(model_probs)))]
        BA = balanced_accuracy_score(t_labels, new_list)
        if BA > BA_ref:
            BA_ref = BA
            cutoff = i


    # model probs = 1-probs that i got
    precision, recall, _ = precision_recall_curve(t_labels, model_probs, pos_label)
    auc_score = auc(recall, precision)


    gene_list.append(gene)
    cutoff_list.append(cutoff)
    accuracy_list.append(BA_ref)
    auc_list.append(auc_score)
    ratios.append(high/low)

d = {'hugo_symbol': gene_list, 'cutoff': cutoff_list, 'balanced_accuracy': accuracy_list, 'precision_recall_auc': auc_list, 'ratio_hightolow': ratios}
df = pd.DataFrame(data=d)
df.to_csv(f"/home/guysh/Documents/servers_data/dgx/oncogenes/other/oncogenes_auc_and_accuracy.csv")
