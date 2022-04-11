import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, auc, cohen_kappa_score
import numpy as np
import gzip

output_dirs = "/storage/bfe_maruvka/guyshani/DGX_files/output_files_3000_4000"

outputdirs = os.listdir(output_dirs)

gene_list = []
accuracy_list = []
kappa_list = []
auc_list = []
ratios = []
pos_ratio_list = []

for dir in outputdirs:

    gene = dir.split("_")[1]
    try:
        print(f"{output_dirs}/{dir}/{gene}_patients_probs.csv")
        df = pd.read_csv(f"{output_dirs}/{dir}/{gene}_patients_probs.csv")
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
df.to_csv(f"{output_dirs}/auc_and_accuracy.csv")
