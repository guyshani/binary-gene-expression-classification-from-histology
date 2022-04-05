import os
import re
import pandas as pd
import numpy as np
from scipy import stats


output_dirs = "/storage/bfe_maruvka/guyshani/DGX_files/output_guy_20_2"

outputdirs = os.listdir(output_dirs)

gene_list = []
average_ttest = []
median_ttest = []
average_ranked = []
median_ranked = []
logfoldchange = []

for dir in outputdirs:

    gene = dir.split("_")[0]
    try:
        df = pd.read_csv(f"{output_dirs}/{dir}/{gene}_patients_probs.csv")
    except FileNotFoundError:
        print("No such file")
        continue
    except NotADirectoryError:
        break
    # get probabilitys (class low/0)
    average_probs_low = np.array(df["average"].loc[df['label'] == "low"])
    average_probs_high = np.array(df["average"].loc[df['label'] == "high"])
    median_probs_low = np.array(df["median"].loc[df['label'] == "low"])
    median_probs_high = np.array(df["median"].loc[df['label'] == "high"])



    average_ttest.append(stats.ttest_ind(average_probs_high, average_probs_low, equal_var=True, alternative='less')[1])
    median_ttest.append(stats.ttest_ind(median_probs_high, median_probs_low, equal_var=True, alternative='less')[1])
    average_ranked.append(stats.mannwhitneyu(average_probs_high, average_probs_low, alternative='less')[1])
    median_ranked.append(stats.mannwhitneyu(median_probs_high, median_probs_low, alternative='less')[1])
    gene_list.append(gene)


    # calculate fold change

    logfoldchange.append(np.log2(np.mean(average_probs_low)/np.mean(average_probs_high)-1))


d = {'hugo_symbol': gene_list, 'average_t_test': average_ttest, 'median_t_test': median_ttest, 'average_ranked': average_ranked, 'median_ranked': median_ranked, 'log2foldchange': logfoldchange}
df = pd.DataFrame(data=d)
df.to_csv(f"{output_dirs}/pvalues.csv")
