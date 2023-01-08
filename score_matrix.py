import pandas as pd
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn
import matplotlib.pylab as plt










def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None):
    r"""Compute Cohen's kappa: a statistic that measures inter-annotator agreement.
    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as
    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)
    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.
    Read more in the :ref:`User Guide <cohen_kappa>`.
    Parameters
    ----------
    y1 : array of shape (n_samples,)
        Labels assigned by the first annotator.
    y2 : array of shape (n_samples,)
        Labels assigned by the second annotator. The kappa statistic is
        symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.
    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to select a
        subset of labels. If `None`, all labels that appear at least once in
        ``y1`` or ``y2`` are used.
    weights : {'linear', 'quadratic'}, default=None
        Weighting type to calculate the score. `None` means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.
    References
    ----------
    .. [1] :doi:`J. Cohen (1960). "A coefficient of agreement for nominal scales".
           Educational and Psychological Measurement 20(1):37-46.
           <10.1177/001316446002000104>`
    .. [2] `R. Artstein and M. Poesio (2008). "Inter-coder agreement for
           computational linguistics". Computational Linguistics 34(4):555-596
           <https://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-034-R2>`_.
    .. [3] `Wikipedia entry for the Cohen's kappa
            <https://en.wikipedia.org/wiki/Cohen%27s_kappa>`_.
    """
    confusion = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k








#output_dirs = "/storage/bfe_maruvka/guyshani/DGX_files/output_files_3000_4000"
output_dirs = "/home/guysh/Documents/output_files/"
outputdirs = os.listdir(output_dirs)


df = pd.read_csv("~/Documents/output_files/output_STK11/STK11_patients_probs.csv")
df = df.sort_values(by=['tcga_name'])
df2 = df[['tcga_name']].copy()
df2.reset_index(drop=True)
gene = "STK11"


for dir in outputdirs:

    # get gene name
    gene = dir.split("_")[1]
    # load the patient_probs.csv file into a pandas dataframe
    
    print(f"{output_dirs}/{dir}/{gene}_patients_probs.csv")
    # load dataframe with scores for each sample
    df = pd.read_csv(f"{output_dirs}/{dir}/{gene}_patients_probs.csv")
    df = df.sort_values(by=['tcga_name'])


    # probabilitys per sample by average
    model_probs = np.array(df["average"])

    # get ground truth
    t_labels = df["label"]
    # replace "low" with 0 and "high" with 1
    t_labels = t_labels.replace("low", 0)
    t_labels = t_labels.replace("high", 1)

    # translate probabilitys to predicted class (0 or 1)
    new_list = [0 if model_probs[j] > 0.5 else 1 for j in list(range(len(model_probs)))]
    # compare predictions with ground truth
    predictions = [1 if new_list[j] == t_labels[j] else 0 for j in list(range(len(new_list)))]

    df2[f"{gene}"] = predictions


cohen_kappa_score(df2["MLH1"], df2["STK11"])
cohen_kappa_score(df2["MLH1"], df2["TMEM160"])
cohen_kappa_score(df2["TMEM160"], df2["STK11"])

np.corrcoef(df2["MLH1"], df2["STK11"])
np.corrcoef(df2["TMEM160"], df2["STK11"])
# calculate correlation matrix (using cohen kappa function)
corr_matrix = df2.corr(method = cohen_kappa_score)
# create a heatplot
ax = seaborn.heatmap(corr_matrix, linewidth=0.5, annot=True)
plt.show()
# create a cluster map
seaborn.clustermap(corr_matrix, metric="correlation", method="single", cmap="Blues", standard_scale=1)
plt.show()
