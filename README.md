# binary-gene-expression-classification-from-histology

- create 2 class of gene expression (RNAseq) and label samples accordinly.
- use k-fold and labels to train and test a CNN on the histological images of the samples.
- get measurments for the ability of the CNN to diffrentiate between the classes.



This project create 2 class for a given gene, based on RNAseq counts files that have been normlized with DEseq2 (R package).
the labels from the classification are then used to create a label matrix with gene as rows and samples as columns.
A CNN classifies histological images from the same samples (RNAseq analisys) according to the labels in the label matrix for each gene.

Each sample is given a prediction score for the image for class 0 ("low").
This procces is done via k-fold method, in which, the samples are divided into k groups and then k folds are performed.
In each fold, the CNN trains on 3 of the groups and test on the fourth group.

Log2fold and p_value (t-test) are calculated from the prediction score of the samples from both classes.
ROC is also calculated from the prediction probabilitys.
