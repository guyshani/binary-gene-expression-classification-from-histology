'''

a script that takes slurm output files (from the technion DGX server) and the files with the gene names that Run.py is taking as an input.

output:
    summary.csv with the following columns: gene symbol, p-value, log2fold, patients low number, patients high number, low/high patchs, roc probs
    - summarize al the important parameters from the output files of Run.py

    missing_genes.csv with the following columns: ensembl, gene file, status
    - list of all the genes that didnt have output and the reason why (the algorithem was not able to create 2 classes / the run stopped in the middle)
        adds the number of the gene names file that the missing gene is from.

'''

import os
import re
import pandas as pd

a =0
file_number = 149
file_path = "/home/guysh/Documents/servers_data/dgx/oncogenes/"
gene_files = "/home/guysh/Documents/gene_lists_forcountsrun/oncogenes/"

pval_name = "pvalues_oncogenes_genes_counts_"
output_name = "slurm-13376_"
gene_name = "oncogenes_genes_counts_"


with open(file_path+"missing genes.csv", "w") as missing_genes:
    with open(file_path + "summary.csv","w") as summary:

        missing_genes.write("ensembl,gene file,status"+"\n")
        summary.write("gene symbol,p-value,log2fold,patients low number,patients high number,low/high patchs,roc probs"+"\n")

        for i in list(range(file_number+1)):

            # creaate a dataframe from the gene name file
            names = pd.read_csv(gene_files+gene_name+str(i)+".csv", header = None)
            analyzed_genes = []

            counter = 0
            fold_num = 0
            roc_prob_list = []
            low_vs_high_list = []


            with open(file_path+output_name+str(i)+".out", "r") as output:

                # split the file by genes
                file = output.read().split("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
                for gene in file:

                    # did the run stopped in the middle
                    if re.findall(r'slurmstepd: error:', gene) != []:
                        missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"run finished"+"\n")
                        analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])
                        continue
                    # reached the end of list (file)
                    if re.findall(r'gene:', gene) == []:
                        continue

                    # if there is a gene that didnt have 2 classes
                    if "only one class" in gene:
                        if len(gene.split("only one class")) == 2 and gene.split("only one class")[1] == "\n":
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"one class"+"\n")
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])
                            break
                        elif len(gene.split("only one class")) == 2 and gene.split("only one class")[1] != "\n":
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"one class"+"\n")
                            gene_symbol = re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1]
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1])
                        # if there are 2 genes that didnt have 2 classes
                        elif len(gene.split("only one class")) == 3 and gene.split("only one class")[2] == "\n":
                            # add the one class gene to the missing genes list
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"one class"+"\n")
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1]+","+str(i)+","+"one class"+"\n")
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1])
                            break
                        elif len(gene.split("only one class")) == 3:
                            # add the one class gene to the missing genes list
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"one class"+"\n")
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1]+","+str(i)+","+"one class"+"\n")
                            gene_symbol = re.findall(r'ENSG[0-9]+.[0-9]*', gene)[2]
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1])
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[2])
                        else:
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"one class"+"\n")
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1]+","+str(i)+","+"one class"+"\n")
                            missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[2]+","+str(i)+","+"one class"+"\n")
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[1])
                            analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[2])
                            break

                    # only one gene appears in this list item (gene)
                    else:
                        gene_symbol = re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]
                        analyzed_genes.append(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0])

                    patients_low = str("{:.3f}".format(float(re.findall(r'number\sof\spatients\sin\s"high"\sgroup:\s[0-9]+', gene)[0].split(" ")[6])))
                    patients_high = str("{:.3f}".format(float(re.findall(r'number\sof\spatients\sin\s"low"\sgroup:\s[0-9]+', gene)[0].split(" ")[6])))

                    for fold in list(range(1,5)):

                        roc_prob = re.findall(r'ROC\sprobs:\s[0-9]+\.[0-9]+', gene.split("fold number: ")[fold])
                        roc_prob = str("{:.3f}".format(float((max([roc_prob[l].split(" ")[2] for l in range(0,3)])))))
                        roc_prob_list.append(roc_prob)
                        # ratio of low/high patches
                        low_vs_high = str("{:.3f}".format(float(re.findall(r'wieght\s\(low\/high\):\s[0-9]+\.[0-9]+', gene.split("fold number: ")[fold])[0].split(" ")[2])))
                        low_vs_high_list.append(low_vs_high)

                    p_val = str(re.findall(r'patients\saverage\st-test:\s[0-9]+\.[0-9]+', gene.split("fold number: ")[4])[0].split(" ")[3])
                    logfold = str("{:.5f}".format(float(re.findall(r'log2\sfold\schange:\s-*[0-9]+\.[0-9]+', gene.split("fold number: ")[4])[0].split(" ")[3])))

                    lh = low_vs_high_list[0]+"-"+low_vs_high_list[1]+"-"+low_vs_high_list[2]+"-"+low_vs_high_list[3]
                    roc = roc_prob_list[0]+"-"+roc_prob_list[1]+"-"+roc_prob_list[2]+"-"+roc_prob_list[3]

                    # write gene stats to summary file
                    summary.write(gene_symbol+","+p_val+","+logfold+","+patients_low+","+patients_high+","+lh+","+roc+"\n")


                        #print(any(name[0].str.contains('ENSG00000144476.5', regex=False)))
'''
    for testing if some genes are missing from summary.csv and missing_genes.csv

            a+=len(set(analyzed_genes))

            for k in names[0]:
                if str(k) not in analyzed_genes:
                    print("k = "+k+"    "+str(i))
        print(a)
'''









'''
- open pvalues and approproate gene list
- check if genes are missing in the pvalues list, if so, add them to the misssing genes list
- get average pvalues and logfoldchange
- get average roc probs (average of the rocs from the k models used for testing)
- get high/low ratio
'''
