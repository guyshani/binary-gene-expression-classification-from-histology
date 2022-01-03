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


file_number = 67
#file_path = "/home/guysh/Documents/servers_data/dgx/oncogenes/"
#gene_files = "/home/guysh/Documents/gene_lists_forcountsrun/oncogenes/"

# location of output files
file_path = "/home/guysh/Documents/servers_data/dgx/artificiale_data/output/"
# location of the gene list files
gene_files = "/home/guysh/Documents/predict_gene_expression_from_histology/artificiale_data/gene_files/"


#output_name = "slurm-13376_"
#gene_name = "oncogenes_genes_counts_"

output_name = "slurm-13081_"
gene_name = "3500_6000_genes_counts_"


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

            # skip if file does not exist
            if not os.path.exists(file_path+output_name+str(i)+".out"):
                print("file not found")
                continue
            
            with open(file_path+output_name+str(i)+".out", "r") as output:


                # split the file by genes
                file = output.read().split("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
                for gene in file:

                    roc_prob_list = []
                    low_vs_high_list = []

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
                        missing_genes.write(re.findall(r'ENSG[0-9]+.[0-9]*', gene)[0]+","+str(i)+","+"one class"+"\n")
                        continue

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

                    p_val = str(re.findall(r'patients\saverage\st-test:\s[0-9]+\.[0-9]+e*-*[0-9]*', gene.split("fold number: ")[4])[0].split(" ")[3])
                    logfold = str("{:.5f}".format(float(re.findall(r'log2\sfold\schange:\s-*[0-9]+\.[0-9]+', gene.split("fold number: ")[4])[0].split(" ")[3])))

                    lh = low_vs_high_list[0]+"-"+low_vs_high_list[1]+"-"+low_vs_high_list[2]+"-"+low_vs_high_list[3]
                    roc = roc_prob_list[0]+"-"+roc_prob_list[1]+"-"+roc_prob_list[2]+"-"+roc_prob_list[3]

                    # write gene stats to summary file
                    summary.write(gene_symbol+","+p_val+","+logfold+","+patients_low+","+patients_high+","+lh+","+roc+"\n")
