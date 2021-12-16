import math

def split_function_trios(main_list):
    for i in range(int(math.ceil(len(main_list)/3))):

        with open(f"/home/guysh/Documents/gene_lists_forcountsrun/6000_7500/6000_7500_genes_counts_{i}.csv", "w") as file:
            #file.write(str(main_list[i*5:(i+1)*5]))

            if i*3+2 <= len(main_list):
                file.write(main_list[i*3])
                file.write(main_list[i*3+1])
                file.write(main_list[i*3+2])
            else:
                file.write(main_list[i*3])

def split_function_pairs(main_list):
    for i in range(int(math.ceil(len(main_list)/2))):

        with open(f"/home/guysh/Documents/gene_lists_forcountsrun/6000_7500/6000_7500_genes_counts_{i}.csv", "w") as file:
            #file.write(str(main_list[i*5:(i+1)*5]))

            if i*2+1 <= len(main_list):
                file.write(main_list[i*2])
                file.write(main_list[i*2+1])
            else:
                file.write(main_list[i*2])


with open("/home/guysh/Documents/gene_lists_forcountsrun/6000_7500_genes_counts.csv", "r") as gene_list:

    main_list = gene_list.readlines()
    split_function_pairs(main_list)
