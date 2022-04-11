import math

input_dir = "/home/guysh/Documents/gene_lists_forcountsrun/shuffled_run/5000_6000/"
output_dir = "/home/guysh/Documents/gene_lists_forcountsrun/shuffled_run/5000_6000/"
file_name = "5000_6000"

def split_function_trios(main_list):
    for i in range(int(math.ceil(len(main_list)/3))):

        with open(output_dir + f"{file_name}_{i}.csv", "w") as file:
            #file.write(str(main_list[i*5:(i+1)*5]))

            if i*3+2 <= len(main_list):
                file.write(main_list[i*3])
                file.write(main_list[i*3+1])
                file.write(main_list[i*3+2])
            else:
                file.write(main_list[i*3])

def split_function_pairs(main_list):
    for i in range(int(math.ceil(len(main_list)/2))):

        with open(output_dir + f"{file_name}_{i}.csv", "w") as file:
            #file.write(str(main_list[i*5:(i+1)*5]))

            if i*2+1 <= len(main_list):
                file.write(main_list[i*2])
                file.write(main_list[i*2+1])
            else:
                file.write(main_list[i*2])


with open(input_dir + file_name + ".csv", "r") as gene_list:

    main_list = gene_list.readlines()
    split_function_pairs(main_list)
