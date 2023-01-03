from CNN_train_and_test import *
from Labeling_functions import *
import os
import re
import sys
import statistics
import numpy as np
from plotnine import *
import pandas as pd
import shutil
from scipy import stats
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, roc_auc_score, balanced_accuracy_score
import torch
import time
import matplotlib.pyplot as plt
from PIL import Image
import cgitb
os.environ['KMP_DUPLICATE_LIB_OK']='True'
pd.options.mode.chained_assignment = None  # default='warn'



#choose parameters for the run
num_epochs = 3
batch_size = 16
num_workers = 2
num_workers_test = 2
# choose number of folds
k = 4


<<<<<<< HEAD
csv_location = "/home1/guyshani/predict_expression_counts/csv_files/"
image_location = "/home1/guyshani/predict_expression_counts/complete_dataset/"
temp_files = "/home1/guyshani/predict_expression_counts/temp_files/"
output_loc = "/home1/guyshani/predict_expression_counts/output_files/"
gene_files = "/home1/guyshani/predict_expression_counts/gene_files/"
=======
csv_location = "/home/maruvka/Documents/predict_expression/csv_files/"
image_location = "/home/maruvka/Documents/predict_expression/complete_dataset/"
temp_files = "/home/maruvka/Documents/predict_expression/temp_files/"
output_loc = "/home/maruvka/Documents/predict_expression/output_files/"
gene_files = "/home/maruvka/Documents/predict_expression/gene_files/"
>>>>>>> 28a0459 (added the new files from the lab computer (CAM_analysis and train_moco) no real updates on other files)
'''
csv_location = "/home/guysh/Documents/test_run/csvs/"
image_location = "/home/guysh/Documents/work_computer/image_recognition/images/crc/complete_dataset/"
temp_files = "/home/guysh/Documents/test_run/"
output_loc = "/home/guysh/Documents/test_run/"
'''
# name of the file that contains list of genes
<<<<<<< HEAD
gene_file = "3500_6000_genes_counts_"+str(sys.argv[1])
=======
gene_file = "remaining_genes_12_10__1.csv"
cuda_num = "cuda:1"
>>>>>>> 28a0459 (added the new files from the lab computer (CAM_analysis and train_moco) no real updates on other files)






<<<<<<< HEAD
def label_data_train_and_predict(gene, ensembl, k, batch_size, num_workers, num_epochs, device):

	'''
	parameters:
		- gene: gene name
		- ensembl: gene's ensembl
		- k: number of folds
		...

	returns:
		- patients_probs: a dataset containing patients class 0 probability predictions
		- tiles_probs: a dataset containing patients class 0 probability predictions

	'''


	dataset = f'CRC-DX_classes_{gene}'


	# create an expression file and a dataframe for the gene
	dataframe = expression_dataframe(gene)
	# fit a GM model and divide into classes accordinly
	try:
		dataframe = fit_GM(dataframe, gene, csv_location, output_files)
	except:
	# the expression value is zero for all the patients (therefor only one class)
		return "ERROR", "ERROR"

	# creates k csv files.
	cross_val(dataframe, k, image_location, csv_location, gene)


	tiles_probs = []
	patients_probs = []

	# train and test k models
	for j in range(k):

		print(f'fold number: {j}')
		# combine all training files
		all_filenames=[]
		for m in range(k-1):
			if m+j+1 < k:
				n = m+j+1
				all_filenames.append(csv_location+f"{gene}_crossval_group_{n}.csv")
			else:
				n = m+j+1-k
				all_filenames.append(csv_location+f"{gene}_crossval_group_{n}.csv")
		combined_csv = pd.concat([pd.read_csv(f, names = ["name", "label"]) for f in all_filenames ],  ignore_index=True)
		# export training files to csv
		combined_csv.to_csv(csv_location+f"{gene}_crossval_trainingset_{j}.csv", index=False, header = False)

		# define image transformations and load the data
		train_data_loader, train_data_size, image_transforms = load_data(f"{gene}_crossval_trainingset_{j}.csv", batch_size, num_workers, csv_location, image_location, device)

		# get class statistics and define weights for the loss function
		weight=stats_and_weights(f"{gene}_crossval_trainingset_{j}.csv", f"{gene}_crossval_group_{j}.csv", csv_location)

		# load neural network
		model, loss_func, optimizer = load_resnet(weight)

		#train the model
		trained_model = train_model(model, loss_func, optimizer, num_epochs, train_data_loader, train_data_size, image_transforms,dataset, device, output_files, j)

		# load model
		model = models.resnet18()
		model.fc = nn.Linear(512, 2)
		model.load_state_dict(torch.load(output_files+dataset+f'_model_{j}.pt'))

		# call patient classification function
		patient_results, tiles = model_evaluation(f"{gene}_crossval_group_{j}.csv", model, image_transforms, csv_location, temp_files, image_location, device, num_workers_test)

		# append results from all folds
		tiles_probs.append(tiles)
		patients_probs.append(patient_results)


	return patients_probs, tiles_probs
=======
def label_data_train_and_predict(gene, ensembl, k, batch_size, num_workers, num_epochs, device, cuda_num):

        '''
        parameters:
                - gene: gene name
                - ensembl: gene's ensembl
                - k: number of folds
                ...

        returns:
                - patients_probs: a dataset containing patients class 0 probability predictions
                - tiles_probs: a dataset containing patients class 0 probability predictions

        '''


        dataset = f'CRC-DX_classes_{gene}'


        # create an expression file and a dataframe for the gene
        dataframe = expression_dataframe(gene)
        # fit a GM model and divide into classes accordinly
        #try:
        dataframe = fit_GM(dataframe, gene, csv_location, output_files)
        #except:
        # the expression value is zero for all the patients (therefor only one class)
        #       return "ERROR", "ERROR"

        # creates k csv files.
        cross_val(dataframe, k, image_location, csv_location, gene)


        tiles_probs = []
        patients_probs = []

        # train and test k models
        for j in range(k):

                print(f'fold number: {j}')
                # combine all training files
                all_filenames=[]
                for m in range(k-1):
                        if m+j+1 < k:
                                n = m+j+1
                                all_filenames.append(csv_location+f"{gene}_crossval_group_{n}.csv")
                        else:
                                n = m+j+1-k
                                all_filenames.append(csv_location+f"{gene}_crossval_group_{n}.csv")
                combined_csv = pd.concat([pd.read_csv(f, names = ["name", "label"]) for f in all_filenames ],  ignore_index=True)
                # export training files to csv
                combined_csv.to_csv(csv_location+f"{gene}_crossval_trainingset_{j}.csv", index=False, header = False)

                # define image transformations and load the data
                train_data_loader, train_data_size, image_transforms = load_data(f"{gene}_crossval_trainingset_{j}.csv", batch_size, num_workers, csv_location, image_location, device, cuda_num)

                # get class statistics and define weights for the loss function
                weight=stats_and_weights(f"{gene}_crossval_trainingset_{j}.csv", f"{gene}_crossval_group_{j}.csv", csv_location)

                # check for error after the last function
                if weight == "err":
                        return "err", "err"

                # load neural network
                model, loss_func, optimizer = load_resnet(weight, device)

                #train the model
                trained_model = train_model(model, loss_func, optimizer, num_epochs, train_data_loader, train_data_size, image_transforms,dataset, device, output_files, j)

                # load model
                model = models.resnet18()
                model.fc = nn.Linear(512, 2)
                model.load_state_dict(torch.load(output_files+dataset+f'_model_{j}.pt'))

                # call patient classification function
                patient_results, tiles = model_evaluation(f"{gene}_crossval_group_{j}.csv", model, image_transforms, csv_location, temp_files, image_location, device, cuda_num, num_workers_test)

                # append results from all folds
                tiles_probs.append(tiles)
                patients_probs.append(patient_results)


        return patients_probs, tiles_probs
>>>>>>> 28a0459 (added the new files from the lab computer (CAM_analysis and train_moco) no real updates on other files)



"""

<<<<<<< HEAD
	$$$$$$$$$$$$$$$$$$$$ main chunck
=======
        $$$$$$$$$$$$$$$$$$$$ main chunck
>>>>>>> 28a0459 (added the new files from the lab computer (CAM_analysis and train_moco) no real updates on other files)

"""



cgitb.enable(format='text')
<<<<<<< HEAD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
=======
device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
>>>>>>> 28a0459 (added the new files from the lab computer (CAM_analysis and train_moco) no real updates on other files)
print(str(device))
print(time.ctime())


<<<<<<< HEAD
with open(gene_files+gene_file+".csv", "r") as gene_list:

	for gene_line in gene_list:
		if __name__ == '__main__':
			# enter gene name and ensembl
			start = time.time()
			gene = gene_line.split(',')[1].strip()
			ensembl = gene_line.split(',')[0].strip()

			# create a directory for output files
			os.mkdir(output_loc+f"output_{gene}/")
			output_files = output_loc+f"output_{gene}/"

			print("gene: "+gene+" "+ensembl)

			# get probabilitys per tile and per patient, with the ground truth label
			patients_probs, tiles_probs = label_data_train_and_predict(gene, ensembl, k, batch_size, num_workers, num_epochs, device)
			# if there was an error while creating classes
			if patients_probs == "ERROR":
				print("only one class")
				print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
				continue

			# create dataframes for model's patient level and tile level predictions
			patients_probs = pd.concat(patients_probs)
			tiles_probs = pd.concat(tiles_probs)
			# save a file with the model's predictions for the patient level and for the tile level
			patients_probs.to_csv(output_files+f"{gene}_patients_probs.csv", index=False, header = True)
			tiles_probs.to_csv(output_files+f"{gene}_tiles_probs.csv", index=False, header = True)

			end = time.time()
			print("analysis time for gene: "+ str(end-start))
			print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
=======
with open(gene_file, "r") as gene_list:

        for gene_line in gene_list:
                if __name__ == '__main__':
                        # enter gene name and ensembl
                        start = time.time()
                        gene = gene_line.split(',')[1].strip()
                        ensembl = gene_line.split(',')[0].strip()

                        # create a directory for output files
                        try:
                                os.mkdir(output_loc+f"output_{gene}/")
                        except FileExistsError:
                                print(f'directory for {gene} already exists')
                                continue

                        output_files = output_loc+f"output_{gene}/"

                        print("gene: "+gene+" "+ensembl)

                        # get probabilitys per tile and per patient, with the ground truth label
                        patients_probs, tiles_probs = label_data_train_and_predict(gene, ensembl, k, batch_size, num_workers, num_epochs, device, cuda_num)
                        # if there was an error while creating classes
                        if patients_probs == "ERROR":
                                print("only one class")
                                print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE          @@@@@@@@@@@@@@@@@@@@@@")
                                continue
                        elif patients_probs == "err":
                                print("low expression")
                                print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE          @@@@@@@@@@@@@@@@@@@@@@")
                                continue

                        # create dataframes for model's patient level and tile level predictions
                        patients_probs = pd.concat(patients_probs)
                        tiles_probs = pd.concat(tiles_probs)
                        # save a file with the model's predictions for the patient level and for the tile level
                        patients_probs.to_csv(output_files+f"{gene}_patients_probs.csv", index=False, header = True)
                        tiles_probs.to_csv(output_files+f"{gene}_tiles_probs.csv", index=False, header = True)

                        end = time.time()
                        print("analysis time for gene: "+ str(end-start))
                        print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE          @@@@@@@@@@@@@@@@@@@@@@")
>>>>>>> 28a0459 (added the new files from the lab computer (CAM_analysis and train_moco) no real updates on other files)
