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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cgitb


#choose parameters for the run
num_epochs = 3
batch_size = 16
num_workers = 2
# choose number of folds
k = 4


csv_location = "/home/guyshani/predict_expression_counts/csv_files/"
image_location = "/home/guyshani/predict_expression_counts/complete_dataset/"
temp_files = "/home/guyshani/predict_expression_counts/temp_files/"
output_loc = "/home/guyshani/predict_expression_counts/output_files/"
'''
csv_location = "/home/guysh/Documents/test_run/csvs/"
image_location = "/home/guysh/Documents/work_computer/image_recognition/images/crc/complete_dataset/"
temp_files = "/home/guysh/Documents/test_run/"
output_loc = "/home/guysh/Documents/test_run/"
'''
# name of the file that contains list of genes
gene_file = "3500_6000_genes_counts_"+str(sys.argv[1])






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
		train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms = load_data(f"{gene}_crossval_trainingset_{j}.csv", f"{gene}_crossval_group_{j}.csv", batch_size, num_workers, csv_location, image_location, device)

		# get class statistics and define weights for the loss function
		weight=stats_and_weights(f"{gene}_crossval_trainingset_{j}.csv", f"{gene}_crossval_group_{j}.csv", csv_location)

		# load neural network
		model, loss_func, optimizer, scheduler = load_resnet(weight)

		#train the model
		trained_model, best_epoch = train_and_validate(model, loss_func, optimizer, scheduler, num_epochs, train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms,dataset, device, output_files, j)

		# load the best model
		model = models.resnet18()
		model.fc = nn.Linear(512, 2)
		model.load_state_dict(torch.load(output_files+dataset+'_model_'+str(best_epoch)+f'_{j}_fold.pt'))

		# delete all models except the best one
		for l in range(num_epochs):
			if l != best_epoch:
				try:
					os.remove(output_files+dataset+'_model_'+str(l)+f'_{j}_fold.pt')
				except:
					continue

		# call patient classification function
		patient_results, low_tiles, high_tiles = model_evaluation(f"{gene}_crossval_group_{j}.csv", model, image_transforms, csv_location, temp_files, image_location, device)

		low = pd.DataFrame({'low_prob': low_tiles})
		low = low.assign(label = 'low')
		high = pd.DataFrame({'low_prob': high_tiles})
		high = high.assign(label = 'high')
		tiles = pd.concat([low,high], ignore_index=True, axis=0)



		tiles_probs.append(tiles)
		patients_probs.append(patient_results)


	return patients_probs, tiles_probs








"""

	$$$$$$$$$$$$$$$$$$$$ main chunck

"""



cgitb.enable(format='text')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(str(device))
print(time.ctime())

with open("pvalues_"+gene_file+".txt", 'w') as final_file:

	final_file.write("ensembl,gene,tiles ttest,patients ttest average,patients ttest median,log 2 fold change"+"\n")

	with open("/home/guyshani/predict_expression_counts/gene_files/"+gene_file+".csv", "r") as gene_list:

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
				if patients_probs == "ERROR":
					print("only one class")
					print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
					continue

				patients_probs = pd.concat(patients_probs)
				tiles_probs = pd.concat(tiles_probs)

				patients_probs.to_csv(output_loc+f"{gene}_patients_probs.csv", index=False, header = True)


				# t test for tiles
				low = tiles_probs['low_prob'].loc[tiles_probs.query(f'(label == "low")').index.values].values
				high = tiles_probs['low_prob'].loc[tiles_probs.query(f'(label == "high")').index.values].values
				tiles_ttest = stats.ttest_ind(low, high, equal_var=False)[1]

				# t test for patients median
				low = patients_probs['median'].loc[patients_probs.query(f'(label == "low")').index.values].values
				high = patients_probs['median'].loc[patients_probs.query(f'(label == "high")').index.values].values
				patients_ttest_median = stats.ttest_ind(low, high, equal_var=False)[1]

				# t test for patients average
				low = patients_probs['average'].loc[patients_probs.query(f'(label == "low")').index.values].values
				high = patients_probs['average'].loc[patients_probs.query(f'(label == "high")').index.values].values
				patients_ttest_average = stats.ttest_ind(low, high, equal_var=False)[1]

				# claculate log fold change with average probs valuees of patients using average probs values of tiles for each patient
				avg_high = sum(high)/len(high)
				avg_low = sum(low)/len(low)
				logfoldchange = (avg_high/avg_low-1)
				'''
				p1 = (ggplot(patients_probs)
					+geom_point(aes(x = 'label', y = 'median'), alpha = 0.5)
					+geom_boxplot(aes(x = 'label', y = 'median'), alpha = 0.5)
					+labs(title = f'{gene}  median class 0 probability, t-test: {patients_ttest_median}'))
				'''
				p2 = (ggplot(patients_probs)
					+geom_point(aes(x = 'label', y = 'average'), alpha = 0.5)
					+geom_boxplot(aes(x = 'label', y = 'average'), alpha = 0.5)
					+labs(title = f'{gene}  average class 0 probability t-test: {patients_ttest_average}'))

				p3 = (ggplot(tiles_probs)
					+geom_point(aes(x = 'label', y = 'low_prob'), alpha = 0.5)
					+geom_boxplot(aes(x = 'label', y = 'low_prob'), alpha = 0.5)
					+labs(title = f'{gene} class 0 probabilitys tiles, t-test: {tiles_ttest}'))

				#ggsave(plot = p1, filename = f"{gene}_median_class_0_probability.png", path = output_files)
				ggsave(plot = p2, filename = f"{gene}_average_class_0_probability.pdf", path = output_files)
				ggsave(plot = p3, filename = f"{gene}_class_0_probability_tiles.pdf", path = output_files)

				final_file.write(ensembl+","+gene+","+str(tiles_ttest)+","+str(patients_ttest_average)+","+str(patients_ttest_median)+","+str(logfoldchange)+"\n")

				print("t-test results:")
				print("tiles t-test: "+str(tiles_ttest))
				print("patients median t-test: "+str(patients_ttest_median))
				print("patients average t-test: "+str(patients_ttest_average))
				print("log2 fold change: "+str(logfoldchange))
				end = time.time()
				print("analysis time for gene: "+ str(end-start))
				print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
