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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, roc_auc_score, balanced_accuracy_score
from matplotlib import pyplot as plt
import torch
import time
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
from customdatasets import CATEGORICAL
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cgitb


'''
def expression_file(gene):
    # create an expression file         ***FPKM-UQ

    # this part is going to create a id_list_{gene},csv that will contain the FPKM name, TCGA name and the expression for that gene,
    # for that patient

    # id_list.csv - a filr that containes the name of the FPKM file on the first column,
    # and the patient tag (TCGA-3456-01A) on the second column
    with open(FPKM_files+"id_list_FPKM_UQ_CRCdataset.csv", "r") as id_lst:
        with open(csv_location+f"id_list_UQ_{gene}.csv", "w") as new_lst:
            for name in id_lst:
                filename = name.split(',')[1]
                tcga = name.split(',')[0]
                filename = filename.strip()
                fpkm = open(FPKM_files+f"{filename}", "r")
                for line in fpkm:
                    if re.match(f'{ensembl}', line):
                        line = line.split("\t")[1]
                        line = line.strip()
                        new_lst.write(tcga + ',' + line + '\n')
    fpkm.close()
    df = pd.read_csv(csv_location+f'id_list_UQ_{gene}.csv', header =None)
    df.columns=['tcga_name','expression']

    return df
'''

def expression_dataframe(gene):

    # create a dataframe df with TCGA names and expression for the gene
    
    #data = pd.read_csv("/home/guysh/Downloads/counts/countmatrix_normalized_10000_symbols.csv")
    data = pd.read_csv("countmatrix_normalized_10000_symbols.csv")
    df = data.loc[data['symbols'] == gene]
    df=df.drop(["symbols","Row.names"], axis = 1).T
    df = df.reset_index()
    df.columns = ["tcga_name", "expression"]
    df = df.drop(index = 0)
    df = df.reset_index(drop = True)

    return df

def fit_GM(df, gene):
    GM = GaussianMixture(n_components=2).fit(df[['expression']])
    #GM.means_
    cluster = GM.predict(df[['expression']])
    params=GM.get_params()

    df["cluster"] = cluster
    #df["expression"]
    if np.mean(df['expression'].loc[df.query(f'(cluster == "0")').index.values].values) > np.mean(df['expression'].loc[df.query(f'(cluster == "1")').index.values].values):
        df['cluster'] = df['cluster'].replace(1,'low')
        df['cluster'] = df['cluster'].replace(0,'high')
    else:
        df['cluster'] = df['cluster'].replace(0,'low')
        df['cluster'] = df['cluster'].replace(1,'high')

    # checks if one of the groups is not too small
    if len(np.where(df['cluster'] == 'high')[0]) < 10 or len(np.where(df['cluster'] == 'low')[0]) <10:

        # if one of the classes have less then 10 patients then split the class by the average expression

        mean = statistics.mean(df['expression'])
        df["cluster"].loc[df.query(f'(expression <= {mean})').index.values] = 'low'
        df["cluster"].loc[df.query(f'(expression > {mean})').index.values] = 'high'

        print("^^^^^^^" +gene+"  is using average for class split")



    #seperation between the groups (1 is best and -1 is worse)
    silhouette_score(df[['expression']], cluster)
    df.to_csv(csv_location+f"{gene}_crossval.csv", columns= ["tcga_name", "cluster"],header=False, index=False)

    p = (
    ggplot(df)
    +geom_freqpoly(aes(x = 'expression', group = 'cluster', color = 'cluster'))
    +geom_histogram(aes(x = 'expression', fill = 'cluster'), alpha = 0.5)
    +labs(title = f'{gene}')
    )
    ggsave(plot = p, filename = f"{gene}_crossval.png", path = output_files)
    
    return df


def cross_val(df, k, image_location):
    
    '''
    this function devides the dataset into k group, stratified by class.
    it saves a csv with image name and class for every group.
    '''
    

    df_temp = df
    df["dataset"] = ''
    image_lst = os.listdir(image_location)

    # split into datasets and add the dataset label in dataset column
    for label in ['high', 'low']:
        # count the number of samples in each class
        tot = len(df_temp.query('(cluster == @label)'))
        
        for i in range(k):
            group = np.random.choice(df_temp.query('(cluster == @label)').index.values, int(tot/k))
            #print("group "+str(i)+"  "+str(group))
            df["dataset"].loc[group] = i
            df_temp = df_temp.drop(group)

            
    for i in range(k):
        with open(csv_location+f"{gene}_crossval_group_{i}.csv", "w") as cgroup:

            for j in range(len(df)):
                tcga = df["tcga_name"].loc[j]
                label = df["cluster"].loc[j]
                dataset = df["dataset"].loc[j]
                for image in image_lst:
                    if tcga == image[17:29]:
                        if dataset == i:
                            cgroup.write(image+","+label+"\n")


def stats_and_weights(train_csv, test_csv):
    image_num = 0
    train = []
    test = []
    for data in [train_csv, test_csv]:
        # create a list of class labels for train set and for test set
        with open(csv_location+f"{data}", "r") as datafile:
            for line in datafile:
                image_num += 1
                if data == train_csv:
                    train.append(line.split(",")[1].strip())
                elif data == test_csv:
                    test.append(line.split(",")[1].strip())

    print("total image number in train: "+str(len(train)))
    print("low: "+str(train.count("low")))
    print("high: "+str(train.count("high")))
    print("total image number in test: "+str(len(test)))
    print("low: "+str(test.count("low")))
    print("high: "+str(test.count("high")))


    # set the weight ration for the loss function

    try:
        weight = test.count("low")/test.count("high")
        print("wieght (low/high): "+str(weight))
    except ZeroDivisionError:
        print(ZeroDivisionError)
    
    return weight


def load_data(train_csv, test_csv, bs, num_workers):
    
    # a function that set the transformations and loads the training and testing data.
    
    history = []

    image_transforms = {
        'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(p=0.1),
        #transforms.RandomPerspective(distortion_scale=0.2, p=0.1,),
        #transforms.RandomAffine(120),
        #transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Load Data from folders
    data = {
        'train': CATEGORICAL(csv_file = csv_location+f"{train_csv}", rootdir = image_location ,transform=image_transforms['train']),
        'valid': CATEGORICAL(csv_file = csv_location+f"{test_csv}", rootdir = image_location ,transform=image_transforms['valid'])
    }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True, num_workers=num_workers)
    
    return train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms


def load_resnet(weight):
    resnet18 = models.resnet18(pretrained = True)

    # True - adjust all the parameters. False - feature extracting, only compute gradients for newly initialized layer.
    for param in resnet18.parameters():
        param.requires_grad = False

    resnet18.fc = nn.Linear(512, 2)

    for param in resnet18.layer4.parameters():
        param.requires_grad = True

    #loss_func = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss(weight = torch.FloatTensor([1,weight]).cuda())

    #the model parameters are registered here in the optimizer (change learning rate)
    optimizer = optim.Adam(resnet18.layer4.parameters(), lr = 1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 8, threshold = 0.2)
    
    return resnet18, loss_func, optimizer, scheduler



def train_and_validate(model, loss_func, optimizer, scheduler, epochs, train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms, dataset):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    best_valid_loss = 0.0
    history = []
    t_predictions = []
    t_labels = []
    v_predictions = []
    v_labels = []
    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the modelff
            outputs = model(inputs)

            # Compute loss
            loss = loss_func(outputs, labels)

            # Backpropagate the gradients (backwards pass, calculate gradients for each parameter)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            normal = torch.nn.functional.softmax(outputs.data, dim=1)

            # Compute the accuracy
            ret, predictions = torch.max(normal, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            # append predictions and labels for later claculations
            t_predictions = np.append(t_predictions ,np.array(predictions.cpu()), axis =0)
            t_labels = np.append(t_labels ,labels.data.view_as(predictions).cpu(), axis =0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_func(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                normal = torch.nn.functional.softmax(outputs.data, dim=1)
                ret, predictions = torch.max(normal, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                
                # append predictions and labels for later claculations
                v_predictions = np.append(v_predictions ,np.array(predictions.cpu()), axis =0)
                v_labels = np.append(v_labels ,labels.data.view_as(predictions).cpu(), axis =0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        
        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        
        # find accuracys per class
        valid_roc = roc_auc_score(v_labels, v_predictions)
        balanced_accuracy = balanced_accuracy_score(v_labels, v_predictions)
        
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, valid_roc])
        # update the learning rate based on validation/epoch
        scheduler.step(avg_valid_loss)

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("learning rate: "+str(optimizer.param_groups[0]['lr']))
        print("Validation: "+"\n"+"ROC: "+str(valid_roc))
        
        # save the best model
        if epoch ==0:
            best_valid = valid_roc
            best_epoch = epoch
            torch.save(model.state_dict(), output_files+ dataset+'_model_'+str(epoch)+'.pt')
        if valid_roc > best_valid:
            best_valid = valid_roc
            best_epoch = epoch
            print('best validation AUC ROC so far')
            torch.save(model.state_dict(), output_files+dataset+'_model_'+str(epoch)+'.pt')
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
    
    #save the training history
    torch.save(history,output_files+dataset+'_history.pt')
    
    # get statistics
    
    
    return model, best_epoch


def model_evaluation(csv_file, model, image_transforms):
    
    batch_s = 100
    # a list with the median class 0 probabilities for each patient
    classzero_median = []
    classzero_average = []
    # a list with the label for each patient
    patient_labels = []
    patient_names = []
    # lists for the tiles from each class
    high_tiles = np.array([])
    low_tiles = np.array([])
    
    df = pd.read_csv(csv_location+csv_file, header =None)
    df.columns=['tcga_name','label']
    # create a list of patients names
    patient_list = df["tcga_name"].str[17:29].unique()
    
    
    for patient in patient_list:
        
        full_output = np.empty(shape=[0,2])
        # create a csv file containing the tiles and labels of the patient (in order to load each patient to the model)
        df[df["tcga_name"].str.contains(patient)].to_csv(temp_files+f"{patient}.csv", columns= ["tcga_name", "label"],header=False, index=False)
        
        # load the tiles of the patient
        load = CATEGORICAL(csv_file = temp_files+f"{patient}.csv", rootdir = image_location ,transform=image_transforms['test'])
        valid_data_loader = DataLoader(load, batch_size=batch_s, shuffle=False)
        
        # pass all tiles from each patient
        with torch.no_grad():
            model.eval()
            model.cuda()
            for (inputs, labels) in valid_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                
                # apply softmax to the output, in order to get probabilities
                normal = torch.nn.functional.softmax(outputs.data, dim=1)
                normal = np.array(normal.cpu())
                full_output = np.concatenate((full_output, normal), axis=0)
        
                
        # take the probabilities for class 0, and calculate the median.      
        classzero = full_output[:,0]
        classzero_median.append(np.median(classzero))
        classzero_average.append(np.average(classzero))
        
        # get patient label
        patient_labels.append(df['label'].iloc[df[df["tcga_name"].str.contains(patient)].index[0]])
        patient_names.append(patient)
        if df['label'].iloc[df[df["tcga_name"].str.contains(patient)].index[0]] == "high":
            high_tiles = np.concatenate((high_tiles, classzero), axis=0)
        else:
            low_tiles = np.concatenate((low_tiles, classzero), axis=0)

    # create a dataframe with the label and median of each patient
    d = {'patient': patient_names, 'labels': patient_labels, 'median': classzero_median, 'average': classzero_average}
    dataf = pd.DataFrame(data=d)
    dataf.columns=['tcga_name','label','median', 'average']
    
    return dataf, low_tiles, high_tiles






def main_function(gene, ensembl, k, batch_size, num_workers, num_epochs, device):
        

    dataset = f'CRC-DX_classes_{gene}'
        

    # create an expression file and a dataframe for the gene
    dataframe = expression_dataframe(gene)
    # fit a GM model and divide into classes accordinly 
    dataframe = fit_GM(dataframe, gene)

    cross_val(dataframe, k, image_location) # creates k csv files.


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
        train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms = load_data(f"{gene}_crossval_trainingset_{j}.csv", f"{gene}_crossval_group_{j}.csv", batch_size, num_workers)
        # get class statistics and define weights for the loss function 

        weight=stats_and_weights(f"{gene}_crossval_trainingset_{j}.csv", f"{gene}_crossval_group_{j}.csv")
        model, loss_func, optimizer, scheduler = load_resnet(weight)
        
        #train the model

        if __name__ == '__main__':
            trained_model, best_epoch = train_and_validate(model, loss_func, optimizer, scheduler, num_epochs, train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms,dataset)
            
        # load the best model
        model = models.resnet18()
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load(output_files+dataset+'_model_'+str(best_epoch)+'.pt'))
        
        # call patient classification function
        patient_results, low_tiles, high_tiles = model_evaluation(f"{gene}_crossval_group_{j}.csv", model, image_transforms)
        
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
csv_location = "/home/guysh/Documents/counts_run/csv_files/"
image_location = "/home/guysh/Documents/work_computer/image_recognition/images/crc/complete_dataset/"
temp_files = "/home/guysh/Documents/counts_run/temp_files/"
output_loc = "/home/guysh/Documents/counts_run/output_files/"
'''

gene_file = "2500_3000_genes_counts_"+str(sys.argv[1])

with open("pvalues_"+gene_file+".txt", 'w') as final_file:

    final_file.write("ensembl,gene,tiles_ttest,patients_ttest_average,patients_ttest_median"+"\n")

    with open("/home/guyshani/predict_expression_counts/gene_files/"+gene_file+".csv", "r") as gene_list:
        
        for gene_line in gene_list:

            # enter gene name and ensembl
            gene = gene_line.split(',')[1].strip()
            ensembl = gene_line.split(',')[0].strip()

            # create a directory for output files
            os.mkdir(output_loc+f"output_{gene}/")
            output_files = output_loc+f"output_{gene}/"
            
            print("gene: "+gene+" "+ensembl)

            # get probabilitys per tile and per patient, with the ground truth label
            patients_probs, tiles_probs = main_function(gene, ensembl, k, batch_size, num_workers, num_epochs, device)
            patients_probs = pd.concat(patients_probs)
            tiles_probs = pd.concat(tiles_probs)
            print("patients_probs")
            print(patients_probs)
            print("tiles_probs: ")
            print(tiles_probs)
            
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
            
            
            p1 = (ggplot(patients_probs)
                +geom_point(aes(x = 'label', y = 'median'), alpha = 0.5)
                +geom_boxplot(aes(x = 'label', y = 'median'), alpha = 0.5)
                +labs(title = f'{gene}  median class 0 probability, t-test: {patients_ttest_median}'))
        
            p2 = (ggplot(patients_probs)
                +geom_point(aes(x = 'label', y = 'average'), alpha = 0.5)
                +geom_boxplot(aes(x = 'label', y = 'average'), alpha = 0.5)
                +labs(title = f'{gene}  average class 0 probability t-test: {patients_ttest_average}'))
        
            p3 = (ggplot(tiles_probs)
                +geom_point(aes(x = 'label', y = 'low_prob'), alpha = 0.5)
                +geom_boxplot(aes(x = 'label', y = 'low_prob'), alpha = 0.5)
                +labs(title = f'{gene} class 0 probabilitys tiles, t-test: {tiles_ttest}'))
            
            ggsave(plot = p1, filename = f"{gene}_median_class_0_probability.png", path = output_files)
            ggsave(plot = p2, filename = f"{gene}_average_class_0_probability.png", path = output_files)
            ggsave(plot = p3, filename = f"{gene}_class_0_probability_tiles.png", path = output_files)

            final_file.write(ensembl+","+gene+","+str(tiles_ttest)+","+str(patients_ttest_average)+","+str(patients_ttest_median)+"\n")

            print("t-test results:")
            print("tiles t-test: "+str(tiles_ttest))
            print("patients median t-test: "+str(patients_ttest_median))
            print("patients average t-test: "+str(patients_ttest_average))
            print("@@@@@@@@@@@@@@@@@@@  END OF CURRENT GENE  @@@@@@@@@@@@@@@@@@@@@@")
