import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from customdatasets import CATEGORICAL


def cross_val(df, k, image_location, csv_location, gene):

    '''
    this function devides the dataset into k group, stratified by class.
    it saves a list of image name and class for every group in a csv format.
    '''
    # create a copy of the label matrix
    df_temp = df.copy()
    print("lnumber of patients (cross_val): "+str(len(df)))
    df["dataset"] = ''
    # create a list of all image names
    image_lst = os.listdir(image_location)

    # split into groups (folds) and add a dataset column with coresponding indecies
    for label in ['high', 'low']:
        # count the number of samples in each class
        tot = len(df_temp.query('(cluster == @label)'))

        for i in range(k):
            # randomly pick samples for a group (size = tot/numbert of folds)
            group = np.random.choice(df_temp.query('(cluster == @label)').index.values, int(tot/k), replace = False)
            # tag the group samples with an index
            df["dataset"].iloc[group] = i
            # drop the group samples from the temporary df
            df_temp = df_temp.drop(group)

    # write a csv file for each group with sample names and label
    for i in range(k):
        with open(csv_location+f"{gene}_crossval_group_{i}.csv", "w") as cgroup:

            # run over samples in the df, and write the image names of the samples from group i into the file
            for j in range(len(df)):
                tcga = df["tcga_name"].iloc[j]
                label = df["cluster"].iloc[j]
                dataset = df["dataset"].iloc[j]
                if dataset == i:
                    for image in image_lst:
                        if tcga == image[17:29]:
                            cgroup.write(image+","+label+"\n")



def load_data(train_csv, bs, num_workers, csv_location, image_location, device):

    '''
    a function that set the transformations and loads the training data.
    '''

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
        'train': CATEGORICAL(csv_file = csv_location+f"{train_csv}", rootdir = image_location ,transform=image_transforms['train'])
    }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])

    # Create iterators for the Data loaded using DataLoader module
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == "cuda:0" else {}
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True, **kwargs)

    return train_data_loader, train_data_size, image_transforms


def load_resnet(weight):

    '''
    load resnet18, set loss function and training parameters
    '''

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

    return resnet18, loss_func, optimizer



def train_model(model, loss_func, optimizer, epochs, train_data_loader, train_data_size, image_transforms, dataset, device, output_files, fold):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        best epoch: the epoch number with highest balanced accuracy
    '''

    t_predictions = []
    t_labels = []
    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        model.train()

        # Loss within the epoch
        train_loss = 0.0
        train_acc = 0.0

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


        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, epoch_end - epoch_start))
        print("learning rate: "+str(optimizer.param_groups[0]['lr']))

        # save the last model
        if epoch + 1 == epochs:
            torch.save(model.state_dict(), output_files+dataset+f'_model_{fold}.pt')
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%")

    return model


def model_evaluation(csv_file, model, image_transforms, csv_location, temp_files, image_location, device, num_workers_test):

    '''
    parameters:
        - csv file that has a list with image names and labels for the testing group
        - trained model
        - image transformation

    returns:
        - dataf: a dataframe containing sample name, ground truth label, median class 0 prob, average class 0 prob
        - low_tiles: a list with the class 0 porbabilitys of the image tiles of class 0 patietns
        - high_tiles: a list with the class 0 porbabilitys of the image tiles of class 1 patietns
    '''

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
    image_names_high =[]
    image_names_low =[]

    # create a dataframe with all image names and labels for the test set
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

        kwargs = {'num_workers': num_workers_test, 'pin_memory': True} if device == "cuda:0" else {}
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
            # append tile names to the names list
            image_names_high.extend(df['tcga_name'].iloc[df[df["tcga_name"].str.contains(patient)].index[1:]])
        else:
            low_tiles = np.concatenate((low_tiles, classzero), axis=0)
            # append tile names to the names list
            image_names_low.extend(df['tcga_name'].iloc[df[df["tcga_name"].str.contains(patient)].index[1:]])


    # combine the tile data for all of the patients in the fold
    low = pd.DataFrame({'low_prob': low_tiles, 'patient': image_names_low})
    low = low.assign(label = 'low')
    high = pd.DataFrame({'low_prob': high_tiles, 'patient': image_names_high})
    high = high.assign(label = 'high')
    tiles = pd.concat([low,high], ignore_index=True, axis=0)

    # create a dataframe with the label and median of each patient
    d = {'patient': patient_names, 'labels': patient_labels, 'median': classzero_median, 'average': classzero_average}
    dataf = pd.DataFrame(data=d)
    dataf.columns=['tcga_name','label','median', 'average']

    return dataf, tiles
