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

    df_temp = df.copy()
    print("lnumber of patients (cross_val): "+str(len(df)))
    df["dataset"] = ''
    image_lst = os.listdir(image_location)

    # split into datasets and add the dataset label in dataset column
    for label in ['high', 'low']:
        # count the number of samples in each class
        tot = len(df_temp.query('(cluster == @label)'))

        for i in range(k):
            group = np.random.choice(df_temp.query('(cluster == @label)').index.values, int(tot/k), replace = False)
            df["dataset"].iloc[group] = i
            df_temp = df_temp.drop(group)

    for i in range(k):
        with open(csv_location+f"{gene}_crossval_group_{i}.csv", "w") as cgroup:

            for j in range(len(df)):
                marker = "first"
                tcga = df["tcga_name"].iloc[j]
                label = df["cluster"].iloc[j]
                dataset = df["dataset"].iloc[j]
                for image in image_lst:
                    if tcga == image[17:29]:
                        if dataset == i:
                            cgroup.write(image+","+label+"\n")
                            '''
                            if marker == "first":
                                print("patient name: "+str(tcga))
                                marker = "notfirst"
                            '''



def load_data(train_csv, test_csv, bs, num_workers, csv_location, image_location, device):

    '''
    a function that set the transformations and loads the training and testing data.
    '''

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
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == "cuda:0" else {}
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True, **kwargs)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True, **kwargs)

    return train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms


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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 8, threshold = 0.2)

    return resnet18, loss_func, optimizer, scheduler



def train_and_validate(model, loss_func, optimizer, scheduler, epochs, train_data_loader, valid_data_loader, train_data_size, valid_data_size, image_transforms, dataset, device, output_files, j):
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


    best_valid_loss = 0.0
    history = []
    t_predictions = []
    t_labels = []
    v_predictions = []
    v_labels = []
    v_probs =[]
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

                # get predictions in probabilitys
                normal = np.array(torch.nn.functional.softmax(outputs.data, dim=1).cpu())
                v_probs = np.append(v_probs ,normal[:,1], axis =0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        # find accuracys per class
        valid_roc = roc_auc_score(v_labels, v_predictions)
        valid_roc_probs = roc_auc_score(v_labels, v_probs)
        #balanced_accuracy = balanced_accuracy_score(v_labels, v_predictions)

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
        print("ROC probs: "+str(valid_roc_probs))
        #print("balanced accuracy: "+str(balanced_accuracy))

        # save the best model
        if epoch ==0:
            best_valid = valid_roc_probs
            best_epoch = epoch
            torch.save(model.state_dict(), output_files+ dataset+'_model_'+str(epoch)+f'_{j}_fold.pt')
        if valid_roc_probs > best_valid:
            best_valid = valid_roc_probs
            best_epoch = epoch
            print('best validation AUC ROC so far')
            torch.save(model.state_dict(), output_files+dataset+'_model_'+str(epoch)+f'_{j}_fold.pt')
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%")

    #save the training history
    torch.save(history,output_files+dataset+f'_history_{j}_fold.pt')

    # get statistics


    return model, best_epoch


def model_evaluation(csv_file, model, image_transforms, csv_location, temp_files, image_location, device):

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

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda:0" else {}
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
