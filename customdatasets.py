import os
import pandas as pd
import torch
from torch.utils.data import dataset
import PIL
import zipfile

__all__ = ['cancer_images','MSI_images_zip']

'''
def _find_classes(self, csv):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """

    #$$
    pd.read_csv(csv)
    classes = []


    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
'''

class MSI_images:
    def __init__(self, csv_file, rootdir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.rootdir = rootdir
        self.transform = transform
        self.class2index = {"MSS":0, "MSI":1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.rootdir, self.annotations.iloc[index, 0])
        image = PIL.Image.open(img_path)
        categor = torch.tensor(int(self.class2index[self.annotations.iloc[index, 1]]))

        if self.transform:
            image = self.transform(image)

        return image, categor


class MSI_images_zip:

    #csv_file - name of the csv file including location to it.
    #zip_file - name of the zip file (containing images without any folders) including location to it.
    #there should be a different csv file for training, validating or testing.

    def __init__(self, csv_file, zip_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.zip_file = zip_file
        self.transform = transform
        self.class2index = {"MSS":0, "MSI":1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imgzip = zipfile.ZipFile(self.zip_file)
        #namelist = imgzip.namelist()
        ifile = imgzip.open(self.annotations.iloc[index, 0])
        image = PIL.Image.open(ifile)
        expression = torch.tensor(int(self.class2index[self.annotations.iloc[index, 1]]))

        if self.transform:
            image = self.transform(image)

        return image, expression


class NUMERICAL:
    def __init__(self, csv_file, rootdir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.rootdir = rootdir
        self.transform = transform
        #self.class2index = {"MSS":0, "MSI":1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.rootdir, self.annotations.iloc[index, 0])
        image = PIL.Image.open(img_path)
        predicted_value = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, predicted_value


class CATEGORICAL:
    def __init__(self, csv_file, rootdir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.rootdir = rootdir
        self.transform = transform
        self.class2index = {"low":0, "high":1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.rootdir, self.annotations.iloc[index, 0])
        image = PIL.Image.open(img_path)
        categor = torch.tensor(int(self.class2index[self.annotations.iloc[index, 1]]))

        ''' for testing purpose
        if output_f != 1:
            with open(f"/home1/guyshani/predict_expression_counts/output_files/{self.output_f}_imageorder.csv","a") as out:
                out.write(self.annotations.iloc[index, 0]+"\n")
        '''

        if self.transform:
            image = self.transform(image)

        return image, categor




class CATEGORICAL_zip:

    def __init__(self, csv_file, zip_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.zip_file = zip_file
        self.transform = transform
        self.class2index = {"low":0, "high":1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imgzip = zipfile.ZipFile(self.zip_file)
        #namelist = imgzip.namelist()
        ifile = imgzip.open(self.annotations.iloc[index, 0])
        image = PIL.Image.open(ifile)
        expression = torch.tensor(int(self.class2index[self.annotations.iloc[index, 1]]))

        if self.transform:
            image = self.transform(image)

        return image, expression
