import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import torch
from torch import nn
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import pandas as pd

# data location
dirs =  "/home/maruvka/Documents/predict_expression/output_files/"
# load a list of genes to process
gene_list = ""
# number of patients from each gene
patient_num = 10
# number of tiles from each patient
tiles_num = 10

def evaluate(image_name, image_label):
    '''
    inputs:
    - image name
    - image label - the true label (high or low) of the image

    output:
    saves an image with CAM visualization
    '''
    if image_label == "high":
        o_target = 1
    elif image_label == "low":
        o_target = 0
    else:
        print("label error")

    model.eval()
    # open and resize image
    image_loc = f"/home/maruvka/Documents/predict_expression/complete_dataset/{image_name}"
    img = np.array(Image.open(image_loc))
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # The target for the CAM is the Bear category.
    # As usual for classication, the target is the logit output
    # before softmax, for that category.
    targets = [ClassifierOutputTarget(o_target)]
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img), cam , cam_image))
    out_image = Image.fromarray(images)
    out_image.save(f"/home/maruvka/Documents/predict_expression/CAM_images/{image_name}_CAM.png")


# load model
model = models.resnet18()
model.fc = nn.Linear(512, 2)

#gene = "AKAP9"
#sample = pd.read_csv(dirs+f"output_{gene}/{gene}_patients_probs.csv")
#sample.sort_values('average', ascending= False)
#sample = sample.head( n = 10)
#sample = sample['tcga_name'].tolist()


for gene in gene_list:

    # load dataframe with patient level scores
    sample = pd.read_csv(dirs+f"output_{gene}/{gene}_patients_probs.csv")
    sample.sort_values('average', ascending= False)
    sample = sample.head( n = patient_num)
    # create a list with patient names for 10 highest ranking patients
    sample = sample['tcga_name'].tolist()

    # load dataframe with tile scores
    tiles = pd.read_csv(dirs+f"output_{gene}/{gene}_tiles_probs.csv")
    low_tiles = tiles[tiles['patient'].str.contains("TCGA-AH-6643")].sort_values('low_prob', ascending=False).head(n = tiles_num)
    low_tiles_names = low_tiles['patient'].tolist()
    high_tiles = tiles[tiles['patient'].str.contains("TCGA-AH-6643")].sort_values('low_prob', ascending=True).head(n = tiles_num)
    high_tiles_names = high_tiles['patient'].tolist()

    # load model
    model.load_state_dict(torch.load(dirs+f"output_{gene}/CRC-DX_classes_{gene}_model_0.pt"))

    for name in low_tiles_names:
        evaluate(name, 0)
    
    for name in high_tiles_names:
        evaluate(name, 1)
