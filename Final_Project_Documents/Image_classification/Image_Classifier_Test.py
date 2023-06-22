import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import time
from efficientnet_pytorch import EfficientNet

#User parameter setting 
BATCH_SIZE = 16 #Number of image samples selected in the batch during training
NUM_OF_CLASS = 4  #Number of image class in the dataset
NUM_EPOCHS = 10

RESNET18 = False #Activate resnet18
MOBILENETV2 = True #Activate mobilenetv2
EFFNET = False #Activate efficient net
INDVIMGCHECK = False #Activate individual image check
RESNET34 = False #Activate resnet34

# Define the categories
categories = ['Sugarbeet', 'Charlock', 'Stop', 'Priority']

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []

        for idx, category in enumerate(categories):
            category_dir = os.path.join(root_dir, category)
            file_names = os.listdir(category_dir)
            file_paths = [os.path.join(category_dir, file_name) for file_name in file_names]
            self.file_list.extend([(file_path, idx) for file_path in file_paths])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path, label = self.file_list[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
if INDVIMGCHECK == False:
    # Define the transformations for the test images
    transform = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if MOBILENETV2:
        #Selecting the resnet 18 model
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.last_channel, NUM_OF_CLASS)

    if RESNET18:
        #Selecting the resnet 18 model
        model = models.resnet18(pretrained=True)
        numberOfFeatures = model.fc.in_features
        model.fc = torch.nn.Linear(numberOfFeatures, NUM_OF_CLASS)  

    if RESNET34:
        #Selecting the resnet 18 model
        model = models.resnet34(pretrained=True)
        numberOfFeatures = model.fc.in_features
        model.fc = torch.nn.Linear(numberOfFeatures, NUM_OF_CLASS)  

    if EFFNET:
        model = EfficientNet.from_pretrained("efficientnet-b0")
        model._fc = torch.nn.Linear(1280, NUM_OF_CLASS)

    # Load the saved model weights
    model.load_state_dict(torch.load("load model path"))
    model.eval()

    # Load the test dataset
    test_dataset = CustomDataset(root_dir='provide test folder dataset', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Measure inference time and accuracy
    total_samples = len(test_dataset)
    correct_predictions = 0
    total_time = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            _, predicted_idx = torch.max(outputs, 1)
            correct_predictions += (predicted_idx == labels).sum().item()
            total_time += end_time - start_time

    accuracy = correct_predictions / total_samples
    avg_inference_time = total_time / total_samples

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Average Inference Time: {avg_inference_time:.4f} seconds')

#This part checks an individual image input
if INDVIMGCHECK:
    # Define the categories
    categories = ['Sugarbeet', 'Charlock', 'Stop', 'Priority']

    # Define the transformations for the test images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if MOBILENETV2:
        #Selecting the resnet 18 model
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.last_channel, NUM_OF_CLASS)

    if RESNET18:
        #Selecting the resnet 18 model
        model = models.resnet34(pretrained=False)
        numberOfFeatures = model.fc.in_features
        model.fc = torch.nn.Linear(numberOfFeatures, len(categories))  

    if EFFNET:
        model = EfficientNet.from_pretrained("efficientnet-b0")
        model._fc = torch.nn.Linear(1280, NUM_OF_CLASS)

    # Load the saved model
    model.load_state_dict(torch.load('load model path'))
    model.eval()

    # Load and preprocess the test image
    test_image_path = 'provide image path'  
    test_image = Image.open(test_image_path).convert('RGB')
    test_image = transform(test_image)
    test_image = torch.unsqueeze(test_image, 0)  


    with torch.no_grad():
        outputs = model(test_image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = categories[predicted_idx.item()]

    print(f'The predicted label for the test image is: {predicted_label}')