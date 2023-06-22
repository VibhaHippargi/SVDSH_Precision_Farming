import os
import torch, torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from efficientnet_pytorch import EfficientNet

#User parameter setting 
BATCH_SIZE = 16 #Number of image samples selected in the batch during training
NUM_OF_CLASS = 4  #Number of image class in the dataset
NUM_EPOCHS = 10

RESNET18 = False
MOBILENETV2 = True
RESNET34 = False
EFFNET = False

# Define the categories
categories = ['Sugarbeet', 'Charlock', 'Stop', 'Priority']

# Define a custom dataset
class classifierDataset(Dataset):
    def __init__(self, baseDir, transform=None):
        self.root_dir = baseDir
        self.transform = transform
        self.file_list = []

        for idx, category in enumerate(categories):
            category_dir = os.path.join(baseDir, category)
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

#Setting image transformation parameters
#Correcting brightness, contrast, hue
#Image resize
#Conversion to numpy array and converting pixel value to 0-1
#Normalise image tensor
transform = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Getting link to image dataset
dataset = classifierDataset(baseDir='Provide path to image dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if MOBILENETV2:
    #Selecting the resnet 18 model
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(model.last_channel, NUM_OF_CLASS)

if RESNET18:
    #Selecting the resnet 18 model
    model = torchvision.models.resnet18(pretrained=True)
    numberOfFeatures = model.fc.in_features
    model.fc = torch.nn.Linear(numberOfFeatures, NUM_OF_CLASS)  

if RESNET34:
    #Selecting the resnet 18 model
    model = torchvision.models.resnet34(pretrained=True)
    numberOfFeatures = model.fc.in_features
    model.fc = torch.nn.Linear(numberOfFeatures, NUM_OF_CLASS)  

if EFFNET:
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = torch.nn.Linear(1280, NUM_OF_CLASS)

#Loading the model to the selected device
#Select cuda if available else select cpu
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 10)

    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Loss: {epoch_loss:.4f}')

print('Done with training the model')

# Save the trained model
torch.save(model.state_dict(), 'Provide path to save model')