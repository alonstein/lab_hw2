import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from PIL import Image
import numpy as np

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5227, 0.4495, 0.4206], std=[0.2383, 0.2270, 0.2233])
])
batch_size = 64


class DataSet(Dataset):
    def __init__(self, directory, transform):
        self.dir = directory
        self.transform = transform
        self.all_imgs = os.listdir(directory)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        image_file_name = self.all_imgs[index]
        image_path = args.input_folder+'/'+image_file_name
        img = Image.open(image_path)
        return image_file_name, self.transform(img)


def load_dataset(data_path, transform):
    data = DataSet(data_path, transform)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )
    return loader


test_loader = load_dataset(args.input_folder, test_transform)


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(8192, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)


cnn = CNN()
cnn = to_gpu(cnn)
cnn.load_state_dict(torch.load('model'))
cnn.eval()
with torch.no_grad():
    id_list = []
    predictions_list = []
    for ids, images in test_loader:
        images = to_gpu(images)
        outputs = cnn(images)
        predicted = torch.round(outputs).view(-1)
        predictions_list = predictions_list + predicted.cpu().tolist()
        id_list = id_list + list(ids)


# predictions_list = [x.item() for x in predictions_list]
prediction_df = pd.DataFrame(zip(id_list, predictions_list), columns=['id', 'label'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Example - Calculating F1 Score using sklrean.metrics.f1_score
from sklearn.metrics import f1_score
y_true = prediction_df['id'].apply(lambda x: int(x[7:8])).values
f1 = f1_score(y_true, prediction_df['label'].values, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.

print("F1 Score is: {:.4f}".format(f1))


