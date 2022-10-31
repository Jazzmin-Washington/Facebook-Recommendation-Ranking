#%%
# Building CNN Network
import torch
import os
import torch.nn 
import numpy as np
from Image_Dataset import FbMarketImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


path = f"/Users/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data/ML_Models"
class CNNBuild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(215296, 1200),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1200, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(600, 300),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(300, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,13),
            )
    def forward(self, features):
       return self.cnn_layers(features)

    def predict_probs(self, features):
        with torch.no_grad():
            return self.forward(features)
 

def train(model, epoch = 10):
    #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    #model.to(device)
    writer = SummaryWriter()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    batch_idx = 0
    for epochs in range(epoch):
        for batch in train_loader:
            features, labels = batch
            #features = features.to(device)
            #labels = labels.to(device)
            prediction = model(features)
            loss = criterion(prediction, labels)
            loss.backward()
            print(loss.item())
            train_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("Train Loss", loss.item(), batch_idx)
            writer.add_scalar("Train Accuracy", train_accuracy, batch_idx)
            batch_idx += 1

    model_save_name = f'{path}/image_model_evaluation/image_cnn.pt'
    torch.save(model.state_dict(), model_save_name)

if __name__ == '__main__':
    ngpu = 2
    batch_size = 32
    dataset = FbMarketImageDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers = 1)
    model = CNNBuild()
    train(model)
