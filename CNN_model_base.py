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
            torch.nn.Conv2d(16, 24, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(146016, 1200),
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
    def forward(self, X):
       conv2d= self.cnn_layers(X)
       print(conv2d.shape)
       return conv2d
      
    
    def predict_probs(self, features):
        with torch.no_grad():
            return self.forward(features)

def train(model, epoch = 10):
    writer = SummaryWriter()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_idx = 0
    for epochs in range(epoch):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            print(f'Prediction shape: {prediction.shape} \n',
                f'Prediction Length: {len(prediction)} \n', 
                    f'Prediction Values; {prediction} \n ',
                    f'Labels Shape: {labels.shape} \n',
                    f'Labels: {labels}')
            loss = F.cross_entropy(prediction, labels)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            train_accuracy = metrics.accuracy_score(
                labels.cpu(), prediction.cpu())
            writer.add_scalar("Train Loss", loss.item(), batch_idx)
            writer.add_scalar("Train Accuracy", train_accuracy, batch_idx)
            batch_idx += 1

    model_save_name = f'{path}/image_model_evaluation/image_cnn.pt'
    torch.save(model.state_dict(), model_save_name)

if __name__ == '__main__':
    dataset = FbMarketImageDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CNNBuild()
    train(model)
            

# %%
