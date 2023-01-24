#%%
## Resnet Model
import torch
import torch.nn 
import pickle
import os
import copy
from pathlib import Path
from Image_Dataset import FbMarketImageDataset
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import resnet50
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import metrics
from torch.utils.data import DataLoader
from Transformers import data_transforms
from tqdm import tqdm
from time import sleep
from torch.utils.tensorboard import SummaryWriter

class ResnetCNN(torch.nn.Module):
    '''Using a pretrained model of the ResNet50 model to improve model functionality'''
    def __init__(self, decoder = None):
        super().__init__()
        self.decoder = decoder
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).to(device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
        for i, parameter in enumerate(self.resnet50.parameters()):
            if i < 47:
                parameter.requires_grad = False
            else: 
                parameter.requires_grad = True

        output_fc = self.resnet50.fc.in_features

        self.resnet50.fc = torch.nn.Sequential(
            torch.nn.Linear(output_fc, 512), # Need to change the first number
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 13)
    )
    

    def forward(self, x):
        return self.resnet50(x)
    
   

def train(model:ResnetCNN, epochs=12):
    dir = f"/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/ML_Models/image_model_evaluation"
    timestamp = f'{datetime.now()}'
    timestamp = timestamp.replace(' ', '__').replace(':', '_').split('.')[0]
    path = f"{dir}/{timestamp}_weights"
    Path(path).mkdir(parents=True, exist_ok=True)
    model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, betas =[0.9,0.999], eps=1e-8)
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    for epoch in tqdm(range(epochs), 'Epochs'):
        batch_idx = 0 
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                for i, batch in enumerate(train_loader):
                    train_all_loss = []
                    train_all_acc = []                    
                
                    features, labels = batch
                    features, labels = features.to(device), labels.to(device)
                    optimiser.zero_grad()
                   
                    prediction = model(features)
                    train_loss = criterion(prediction, labels)
                    train_loss.backward()
                    train_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
                    optimiser.step()
                    writer.add_scalar("Train Loss", train_loss.item(), batch_idx)
                    writer.add_scalar("Train Accuracy", train_accuracy, batch_idx)
                    batch_idx +=1 
                    train_all_loss.append(train_loss.item())
                    train_all_acc.append(train_accuracy)
                    print(f"Batch Round: {batch_idx}\
                        Train_Loss = {train_loss.item()}\
                        Train Accuracy: {train_accuracy}")
                             
            if phase == 'validation':
                model.eval()
                for i, batch in enumerate(validation_loader):
                    print('Validating Data...:')
                    with torch.no_grad():
                        validation_all_loss = []
                        validation_all_acc = []
                        features, labels  = next(iter(validation_loader))
                        features, labels = features.to(device), labels.to(device)
                        prediction = model(features)
                        validation_loss = criterion(prediction, labels)
                        validation_loss.requires_grad = True
                        validation_loss.backward()
                        validation_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
                        writer.add_scalar("Validation Loss", validation_loss.item(), batch_idx)
                        writer.add_scalar("Validation Accuracy", validation_accuracy, batch_idx)
                        validation_all_loss.append(validation_loss.item())
                        validation_all_acc.append(validation_accuracy)
                        batch_idx +=1 
                        print(f"Batch Round: {batch_idx} \
                        Validation Loss: {validation_loss.item()}\
                        Validation Accuracy : {validation_accuracy}") 

        epoch +=1
        print(f"Epoch: {epoch}\
                Train Loss: {sum(train_all_loss)/len(train_all_loss)}\
                Train Accuracy : {sum(train_all_acc)/len(train_all_acc)}\
                Validation Loss: {sum(validation_all_loss)/len(validation_all_loss)}\
                Validation Accuracy : {sum(validation_all_acc)/len(validation_all_acc)}")
        
        if sum(validation_all_acc)/len(validation_all_acc) > best_acc:
            best_acc = sum(validation_all_acc)/len(validation_all_acc)
            model_wts = copy.deepcopy(model.state_dict())
            torch.save(model_wts, f'{path}/ADAM-ACC-{validation_accuracy}_final_image_cnn.pt')
        
        writer.flush()
    model.load_state_dict(model_wts)
    torch.save(model,f'{path}/ADAM-ACC-{validation_accuracy}_final__full_model_image_cnn.pt')
        
def check_accuracy(model:ResnetCNN, loader):
    path = f"/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/ML_Models/image_model_evaluation/Epochs"
    print('Testing Model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            scores = model(features)
            _, preds = torch.max(scores, 1) 
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
        acc = float(num_correct / num_samples)
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
        model.train()
        return acc                     

if __name__ == '__main__':
    batch_size = 32
    dataset = FbMarketImageDataset(transformer = data_transforms['train'])
    
    dataset_length = int(len(dataset))
    train_split = round(dataset_length * 0.7)
    test_split = round(dataset_length * 0.10)
    val_split = round(dataset_length * 0.20)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, 
    [train_split, val_split, test_split], generator = torch.Generator().manual_seed(42))

    # Set up transforms
    train_dataset.transformer = data_transforms['train']
    val_dataset.transformer = data_transforms['validation']
    test_dataset.transformer = data_transforms['validation']
    
    train_loader = DataLoader(train_dataset, 
                            shuffle = True,  
                            batch_size= batch_size, 
                            num_workers=2)
    validation_loader = DataLoader(val_dataset, 
                                shuffle = True,  
                                batch_size= batch_size, 
                                num_workers=2)
    test_loader = DataLoader(test_dataset, 
                            shuffle = True, 
                            batch_size= batch_size, 
                            num_workers=2)

    print(dataset[0])
    print('Training Model...')
    model = ResnetCNN()
    train(model)
    path = f"/home/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/ML_Models/image_model_evaluation"
    print('Testing Model...')
    check_accuracy(loader = test_loader, model= model)
    timestamp = datetime.timestamp(datetime.now())
    os.chdir(path)
    model_save_name = f'{timestamp}_image_cnn.pt'
    torch.save(model.state_dict(), model_save_name)
