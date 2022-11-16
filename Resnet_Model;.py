# Resnet Model
import torch
import torch.nn 
import pickle
from Image_Dataset import FbMarketImageDataset
from torch.utils.data import DataLoader
from torchvision import models, datasets
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ResnetCNN(torch.nn.Module):
    '''Using a pretrained model of the ResNet50 model to improve model functionality'''
    def __init__(self, decoder = None):
        super().__init__()
        self.decoder = decoder
        self.resnet50 = self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).to(device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            
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
    
    def predict(self, image):
        with torch. no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim =1)
    
    def predict_class(self, image):
        if self.decoder == None:
            raise Exception('Decoder needs to be passed when instantiating model')
        else:
            with torch.no_grad():
                x = self.forward(image)
                return self.decoder(int(torch.argmax(x, dim = 1)))
        

def train(model:ResnetCNN, epochs = 10):
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.001)
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    batch_idx = 0
    for epoch in range(epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                for i, batch in enumerate(train_loader):
                    print('Training Dataset...')
                    optimiser.zero_grad()
                    features, labels = batch
                    features, labels = features.to(device), labels.to(device)
                    prediction = model(features)
                    train_loss = criterion(prediction, labels)
                    train_loss.backward()
                    train_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
                    optimiser.step()
                    writer.add_scalar("Train Loss", train_loss.item(), batch_idx)
                    writer.add_scalar("Train Accuracy", train_accuracy, batch_idx)
            if phase == 'validation':
                for i, batch in enumerate(validation_loader):
                    with torch.no_grad():
                        print('Validating using subset of data...')
                        model.eval()
                        features, labels  = next(iter(validation_loader))
                        features, labels = features.to(device), labels.to(device)
                        prediction = model(features)
                        validation_loss = criterion(prediction, labels)
                        validation_loss.backward()
                        validation_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
                        writer.add_scalar("Validation Loss", validation_loss.item(), batch_idx)
                        writer.add_scalar("Validation Accuracy", validation_accuracy, batch_idx)
                        batch_idx +=1  
                print(f"Batch Round: {batch_idx} \
                        Train_Loss = {train_loss.item} \
                        Train Accuracy: {train_accuracy} \
                        Validation Loss: {validation_loss.item} \
                        Validation Accuracy : {validation_accuracy}")
            
        print(f"Epoch: {epoch} \
                Train_Loss = {train_loss.item} \
                Train Accuracy: {train_accuracy} \
                Validation Loss: {validation_loss.item} \
                Validation Accuracy : {validation_accuracy}")

        writer.flush()

def check_accuracy(model: ResnetCNN, loader):
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            scores = model.predict(features)
            _, preds = scores.max(1)
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
            acc = float(num_correct/num_samples *100)
            print(f'Testing Accuracy:{acc}')
        return acc

if __name__ == '__main__':
    batch_size = 12
    dataset = FbMarketImageDataset()
    decoder = dataset.decode
    dataset_length = int(len(dataset))
    train_split = round(dataset_length * 0.7)
    test_split = round(dataset_length * 0.1)
    val_split = round(dataset_length * 0.2)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, 
    [train_split, val_split, test_split], generator = torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=1)
    validation_loader = DataLoader(val_dataset, batch_size= batch_size, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, num_workers=1)

    phases = [train_loader, validation_loader]
    print(dataset[0])
    model = ResnetCNN()
    print('Training Model...')
    train(model)
    print('Testing Model...')
    check_accuracy(loader = test_loader)
    path = f"/Users/jazzy/Documents/AiCore_Projects/Facebook-Marketplace-Ranking/data/ML_Models"
    model_save_name = f'{path}/image_model_evaluation/image_cnn.pt'
    torch.save(model.state_dict(), model_save_name)


# %%
