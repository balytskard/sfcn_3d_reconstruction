import torch
import torch.nn as nn

import random
import numpy as np


# Set seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

params = {
    'batch_size': 5,
    'imagex': 160,
    'imagey': 192,
    'imagez': 160,
    'column': "Group_bin"
}

class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2, stride=2, padding=0)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2, stride=2, padding=0)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2, stride=2, padding=0)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2, stride=2, padding=0)
        
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm3d(256)
        self.pool5 = nn.MaxPool3d(2, stride=2, padding=0)
        
        self.conv6 = nn.Conv3d(256, 64, kernel_size=1, padding=0)
        self.norm6 = nn.BatchNorm3d(64)
        
        self.avgpool = nn.AvgPool3d(2)
        self.dropout = nn.Dropout(0.2)
        
        self.flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 96)
        self.fc2 = nn.Linear(96, 2)
        
    def _get_flattened_size(self):
        x = torch.zeros(1, 1, params['imagex'], params['imagey'], params['imagez'])
        x = self.pool1(self.norm1(self.conv1(x)))
        x = self.pool2(self.norm2(self.conv2(x)))
        x = self.pool3(self.norm3(self.conv3(x)))
        x = self.pool4(self.norm4(self.conv4(x)))
        x = self.pool5(self.norm5(self.conv5(x)))
        x = self.norm6(self.conv6(x))
        x = self.avgpool(x)
        x = self.dropout(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.norm1(self.conv1(x))))
        x = self.pool2(torch.relu(self.norm2(self.conv2(x))))
        x = self.pool3(torch.relu(self.norm3(self.conv3(x))))
        x = self.pool4(torch.relu(self.norm4(self.conv4(x))))
        x = self.pool5(torch.relu(self.norm5(self.conv5(x))))
        x = torch.relu(self.norm6(self.conv6(x)))
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        if target.dim() > 1:
            target = target.squeeze() 
        target = target.long() 
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.long().to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            
            preds = outputs.argmax(dim=1) 
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total

