
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from dataset import datasetloader
import model as m

#function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


gpu = th.device("cuda:0" if th.cuda.is_available() else "cpu")  # Say what GPU you want to use

input_size  = 224*224*3   # images are 224*224 pixels and has 3 channels because of RGB color
output_size = 2      # there are 2 classes---Cat and dog

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 64


# define training and test data directories
data_dir = './data/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')


#create transformers
image_size = (224, 224)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                                    transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])
test_transforms = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])


 ## read data set using the custom class
train_dataset = datasetloader(train_dir, transform=train_transform)
test_dataset = datasetloader(test_dir, transform=test_transforms)

## load data using utils
train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size,
     num_workers=num_workers, shuffle=True)
test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
     num_workers=num_workers)

accuracy_list = []





def train(epoch, model, gpu): 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        #print(data[0].shape)
        optimizer.zero_grad()
        data = data.to(gpu) 
        target = target.to(gpu) 
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model,gpu):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(gpu) 
        target = target.to(gpu) 
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))



# Creates folder to save results
try:
    os.mkdir("results")
except OSError as error: 
    print(error)    


# Training settings for model 
numberOfEpochs = 3
model = m.CNN(input_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model)))
for epoch in range(0, numberOfEpochs):
    model = model.to(gpu)     
    train(epoch, model,gpu)
    test(model,gpu)
    
    # save model
    state = {
        'epoch': epoch,
        'network': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
    th.save(state, "results/model.pt")
