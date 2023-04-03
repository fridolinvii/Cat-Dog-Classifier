
import os
import torch
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
import tensorflow as tf
from datetime import datetime

logdir = "runs/test_" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)



#function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


gpu = "cuda:0"  # Say what GPU you want to use

input_size  = 224*224*3   # images are 224*224 pixels and has 3 channels because of RGB color
output_size = 2      # there are 2 classes---Cat and dog

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 64


# define test data directories
data_dir = './data/'
test_dir = os.path.join(data_dir, 'test/')


#create transformers
image_size = (224, 224)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])


 ## read data set using the custom class
test_dataset = datasetloader(test_dir, transform=test_transforms)

## load data using utils
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
     num_workers=num_workers)

accuracy_list = []


            
def test(model,gpu):
    # Start tensorboard writing
    model.eval()
    test_loss = 0
    correct = 0
    count = -1
    img = torch.tensor(-1)
    max_outputs = 25
    for data, target in test_loader:
        data = data.to(gpu) 
        target = target.to(gpu) 
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()


        # Just write out the wrongly labeled data
        # We always write max_outputs = 25 images together
        for i in range(target.shape[0]):
            if pred[i]!=target[i]:
                # be sure that it is in the correct order. For RGB the RGB channel has to be in the last dimension
                # it has to be four dimensional. You can iterate in the first channel over the images
                img = torch.permute(data[i,:,:,:],(1,2,0)).unsqueeze(0).cpu()
                with file_writer.as_default():
                    count += 1
                    tf.summary.image("Wrongly classified", img.numpy(), max_outputs=max_outputs, step=count)


    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))




# Testing the model
numberOfEpochs = 3
model = m.CNN(input_size, output_size)
## load the saved state of the model (best)
state = torch.load("results/model_best.pt")
model.load_state_dict(state['network'])  # apply the weights
model.to(gpu)
print('Number of parameters: {}'.format(get_n_params(model)))
print('Number of epochs: {}'.format(state['epoch']))
test(model,gpu)


