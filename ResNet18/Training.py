# loading packages
from operator import index
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import time


#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
import ResNet as RN
#import sys

use_cuda = torch.cuda.is_available()
#use_cuda = False
print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

# Arguments simple:
# 1: Training size
# 2: Validation size
# 3: ResNet
# 4: Batch size
# 5: Number of epocs

#arg_list = sys.argv




# Importing the MNIST dataset
mnist_trainset = MNIST("./temp/", train=True, download=True) # Size of 60000
mnist_testset = MNIST("./temp/", train=False, download=True) # Size of 60000
# Only taking a subset
tra_size = 50000 # using 5/6 as training size
val_size = 60000 # using 1/6 as validation size

x_train = mnist_trainset.data[:tra_size].view(-1, 784).float()
x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
targets_train = mnist_trainset.targets[:tra_size]


x_valid = mnist_trainset.data[tra_size:val_size].view(-1, 784).float()
x_valid = x_valid.reshape((x_valid.shape[0], 1, 28, 28))
targets_valid = mnist_trainset.targets[tra_size:val_size]
#targets_valid = targets_valid.to(torch.float)



# Should look into pytorch datashaper

# Defining the loss function and the optimizer
channels = 1 # b/w  = 1 channel
classes = 10 # Numbers to predict
net = RN.ResNet18(img_channels = channels, num_classes = classes)
if use_cuda:
    net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-7) # Stochastic gradient descent
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss

# Building the training loop
# Normalizing the inputs
x_train.div_(255)
x_valid.div_(255)

# Creating batches 
batch_size = 200
num_epochs = 20 # training 200 times

num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train// batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid// batch_size

# Setting up the lists for handling loss/accurazy
train_acc, train_loss = [],[]
valid_acc, valid_loss = [],[]
test_acc, test_loss = [],[]

cur_loss = 0
losses = []
train_losses = []


get_slice  = lambda i, size: range(i * size, (i + 1) * size)

for epoch in range(num_epochs+1):
    # Forward -> Backprob -> update params
    ## Train

    cur_loss = 0
    net.train() # Telling putorch we are training the network now
    for i in range(num_batches_train):
        optimizer.zero_grad() # Setting all the gradients to zero, probably shound't be necessary
        slce = get_slice(i, batch_size) # Using our Lambda function to calculate the slices
        #print(x_train[slce].size())
        # Wrapping in Variables
        input = Variable(get_variable(x_train[slce]))
        output = net(input) # I think only training the batch we are looking at 
        #output = output.reshape(len(output)) # reshaping
        # Compute gradients given loss
        target_batch = Variable(get_variable(targets_train[slce])) # Finding the targets for the current batch/slice
        #print(output,output.shape)
        #print(target_batch, target_batch.shape)
        batch_loss = criterion(output, target_batch) # Calculating the losses based on the current batch
        batch_loss.backward() # finding all of the loss gradients
        optimizer.step() # optimizing the gradients, taking the next step

        cur_loss += get_numpy(batch_loss) #Adding the loss for this batch
    
    losses.append(cur_loss/batch_size) # Append the losses

    if epoch%5 == 0 or epoch == num_epochs:
        ### Evaluating training
        net.eval() # Telling pytourch we are evaluating the data now and not training
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size) 
            input = Variable(get_variable(x_train[slce]))
            output = net(input) # Running the training date trough the network

            preds = torch.max(output,1)[1] # Finding the maximum value of the output for each batch
            train_targs += list(targets_train[slce].numpy()) # Adding the data to list
            train_preds += list(get_numpy(preds)) # Not quite sure

        ### Evaluate validation
        val_preds, val_targs = [],[]
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            input = Variable(get_variable(x_valid[slce]))
            output = net(input)
            preds = torch.max(output,1)[1]
            val_targs += list(targets_valid[slce].numpy())
            val_preds += list(get_numpy(preds))

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        #if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
    else:
        print("Epoch %2i : Train Loss %f" % (epoch+1, losses[-1]))

epoch = np.arange(len(train_acc))
np.savetxt('ResNet18_train_acc.txt',train_acc)
np.savetxt('ResNet18_valid_acc.txt',valid_acc)
np.savetxt('ResNet18_train_losses.txt',train_losses)
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b', epoch, train_losses, 'm')
plt.legend(['Train Accucary','Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')
plt.savefig(f'ResNet18_{tra_size}_{val_size}_{num_epochs}')
plt.show()

