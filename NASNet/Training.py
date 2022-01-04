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


def TrainNN(layers): # Layers: [x,x,x,x], block: Which block type(base or botteneck)
    # Defining the network
    net = RN.ResNetX(img_channels = channels, num_classes = classes, layers = layers)
    if use_cuda:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-7) # Stochastic gradient descent
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss

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
    
    return losses[-1] # .detach().numpy() # return the last lost


    '''
    epoch = np.arange(len(train_acc))
    plt.figure()
    plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b', epoch, train_losses, 'm')
    plt.legend(['Train Accucary','Validation Accuracy'])
    plt.xlabel('Updates'), plt.ylabel('Acc')
    plt.savefig(f'ResNet18_{tra_size}_{val_size}_{num_epochs}')
    plt.show()
    '''

    return # some parameter which we want to optimize. 

def choose_layers(b, l): # A method for returning how many times each block should be repeated
    layers = [] # The number of layers for each block
    index = [] # The index corrosponding to each layer(used later when updating probs)

    for i in range(len(b)):
        numbers = [x for x in range(len(b[i]))]
        index.append(np.random.choice(numbers,p=b[i]))
        layers.append(l[i][index[i]])
    return layers, index # Returning two list of length b_i = 4

def choose_optimal_layers(b,l):
    layers = []
    for i in range(len(b)):
        layers.append(l[i][np.argmax(b[i])])
    return layers

def zero_mean_rewards(losses):
    mean_value = np.mean(losses)
    rewards = np.zeros(len(losses))
    for i in range(len(losses)):
        rewards[i] = (losses[i] - mean_value)/mean_value 
    return rewards
'''
def logits(probs):
    for i in range(len(probs)):
        probs[i] = np.log(probs[i]/(1-probs[i]))
    return probs
'''
def logits(probs):
    for i in range(len(probs)):
        if probs[i] == 1: # To avoid dividing with 0
            a = np.ones(len(probs))*(-30)
            a[i] = 30
            return a
        probs[i] = np.log(probs[i]/(1-probs[i]))
    return probs

def softmax(vec):
    max_number = np.max(vec)
    vec = vec - max_number
    exponential = np.exp(vec)
    probabilities = exponential / np.sum(exponential)
    '''
    for i in range(len(vec)):
        if probabilities[i] < 1e-5:
            probabilities[i] = 0
    '''
    return probabilities

def update_probs(b_i_old, ResNets_index, rewards, i, alpha=1): # i is the b_i we are looking at 

    # Finding the logits of b_i
    lgt_i_old = logits(b_i_old) 

    # Finding the softmax values for the logits used in the updating step. Think logits and softmax are cancelling eachother
    sft_max = softmax(lgt_i_old)

    # Creating a vector for loop values (4)
    upd_stp = np.zeros(len(b_i_old)) # It's all initalised at zero
    print(lgt_i_old)
    print(sft_max)
    print(upd_stp)
    print(ResNets_index)
    # The updating step -> the gradient of our softmax likelihood function
    for n in range(len(rewards)):
        slct_val = ResNets_index[n][i] # the selected value for network j when looking at b_i

        upd_stp[slct_val] -= rewards[n] * (1-sft_max[slct_val]) # the negativ is since we are using the loss as rewards
    
    # Taking the mean 
    upd_stp = upd_stp/len(rewards)
    
    # Multiplying with the alpha value
    upd_stp = upd_stp * alpha
    
    # Finally we can the updating steps to the old logits
    lgt_i_new = lgt_i_old + upd_stp

    # lastly we take the softmax such that we get the probabilites
    b_i_new = softmax(lgt_i_new)

    return b_i_new

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

#quit()
#targets_train = targets_train.to(torch.float)

x_valid = mnist_trainset.data[tra_size:val_size].view(-1, 784).float()
x_valid = x_valid.reshape((x_valid.shape[0], 1, 28, 28))
targets_valid = mnist_trainset.targets[tra_size:val_size]
#targets_valid = targets_valid.to(torch.float)

# Building the training loop
# Normalizing the inputs
x_train.div_(255)
x_valid.div_(255)

# Defining the loss function and the optimizer
channels = 1 # b/w  = 1 channel
classes = 10 # Numbers to predict
batch_size = 400
num_epochs = 7


# Initialising super parameters
super_loop = 16 # 10-100
n_networks = 16
b = np.array([[1/3,1/3,1/3], [1/3,1/3,1/3], [1/3,1/3,1/3], [1/3,1/3,1/3]])
l = np.array([[1,2,3], [1,2,3], [1,2,3], [1,2,3]])

b_print = np.empty([super_loop,len(b),len(b[0])])
avg_loss = [] # the average loss for each iteration size [super_loop] * [n_networks]
# Training the super archicture 

# Printing the parameters: 
print(f'Training size: {tra_size}, Validation size: {tra_size}:{val_size}, batch sizes: {batch_size},neural_networs: {n_networks}, Epochs_ResNet: {num_epochs}, Epochs_SuperLoop: {super_loop}')

start_time = time.time()

for k in range(super_loop):
    b_print[k] = b
    #print(f'Epoch {k} : b_values {b} ')
    losses = np.zeros(n_networks) # array for the losses

    # 'creating' n ResNets
    ResNets_layers = np.empty([n_networks,len(l)], dtype=int)
    ResNets_index = np.empty([n_networks,len(l)], dtype=int)

    for i in range(n_networks):
        # ResNets_layers and ResNets_index is n*len(b) (10*4)
        layers, index = choose_layers(b,l)
        ResNets_layers[i] = layers
        ResNets_index[i] = index

    # Training the networks
    for i in range(n_networks):
        ll = TrainNN(layers=ResNets_layers[i]) 
        losses[i] = ll # losses is an list of n elements (10)
    avg_loss.append(losses) # For printing

    # Calculating the zero mean ranking rewards 
    rewards = zero_mean_rewards(losses) # rewards is a list of n elements (10) 

    #Updating the probabilities
    for i in range(len(b)): # Updating the b-values one probability at a time (4)
        b[i] = update_probs(b[i], ResNets_index, rewards, i)


# Finding optimal variables
opt_layers = choose_optimal_layers(b,l)

# Training the optimal network
loss = TrainNN(layers=opt_layers)

#
print("--- %s seconds ---" % (time.time() - start_time))
print(f'The final loss when training with the optimal layers is: {loss}')

# Printing the optimized probabilites
print('b_print[i]')
for i in range(super_loop):
    print(b_print[i])

np.savetxt(f'NASNet_tsize_{tra_size}_vsize_{val_size}_Bsize_{batch_size}_nnetworks_{n_networks}_Epoch_ResNet_{num_epochs}_Epoch_Super_{super_loop}.txt',avg_loss)
x = [i+1 for i in range(super_loop)]
for xe, ye in zip(x,avg_loss):
    plt.scatter([xe] * len(ye), ye, label=f'{xe}')
plt.title('Results over Superloop')
plt.xlabel('Iterations in the SuperLoop')
plt.ylabel('Training error')
plt.legend()
plt.savefig(f'NASNet_tsize_{tra_size}_vsize_{val_size}_Bsize_{batch_size}_nnetworks_{n_networks}_Epoch_ResNet_{num_epochs}_Epoch_Super_{super_loop}.png')





'''
In the article they describe two updatas. What do they mean with the first step? And is that what we already have done? 
    ' The   The weights, w, of the selected operations are then updated to minimize the expected value of a defined loss function,
    L, between the predicted simulation output and the actual simulation output'

Which performance rankings should we focus on? 

Maybe an overall breakdown of the structure. 

alpha = 0.02-0.05 (update rate)


softmax af probabilites. Mere numerisk stabilt muligvis 




'''


