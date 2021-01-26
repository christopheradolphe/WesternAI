# make sure to enable GPU acceleration!
#device = 'cuda'
#device2 = 'cpu'

import torch as pt
import numpy as np
import cv2
import os
import glob
import json
import logging
import shutil
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.utils.data.sampler as t

#GLOBAL VARIABLES
INPUT_DIM = 1
OUTPUT_DIM = 1

#Set a random seed for reproducibility (do not change this)
seed = 42
np.random.seed(seed)
pt.manual_seed(seed)

class TwoLayerNet(pt.nn.Module):
  #aplly all the different layers
  def __init__(self, num_bins):

    # initialize the model by defining each layer
    super(TwoLayerNet, self).__init__()

    # convolution layers
    self.conv1 = pt.nn.Conv2d(1, 8, 3)
    self.conv2 = pt.nn.Conv2d(8, 16, 3)
    self.conv3 = pt.nn.Conv2d(16, 20, 4)

    #batching layers
    self.bn1 = pt.nn.BatchNorm2d(8)
    self.bn2 = pt.nn.BatchNorm2d(16)  

    #Fully connected layers to transform the output of the convolution layers to the final output
    self.fc1 = pt.nn.Linear(292820, 1000)
    
    self.fcbn1 = pt.nn.BatchNorm1d(1000)
    
    self.fc2 = pt.nn.Linear(1000, 100)

    self.fcbn2 = pt.nn.BatchNorm1d(100)

    self.fc3 = pt.nn.Linear(100, num_bins)

    #relu layer
    #self.relu = pt.nn.functional.relu(self.conv(x))

    #pooling layers
    self.pool = pt.nn.MaxPool2d((3,3),stride=3)

    self.dropout = pt.nn.Dropout(p=0.5)

  def forward(self, x):


    #CNN Layers
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)


    #Transition from cnn layers to fully connected
    x = x.view(100, -1)

    #Fully Connected Layers
    x = self.fc1(x)
    x = pt.nn.functional.relu(self.fcbn1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    x = pt.nn.functional.relu(self.fcbn2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    #softmax is best for classification layer
    #soft = pt.nn.functional.softmax(tensor, dim = dimensions)
    return pt.nn.functional.softmax(x, dim = 1)



"""
def download_images(name_of_new_dictionary, path, img_list):

	#Purpose: downloads images into a list of images to feed into neural network
  #Extra Notes: have to put quotations around path
  
  data_path = os.path.join(path,'*g')
  files = glob.glob(data_path)
  name_of_new_dictionary = []
  for f1 in img_list:
      img = cv2.imread(path)
      name_of_new_dictionary.append(img)

"""
#Computing Metrics
def accuracy(out, labels):
  #finds accuracy of a neural network
  #changed from outputs = np.argmax(out, axis=1)
  return np.sum(out==labels)/float(labels.size)

#metrics dictionary
metrics = {'accuracy': accuracy}

#state dictionary
state = {}

class AverageBase(object):
    
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
       
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    
    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value


def numberize_labels(list_name):
  #converts filename list into list of numbers
  for i in range(len(list_name)):
    if list_name[i].startswith('v'):
      list_name[i] = 2
    elif list_name[i].startswith('mi'):
    	list_name[i] = 3
    elif list_name[i].startswith('mo'):
    	list_name[i] = 4
    elif list_name[i].startswith('n'):
    	list_name[i] = 1
    else:
    	list_name.remove(list_name[i])

#Saves the weights and biases of best model
def save_checkpoint(optimizer, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    pt.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = pt.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

def loss_fn(outputs, labels):
  num_examples = outputs.size()[0]
  return -pt.sum(outputs[range(num_examples), labels])/num_examples


if __name__ == "__main__":



  # DEFINING THE MODEL with 4 output bins
  model = TwoLayerNet(4)

  #Sending model to GPU
  #model.to(device)

  #Optimizer
  #pt.optim.Adam(parameters,... optional)
  #optional: lr (learning rate), betas, weight_decay, momentum
  #other good optimizers: SGD, Adagrad
  params = list(model.parameters())

  #Defining the optimizer
  optimizer = pt.optim.Adam(params)

  #Defining parameters for Dataloader----
  #Number of subprocesses to use for data loading
  num_workers = 4

  #Number of samples to load per batch
  batch_size = 100

  #Define transformations that will be applied to images
  image_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.Grayscale(1), #making the image gray scale so there is only one channel
                                        transforms.ToTensor()])

  #Define the datasets ----
  train_data = datasets.ImageFolder(root= '/Users/christopheradolphe/Desktop/WesternAI/Kaggle/Alzheimer_s Dataset/train', transform=image_transforms)
  #test_data = datasets.ImageFolder(root= "/content/drive/My Drive/Kaggle/Alzheimer_s Dataset/test", transform=image_transforms)

  #Obtain indices that will be used for validation ----
  num_train = len(train_data)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(0.2143 * num_train))
  split2 = 900
  train_idx, valid_idx = indices[split:], indices[:split]
  train_idx, test_idx = indices[split2:], indices[:split2]


  #Define samplers for obtaining training and validation batches
  train_sampler = t.SubsetRandomSampler(train_idx)
  valid_sampler = t.SubsetRandomSampler(valid_idx)
  test_sampler = t.SubsetRandomSampler(test_idx)

  #Prepare data loaders (combine dataset with sampler) ----
  train_loader = pt.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = train_sampler, 
                                            num_workers = num_workers)
  valid_loader = pt.utils.data.DataLoader(train_data,
                                            batch_size = batch_size, 
                                            sampler = valid_sampler, 
                                            num_workers = num_workers)
  test_loader = pt.utils.data.DataLoader(train_data,
                                            #shuffle = True,
                                            sampler = test_sampler,
                                            batch_size = batch_size, 
                                            num_workers = num_workers)

  #specify the image classes ----
  classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

  #Making the data and target values for train
  dataiter = iter(train_loader)
  data, target = dataiter.next()


  print("Set up Complete")
    

  #Initializing train_losses for graph
  train_losses = []
  train_accuracies = []
  valid_accuracies = []


  #Establishing number of epochs
  for i in range(0,10):

    #telling model that it is in training mode so it can change weights, biases and dropouts
    model.train()

    #Classifying epoch value
    epoch = i

    #Calculating train_loss as a Moving Average for visual depiction
    train_loss = MovingAverage()
    train_accuracy = MovingAverage()
    valid_accuracy = MovingAverage()
    
    #Training Loop
    for data, target in train_loader:
      
      #Converting the types of data and target values
      data, target = data.type(pt.FloatTensor), target.type(pt.LongTensor)

      # Move the training data and targets to the GPU
      #data = data.to(device)
      #target = target.to(device)

      #Zeroing the gradients of the model weights prior to backprop
      optimizer.zero_grad()

      #Forward pass
      output = model(data)
      
      #Making a tensor of the indicies of the most activated neurons in output layer
      #pt.max(tensor, dimension to reduce)
      predicted, predicted_idx = pt.max(output,1)
      


      #Compute the loss for the batch
      loss = loss_fn(output, target)

      #Backprop through loss
      loss.backward() 

      #Backprop optimizing biases and weights
      optimizer.step() #backprop step 2

      #updating train_loss
      train_loss.update(loss)

      #Sending predictions and targets to CPU for accuracy calculation
      #target = target.to(device2)
      #predicted_idx = predicted_idx.to(device2)


      #Convert Pytorch variables to numpy array
      predicted_idx = predicted_idx.data.numpy()
      target = target.data.numpy()

      train_accuracy.update(np.sum(predicted_idx==target)/float(target.size))
      


      #print("predicted:", predicted_idx)
      #print("target:", target)

      # accuracy relative to the most activated bin 
      print("accuracy:", accuracy(predicted_idx, target))

      #save_checkpoint()

      print('epoch:', i)

    print('Training loss:', train_loss)
    train_losses.append(train_loss.value)
    train_accuracies.append(train_accuracy)

    print("Training complete for epoch ", i)

        
    #Making the validation data and labels
    dataiter = iter(valid_loader)
    vdata, vtarget = dataiter.next()


    #Loading the model biases and weights 
    #model.load_state_dict(pt.load('/content/checkpoints/mnist-004.pkl'))
    
    #Validation Loop
    for vdata, vtarget in valid_loader:

      #Putting model into evaluation mode so that weights and biases remain the same
      model.eval()

      # Move the validation images and targetsto the GPU
      #vdata = vdata.to(device)
      #vtarget = vtarget.to(device)

      
      voutput = model(vdata) #forward pass


      #pt.max(tensor, dimension to reduce)
      vpredicted, vpredicted_idx = pt.max(voutput,1) #predicted is the most activated output (the one is the dimension to reduce)


      vloss = loss_fn(voutput, vtarget) #compute the loss for the batch

      #Sending GPU values to CPU
      #vtarget = vtarget.to(device2)
      #vpredicted_idx = vpredicted_idx.to(device2)


      #Convert Pytorch variables to numpy array
      vpredicted_idx = vpredicted_idx.data.numpy()
      vtarget = vtarget.data.numpy()

      valid_accuracy.update(np.sum(vpredicted_idx==vtarget)/float(vtarget.size))


      #Printing accuracy of the most activated bin to the target 
      print("accuracy:", accuracy(vpredicted_idx, vtarget))

    valid_accuracies.append(valid_accuracy)



   
#Making the test data and labels
    dataiter = iter(test_loader)
    tdata, ttarget = dataiter.next()
    print("test begin")


    model.eval()
    
    #Test Loop
    for tdata, ttarget in test_loader:

        # Move the test images and targetsto the GPU
      #tdata = tdata.to(device)
      #ttarget = ttarget.to(device)

        
      toutput = model(tdata) #forward pass


        #pt.max(tensor, dimension to reduce)
      tpredicted, tpredicted_idx = pt.max(toutput,1) #predicted is the most activated output (the one is the dimension to reduce)


      tloss = loss_fn(toutput, ttarget) #compute the loss for the batch


      #Sending GPU values to CPU
      #ttarget = ttarget.to(device2)
      #tpredicted_idx = tpredicted_idx.to(device2)


        #Convert Pytorch variables to numpy array
      tpredicted_idx = tpredicted_idx.data.numpy()
      ttarget = ttarget.data.numpy()

      print("predicted:", tpredicted_idx)
      print("target:", ttarget)


    #Printing accuracy of the most activated bin to the target 
      print("accuracy:", accuracy(tpredicted_idx, ttarget))



     
  #Save model
  model_filename = 'checkpoints/final_alzeimers_model.h5'
  pt.save(model, model_filename)
  
  #Save a checkpoint
  checkpoint_filename = 'checkpoints/mnist-{:03d}.pkl'.format(epoch)
  pt.save(model.state_dict(), checkpoint_filename)



  #Graphing the Model (Loss)
  epoch = range(1, len(train_losses) + 1)

  plt.figure(figsize=(10,6))
  plt.plot(epoch, train_losses, '-o', label='Training loss')
  plt.legend()
  plt.title('Learning curves')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.xticks(epoch)
  plt.show()

  #Graphing the Model (Accuracy)
  epoch = range(1, len(train_accuracies) + 1)

  plt.figure(figsize=(10,6))
  plt.plot(epoch, train_accuracies, '-o', label='Train Accuracy')
  plt.plot(epoch, valid_accuracies, '-o', label='Validation Accuracy')
  plt.legend()
  plt.title('Learning Curve')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.xticks(epoch)
  plt.show()