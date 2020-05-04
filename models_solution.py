## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #the size of input layer is 224*224 which changes to 222*222 after 1st convolution layer
        
        self.conv1 = nn.Conv2d(1, 16, 5)
        #The pooling step decreases the size further to 111*111, as it has a kernel size of 2*2 with stride 2 pixels
        self.BN1 = nn.BatchNorm2d(16)
        
        #Next in the second layer it changes from 110*110 to 106*106 and then to 53*53
        self.conv2 = nn.Conv2d(16,32,5)
        self.BN2 = nn.BatchNorm2d(32)
        
        # Now to 49*49 and then 24*24
        self.conv3 = nn.Conv2d(32,64,5)
        self.BN3 = nn.BatchNorm2d(64)
        
        # Now to 22*22 further to 11*11
        self.conv4 = nn.Conv2d(64,128,3)
        self.BN4 = nn.BatchNorm2d(128)
        
        #Now to 9*9 further to 4*4
        self.conv5 = nn.Conv2d(128,256,3)
        self.BN5 = nn.BatchNorm2d(256)
        
        #this layer takes the flattened input of 256*4*4 = 4096 and gives 2048 nodes of fully connected layer
        
        self.fc1 = nn.Linear(256*4*4,2048)
        
        #self.fc2 = nn.Linear(4096,2048)
        #self.fc2_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(2048,1024)
        
        self.fc3 = nn.Linear(1024,256)
        
        self.fc4 = nn.Linear(256,136)
        
        #pooling layer
        self.pool = nn.MaxPool2d(2,2)
        
        #dropout layers
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop5(x)
        
        x = x.view(x.size(0),-1)
        
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = F.relu(self.fc2(x))
        x = self.drop4(x)
        x = F.relu(self.fc3(x))
        x = self.drop5(x)
        #x = F.relu(self.fc4(x))
        #x = self.fc4_drop(x)
        x = self.fc4(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
