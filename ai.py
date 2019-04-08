# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing







#part1- Building the AI

# Making the brain

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN,self).__init__()
        self.convolutionl1 = nn.Conv2d(in_channels = 1, out_channel = 32 , kernel_size = 5)
        self.convolutionl2 = nn.Conv2d(in_channels = 32, out_channel = 32 , kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 1, out_channel = 64 , kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons(1,80,80) , out_features = 40 )
        self.fc2 = nn.Linear(in_features = 40  , out_features = number_actions )
        
        


    def count_neurons(self, image_dim):
        x= Variable(torch.rand(1, *image_dim))
        x= F.relu(F.max_pool2d(self.convolution1(x), 3,2))
        x= F.relu(F.max_pool2d(self.convolution2(x), 3,2))
        x= F.relu(F.max_pool2d(self.convolution3(x), 3,2))
        return x.data.view(1,-1).size(1)
    
    def forward(self,x):
        x= F.relu(F.max_pool2d(self.convolution1(x), 3,2))
        x= F.relu(F.max_pool2d(self.convolution2(x), 3,2))
        x= F.relu(F.max_pool2d(self.convolution3(x), 3,2))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
        
        
    

 
















