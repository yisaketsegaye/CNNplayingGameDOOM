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
    
        
# Making the body 
        
class SoftmaxBody(nn.Module):
    
    def __init(self, T):
        super(SoftmaxBody,self).__init__()
        self.T = T
    
    def forward(self, outputs):
       probs = F.softmax(outputs * self.T)
       actions = probs.multinomial()
       return actions
   
    
    
# making the AI
       
class AI:
    
    def __init__(self, brain , body):
        self.brain = brain
        self.body = body
    
    def __call__(self, inputs):
        input= Variable(torch.from.numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()
    
    
# part 2 - Training the AI with the Deep Convolutional Q-Learning
        
#Getting the Doom invironment 
        
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n
     
#Building an AI


cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up experiance Replay
n_steps = experiance_replay.NStepProgress(env = doom_env, ai = ai , n_step =10)
memory = experiance_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

# Implementing Eligiblity Trace

def eligiblity_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array(series[0].state , series[-1].state , dtype = np.float32)))
        output = cnn(input)
        cum_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cuml_reward = step.reward +gamma * cuml_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cum_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

 
















