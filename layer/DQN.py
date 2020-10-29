
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,state, action):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(state,128)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(128,128)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(128,action)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


