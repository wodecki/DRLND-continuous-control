import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import *

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, \
    a_dropout, a_leaky, \
    fc1_units, fc2_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units

        self.a_dropout = a_dropout
        self.a_leaky = a_leaky


        self.fc1 = nn.Linear(state_size, fc1_units)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        if a_dropout:
            self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, a_dropout, a_leaky):
        """Build an actor (policy) network that maps states -> actions."""

        if a_leaky:
            x = F.leaky_relu(self.fc1(state))
            x = F.leaky_relu(self.fc2(x))
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))

        if a_dropout:
            x = self.dropout(x) #dropout layer

        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, \
    c_dropout, c_leaky, c_batch_norm, \
    fc1_units, fc2_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units

        self.c_dropout = c_dropout
        self.c_leaky = c_leaky
        self.c_batch_norm = c_batch_norm

        self.fc1 = nn.Linear(state_size, fc1_units)

        if c_batch_norm:
            self.bn1 = nn.BatchNorm1d(fc1_units) #batch normalization

        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        if c_dropout:
            self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action, c_dropout, c_leaky, c_batch_norm):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        if c_leaky:
            xs = F.leaky_relu(self.fc1(state))
            if c_batch_norm:
                xs = self.bn1(xs)
            x = torch.cat((xs, action), dim=1)
            x = F.leaky_relu(self.fc2(x))
        else:
            xs = F.relu(self.fc1(state))
            if c_batch_norm:
                xs = self.bn1(xs)
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))

        if c_dropout:
            x = self.dropout(x) #dropout layer

        return self.fc3(x)
