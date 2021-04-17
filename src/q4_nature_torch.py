import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from utils.general import get_logger
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear
import copy
from torchsummary import summary

from configs.q4_nature import config
class network(nn.Module):
    def __init__(self, env):
        super(network, self).__init__()
        self.env = env
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        #print("Num actions " + str(num_actions))
        strides = np.array([4, 2, 1])  # The stride size for every conv2d layer
        filter_sizes = np.array([8, 4, 3])  # The filter size for every conv2d layer
        numb_filters = np.array([32, 64, 64])  # number of filters for every conv2d layer
        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ##########

        pad1 = ((strides[0] - 1) * img_height - strides[0] + filter_sizes[0]) // 2
        pad2 = ((strides[1] - 1) * img_height - strides[1] + filter_sizes[1]) // 2
        pad3 = ((strides[2] - 1) * img_height - strides[2] + filter_sizes[2]) // 2
        self.net = nn.Sequential(
            nn.Conv2d(n_channels*4, 32, kernel_size=8, stride=4, padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=pad2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=pad3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_width*img_height*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            #print(x.size())
        return x


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history
        MAKE URE YOU USE THESE VARIABLES FOR INPUT SIZE

        Each network has the following architecture (see th nature paper for more details):

            - Conv2d with 32 8x8 filters and stride 4 + ReLU activation
            - Conv2d with 64 4x4 filters and stride 2 + ReLU activation
            - Conv2d with 64 3x3 filters and stride 1 + ReLU activation
            - Flatten
            - Linear with output 512. What is the size of the input?
                you need to calculate this img_height, img_width, and number of filter.
            - Relu
            - Linear with 512 input and num_actions outputs

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            ((stride - 1) * img_height - stride + filter_size) // 2
        Make sure you follow use this padding for every layer

        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        strides = np.array([4, 2, 1])  # The stride size for every conv2d layer
        filter_sizes = np.array([8, 4, 3])  # The filter size for every conv2d layer
        numb_filters = np.array([32, 64, 64])  # number of filters for every conv2d layer
        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ##########

        pad1 = ((strides[0] - 1) * img_height - strides[0] + filter_sizes[0]) // 2
        pad2 = ((strides[1] - 1) * img_height - strides[1] + filter_sizes[1]) // 2
        pad3 = ((strides[2] - 1) * img_height - strides[2] + filter_sizes[2]) // 2
        self.q_network = network(self.env)
        self.target_network = network(self.env)
        '''self.q_network = nn.Sequential(
            nn.Conv2d(n_channels*self.config.state_history, 32, kernel_size=8, stride=4, padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=pad2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=pad3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_width*img_height*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.target_network = nn.Sequential(
            nn.Conv2d(n_channels * self.config.state_history, 32, kernel_size=8, stride=4, padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=pad2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=pad3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_width*img_height*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )'''

        ##############################################################
        ######################## END YOUR CODE ####################### 

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        state = torch.transpose(state, dim0=1, dim1=-1)
        #print(state)
        #print(self.q_network)
        #print(self.target_network)
        if network == 'q_network':
            out = self.q_network(state)
        else:
            out = self.target_network(state)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
