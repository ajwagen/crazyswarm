import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

class GainTune(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.flag = 1
        self.batch_size = batch_size
        # self.h = torch.randn(2, batch_size, 3)
        # self.rnn = nn.RNN(5, 3, 1, bidirectional=False, batch_first=True)
        
        self.linear1 = nn.Linear(5, 3)
        self.linear2 = nn.Linear(3, 1)


    def forward(self, x):
        x = x.float()
        # h = self.init_hidden()

        # x,h = self.rnn(x,h)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    # def init_hidden(self,):
    #     # This method generates the first hidden state of zeros which we'll use in the forward pass
    #     # We'll send the tensor holding the hidden state to the device we specified earlier as well
    #     hidden = torch.zeros(1, self.batch_size, 3)
    #     return hidden

p = GainTune()

a = torch.tensor(np.random.rand(5,4,5))

op = p(a)