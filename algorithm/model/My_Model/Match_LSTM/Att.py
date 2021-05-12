import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    '''
    Attention Layer
    '''

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        '''
        :param x: (batch_size, max_len, hidden_size)
        :return alpha: (batch_size, max_len)
        '''
        x = torch.tanh(x)  # (batch_size, max_len, hidden_size)
        x = self.attn(x).squeeze(2)  # (batch_size, max_len)
        alpha = F.softmax(x, dim=1).unsqueeze(1)  # (batch_size, 1, max_len)
        return alpha