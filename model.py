import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # applies linear function 
                                            # uses formula y=xA^T+b
                                                # where x= input data(x)
                                                    #   A= weights shape(hidden_Size,input_size) initialized from u(-(k)^1-2,(k)^1/2), where k=1/input_size
                                                    #   b=bias shape(hidden_size) initialized from u(-(k)^1-2,(k)^1/2), where k=1/input_size
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()# applies rectified linear function
                                # uses formula (x)^+ = max(0,x)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
