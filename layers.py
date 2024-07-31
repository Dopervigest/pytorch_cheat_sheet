import torch
from torch import nn

####### Neural Network layers #######
x = torch.randn([100, 1, 64, 64])
# Conv1d, Conv2d, Conv3d
layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,2), 
                  stride=1, padding=0, 
                  bias=True, padding_mode='zeros', device=None, dtype=torch.float32)

out = layer(x)

x = torch.randn([100, 4096])
# Linear 
layer = nn.Linear(in_features=4096, out_features=32, bias=True, device=None, dtype=torch.float32)

out = layer(x)

# LSTM
batch = 5
channels = 1
input_size = 3
num_layers = 2
hidden_size = 4

inputs = torch.randn([batch, channels, input_size]) 
hidden = (torch.randn(num_layers, channels, hidden_size), 
          torch.randn(num_layers, channels, hidden_size))


lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers=num_layers, dropout=0)  

out, hidden = lstm(inputs, hidden)


###### Activation function layers ######

relu = nn.ReLU() # best to use between layers

lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False) # replacement for relu when it outputs only 0

elu = nn.ELU(alpha=1.0, inplace=False) # ReLU replacement that helps with faster convergence 

sigmoid = nn.Sigmoid() # output layer for binary classification or normalization of outputs 

tanh = nn.Tanh() # best to use between layers in RNNs and tasks where negative values are meaningful

softmax = nn.Softmax(dim=-1) # output layer for Multi-Class Classification and hidden in NLP models

###### Data manipulation layers ######

# nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d 
maxpool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0) # takes max kernel_size elements

# nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d use after maxpooling or activation
norm = nn.BatchNorm1d(num_features=1, eps=1e-05, momentum=0.1) # takes channels as num_features from previous layer

dropout = nn.Dropout(p=0.5) 

flatten = nn.Flatten() # torch.squeeze()