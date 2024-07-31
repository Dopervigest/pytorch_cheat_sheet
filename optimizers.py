import torch
from torch import nn

model = nn.Linear(1, 32)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

opt = torch.optim.Adagrad(model.parameters(), lr=0.001, lr_decay=0, weight_decay=0)

opt = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)