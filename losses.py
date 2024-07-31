import torch
from torch import nn
import torch.nn.functional as F


input_reg = torch.randn(3, 5, requires_grad=True)
target_reg = torch.randn(3, 5)

input_class = torch.randn(3, 5, requires_grad=True) # 3 tensors, 5 classes
target_class = torch.tensor([1, 0, 4])

input_binary_class = torch.tensor([0.5, 0.3, 0.9], requires_grad=True)
target_binary_class = torch.tensor([1, 0, 1]).float()

# MAE, used for regression tasks where MSE is too rough 
loss = nn.L1Loss()
output = loss(input_reg, target_reg)
output

# MSE, used for regression
loss = nn.MSELoss()
output = loss(input_reg, target_reg)
output

# negative log likelihood loss used only after Softmax layer in multi-class classification tasks
loss = nn.NLLLoss() 
m = nn.LogSoftmax(dim=-1)
output = loss(m(input_class), target_class)
output

# corss-entropy loss, used for categorical classification tasks 
loss = nn.CrossEntropyLoss()
output = loss(input_class, target_class)
output

# binary corss-entropy loss, used for binary classification tasks 
loss = nn.BCELoss()
output = loss(input_binary_class, target_binary_class)
output

# Hinge Embedding Loss, used to learn embeddings 
# or in classification problems when determining if two inputs are dissimilar or similar
loss = nn.HingeEmbeddingLoss()
output = loss(input_reg, target_reg)
output

# Ranking Loss, used in ranking problems
input_one = torch.randn(3, requires_grad=True)
input_two = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()

loss = nn.MarginRankingLoss()
output = loss(input_one, input_two, target)
output

# KL Divergence, used for approximating functions, Multi-class classification tasks
loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
target = F.softmax(torch.rand(3, 5), dim=1)
output = loss(input, target)
output