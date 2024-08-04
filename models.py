import torch
from torch import nn
import torch.nn.functional as F

n_classes = 5

in_features_conv = 1
in_features_linear = 64 * 64


x_conv = torch.randn([100, in_features_conv, 64, 64])
x_flat = x_conv.view([100, -1])

y = torch.randint(0, n_classes, [100]) 
y = F.one_hot(y, num_classes=n_classes)


x_conv.shape, x_flat.shape, y.shape

###### Sequential models ######
model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = (4,4), stride = (4,4)),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 4, stride = 4),
            nn.ReLU(),
            nn.Conv2d(64,5, kernel_size = 4, stride = 1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
)
model(x_conv).shape


class Sequential_module(nn.Module): 
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size = (4,4), stride = (4,4)),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=4, stride=None, padding=0),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1),
            
            nn.Conv2d(32, n_classes, kernel_size = 4, stride = 1),
            nn.BatchNorm2d(num_features=n_classes),
            nn.Flatten(),
            
            nn.Softmax(dim=-1)
        )
    
    def forward(self, data):
        return self.network(data)
model = Sequential_module(in_features=in_features_conv, n_classes=n_classes)
model(x_conv).shape


class Simple_module_list(nn.Module):
    def __init__(self, in_features, n_classes, hidden_layers=2):
        super(Simple_module_list,self).__init__()
        
        self.module_list= nn.ModuleList()
        
        self.module_list.append(nn.Linear(in_features, 256))
        for _ in range(n_classes):
            self.module_list.append(nn.Linear(256,256))
            
        self.module_list.append(nn.Linear(256, n_classes))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self,x):
        return self.ff(x)
    
    def ff(self, x):
        for module in self.module_list:
            x = module(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.softmax(x)
        return x
model = Simple_module_list(in_features=in_features_linear,
                      n_classes=n_classes,hidden_layers = 2)
model(x_flat).shape


###### Simple models ######
class Simple_module(nn.Module): 
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=32, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) 
        self.linear2 = nn.Linear(in_features=32, out_features=n_classes, dtype=torch.float32)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)      
        
        return x
model = Simple_module(in_features= in_features_linear,
                      n_classes=n_classes)
model(x_flat).shape


class Several_inputs_module(nn.Module):
    def __init__(self, in_features, n_classes):
        super(Several_inputs_module,self).__init__()
        
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(in_features, 256)
        self.linear3 = nn.Linear(512, n_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self,x1, x2):
        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        
        x2 = self.linear2(x2)
        x2 = self.relu(x2)
        
        x = torch.cat([x1,x2], dim=-1) # use concatenation or torch.add(x1,x2)
        x = self.linear3(x)
        x = self.softmax(x)
        
        return x
model = Several_inputs_module(in_features=in_features_linear,
                      n_classes=n_classes)
model(x_flat, x_flat).shape


###### freezing selected layers ######
model = Sequential_module(in_features=in_features_conv, n_classes=n_classes)
for i in range(len(model.network)):
    layer = model.network[i]
    if type(layer) == torch.nn.modules.conv.Conv2d and layer.in_channels != n_classes:
        for param in model.network[i].parameters():
            param.requires_grad = False

for i in iter(model.network.parameters()):
    print(i.requires_grad)


###### freezing all layers and changing head ######
model = Several_inputs_module(in_features=in_features_conv, n_classes=n_classes)
for param in model.parameters():
    param.requires_grad = False
    
model.linear3 = nn.Linear(512, n_classes)

for i in iter(model.parameters()):
    print(i.requires_grad)


###### transfering selected weights for transfer learning ######
model_1 = Sequential_module(in_features=in_features_conv, n_classes=n_classes)
model_2 = Sequential_module(in_features=in_features_conv, n_classes=n_classes)

for i in range(len(model_1.network)):
    layer = model_1.network[i]
    if type(layer) == torch.nn.Conv2d:
        model_2.network[i].load_state_dict(layer.state_dict())

torch.equal(next(model_2.network.parameters()), next(model_1.network.parameters()))



###### custom pytorch layer ######
class custom_layer(torch.nn.Conv2d):
    def __init__(self, symmetry, dim, *kargs, **kwargs): 
        super().__init__(*kargs, **kwargs) 
        self.symmetry = symmetry 
        self.dim = dim

    def forward(self, input): 
        self.new_weights = self.symmetry(self.weight, self.dim)
        return F.conv2d(input, self.new_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

    
def symmetry(weights: torch.tensor, dim: tuple):
    new_weights = (weights + weights.transpose(dim[0], dim[1]))/2
    return new_weights
    
layer = custom_layer(symmetry, dim=[2,3],
                     in_channels=1, out_channels=1, kernel_size=(4,4),stride=(1,1))
output = layer(torch.randn([1,1,8,8]))

layer.new_weights # symmetry along main diagonal



###### training profiling ######
from torch.profiler import profile, record_function, ProfilerActivity
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as p:
    trainer.fit()
print(p.key_averages().table(sort_by="cpu_memory_usage", row_limit=50))