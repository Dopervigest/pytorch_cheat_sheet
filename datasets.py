# Data preparation

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


x_train, x_test = torch.randn([100, 1, 64, 64]), torch.randn([50, 1, 64, 64])
y_train, y_test = torch.randint(0, 5, [100]), torch.randint(0, 5, [50,])

y_train = F.one_hot(y_train, num_classes=5)
y_test = F.one_hot(y_test, num_classes=5)


###### custom dataset template ######

class Data(Dataset):
    def __init__(self, x, y):
        super(Data, self).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self, index):            
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

train_dataset = Data(x_train, y_train)
test_dataset = Data(x_test, y_test)


###### dataset form tensors ######
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

########### hdf5 dataset #########

import os
import h5py

# Writing
dirr = './'
with h5py.File(os.path.join(dirr, 'train.hdf5'), 'w') as f:
    f.create_dataset('data', data=x_train)
    f.create_dataset('labels', data=y_train)
with h5py.File(os.path.join(dirr, 'test.hdf5'), 'w') as f:
    f.create_dataset('data', data=x_test)
    f.create_dataset('labels', data=y_test)

# Reading
class H5Dataset(Dataset):
    def __init__(self, file_path):
        # opening hdf5 here breaks multithreading 
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f['data'])
    
    def open_hdf5(self):
        self.file = h5py.File(self.file_path, 'r')
        self.data = self.file.get('data') 
        self.labels = self.file.get('labels')
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()
            
    def close(self):
        if hasattr(self, 'file'):
            self.file.close()
        

    def __getitem__(self, item: int):
        if not hasattr(self, 'file'):
            self.open_hdf5()
        
        x = torch.from_numpy(self.data[item])
        y = torch.from_numpy(self.labels[item])
        
        return x, y
    
train_dataset = H5Dataset('./train.hdf5')
test_dataset = H5Dataset('./test.hdf5')



from torch.utils.data import DataLoader

BATCH_SIZE = 32
# if data is already on GPU, multiple workers cause CUDA initialization error
DATA_WORKERS = 0 if train_dataset[0][0].device.type == 'cuda' else 8

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATA_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATA_WORKERS)
