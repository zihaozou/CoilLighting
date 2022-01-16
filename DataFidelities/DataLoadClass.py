import torch
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self, train_ipt:torch.Tensor, 
                       train_gdt:torch.Tensor, 
                       train_y:torch.Tensor,
                       train_theta:torch.Tensor,                                        
                ):
          
        super(Dataset, self).__init__()
        self.train_ipt = train_ipt
        self.train_gdt = train_gdt
        self.train_y = train_y
        self.train_theta = train_theta

    def __len__(self):
        return self.train_gdt.shape[0]

    def __getitem__(self, item):
        return self.train_ipt[item], self.train_gdt[item], self.train_y[item], self.train_theta[item]

class ValidDataset(Dataset):

    def __init__(self, valid_ipt:torch.Tensor, 
                       valid_gdt:torch.Tensor, 
                       valid_y:torch.Tensor,
                       valid_mask:torch.Tensor,
                       valid_theta:torch.Tensor,           
                       ):
        super(Dataset, self).__init__()
        self.valid_ipt = valid_ipt
        self.valid_gdt = valid_gdt
        self.valid_y = valid_y
        self.valid_mask = valid_mask
        self.valid_theta = valid_theta

    def __len__(self):
        return self.valid_gdt.shape[0]

    def __getitem__(self, item):
        return self.valid_ipt[item], self.valid_gdt[item], self.valid_y[item], self.valid_mask[item], self.valid_theta[item]