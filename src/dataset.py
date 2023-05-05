import pickle
from torch.utils import data
import torch
class EventDataset(data.Dataset):
    def __init__(self,trian_instances):
        self.data = trian_instances
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        inputs = sample[0]
        label = -1
        if sample[1]=="SuperSub":
            label = 0
        elif sample[1]=="SubSuper":
            label = 1
        elif sample[1]=="Coref":
            label = 2
        else:
            label = 3
        assert label != -1
        return torch.tensor(inputs["input_ids"]),torch.tensor(inputs["attention_mask"]),torch.tensor(inputs["token_type_ids"]),label
