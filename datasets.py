from torch.utils.data import Dataset, DataLoader
from torch import tensor, float32

import os
import pickle
import torch
import numpy as np

from utils import pcc_to_adjacency

class DatasetMDD(Dataset):
    def __init__(self,type, k, threshold=None):
        super().__init__()
        
        assert type in ["train", "test"]
        assert k in range(1,6)
        
        with open(os.path.join("./data",f"mdd_fold{k}.pkl"), "rb") as f:
            temp = pickle.load(f)
        self.data = temp[type]
        self.pcc = self.data['pcc'] 
        if threshold is not None:
            self.adj = pcc_to_adjacency(torch.tensor(self.data['pcc'], dtype=torch.float), threshold)
        else:
            self.adj = self.pcc.copy()
        self.non_fisher = (np.exp(2 * self.data['pcc']) - 1) / (np.exp(2 * self.data['pcc']) + 1)
        if threshold is not None:
            self.non_fisher_adj = pcc_to_adjacency(torch.tensor(self.non_fisher, dtype=torch.float), threshold)
        else:
            self.non_fisher_adj = self.pcc.copy()
        self.label = self.data['label'] 
        _, self.num_node, self.time_length = self.data['bold'].shape
        self.weight = [1-np.sum(self.label==0)/len(self.label), 1-np.sum(self.label==1)/len(self.label)]

    
    def __len__(self):
        if self.pcc_use:
            return len(self.data)
        return self.data['bold'].shape[0]


    def __getitem__(self, idx):
        bold = self.data['bold'][idx]
        pcc = self.pcc[idx]
        non_fisher = self.non_fisher[idx]
        non_fisher_adj = self.non_fisher_adj[idx]
        adj = self.adj[idx]
        label = self.label[idx]


        return {'bold': tensor(bold, dtype=float32), 
                'pcc': tensor(pcc, dtype=float32),
                'non_fisher': tensor(non_fisher, dtype=float32),
                'non_fisher_adj': tensor(non_fisher_adj, dtype=float32),
                'adj':tensor(adj, dtype=float32), 
                'label': tensor(label, dtype=int)}
