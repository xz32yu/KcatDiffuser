import os
import random
import pickle
import torch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.data import Dataset, download_url, Batch, Data

class Brenda(Dataset):
    def __init__(self, list_IDs, baseDIR,transform=None, 
                 pre_transform=None, pre_filter=None,pred_sasa = False, 
                 ori_data = None
                 ):
        super().__init__(baseDIR, transform, pre_transform, pre_filter)
        self.list_IDs = list_IDs
        self.baseDIR = baseDIR
        self.pred_sasa = pred_sasa
        self.fingerprint, self.smileadjacencies, self.sequences, self.proteinadjacencies, self.kcat_tensor = ori_data

    def len(self):
        return len(self.list_IDs)

    def get(self, index):
        ID = self.list_IDs[index]
        data = torch.load(self.baseDIR+ID)
        del data['distances']
        del data['edge_dist']
        mu_r_norm = data.mu_r_norm
        extra_x_feature = torch.cat([data.x[:,20:],mu_r_norm],dim = 1)
        graph = Data(
            x = data.x[:, :20],
            extra_x = extra_x_feature,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ss = data.ss[:data.x.shape[0],:],
            sasa = data.x[:,20]
        )
        return (self.fingerprint[index], self.smileadjacencies[index], self.sequences[index], self.proteinadjacencies[index]), graph, self.kcat_tensor[index]
    