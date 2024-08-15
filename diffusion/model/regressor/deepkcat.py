import os
import argparse
import math
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from collections import defaultdict
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import scipy.sparse as sparse
from input_dataset import Brenda
from GCN import GCN
from protein_transformer import TransformerBlock as protein_transformer
from utils import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition, BlosumTransition

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DeepKcat(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, 
                 layer_output, hidden_dim1, hidden_dim2, dropout, 
                 nhead, hid_size, layers_trans,
                 timesteps=500, noise_type='blosum',
                 time_dim=20,
                 ):
        super(DeepKcat, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_wordGCN = nn.Embedding(n_word, dim)
        self.embed_wordTrans = nn.Embedding(n_word, hid_size)
        self.W_out = nn.ModuleList([nn.Linear(4 * dim, 4 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(4 * dim, 1)
        self.gcn = GCN(dim, hidden_dim1, hidden_dim2, dim, nhead, dropout)
        self.dropout = nn.Dropout(dropout)
        self.smiles_transformer = protein_transformer(nhead, dropout, dim, hid_size, layers_trans, max_len=n_fingerprint)
        self.protein_transformer = protein_transformer(nhead, dropout, dim, hid_size, layers_trans, max_len=n_word)
        self.softmax = nn.Softmax(dim=1)
        self.ELU = nn.ELU(1.0)
        self.timesteps = timesteps
        self.embed_time = nn.Linear(1, time_dim)
        self.noise_type = noise_type
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine',timesteps=self.timesteps,noise_type='blosum')
        self.transition_model = BlosumTransition(x_classes=20)
        
    def fingerprint_gcn(self, smileadjacency, fingerprints, dropout):
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        substrate_vectors = self.gcn(fingerprint_vectors, smileadjacency)
        return substrate_vectors

    def seq_transformer(self, words, dropout):
        words = words.unsqueeze(0)
        words = self.embed_wordTrans(words)
        seq_vectors = self.protein_transformer(words)
        return seq_vectors

    def protein_gcn(self, adjacency, word_vectors):
        #adjacency = torch.tensor(adjacency).cuda()
        #word_vectors = self.embed_wordGCN(words)
        protein_vectors = self.gcn(word_vectors, adjacency)
        return protein_vectors
    
    def apply_noise(self, batch, x, t_float):
        if self.noise_type== 'uniform':
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t_float, device=x.device)
        prob_X = (Qtb[batch]@x[:,:20].unsqueeze(2)).squeeze()
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t, num_classes = 20)
        return noise_X
    
    def forward(self, inputs, layer_output, dropout, batch, x, edge_index, t=None):
        if t is None:
            t = np.random.randint(1, self.timesteps)
        else:
            t = t.item()
        t_float = t / self.timesteps
        t_float = torch.tensor([[t_float]]).cuda()
        X = self.apply_noise(batch, x, t_float)
        X = X.float()
        #graph.edge_index = graph.edge_index.float()
        time_emb = self.embed_time(t_float)
        
        fingerprints, smileadjacency, words, seqadjacency = inputs
        substrate_vectors = self.fingerprint_gcn(smileadjacency, fingerprints, dropout)
        substrate_vectors = torch.unsqueeze(torch.mean(substrate_vectors, 0), 0)
        
        seq_vectors = self.seq_transformer(words, dropout)
        seq_vectors = torch.unsqueeze(torch.mean(seq_vectors, 0), 0)

        # graph edge to adj tensor
        words = X.cuda()
        edge = edge_index #[2, nums]
        n = words.shape[0]
        adj = torch.zeros([n, n]).cuda()

        for index in range(edge.shape[1]):
            adj[edge[0, index].item(), edge[1, index].item()] = 1
        
        #protein_vectors = self.protein_gcn(seqadjacency, words, dropout)
        protein_vectors = self.protein_gcn(adj, words)
        #([n,n], [n])
        protein_vectors = torch.unsqueeze(torch.mean(protein_vectors, 0), 0)

        cat_vector = torch.cat((substrate_vectors, protein_vectors, seq_vectors, time_emb), 1)
        ori_vector = cat_vector.clone()

        for j in range(layer_output):
            cat_vector = F.relu(cat_vector)
            cat_vector = F.dropout(cat_vector, dropout, training=self.training)
            cat_vector = self.W_out[j](cat_vector)

        cat_vector = F.relu(cat_vector)
        cat_vector = F.dropout(cat_vector, dropout, training=self.training)

        return cat_vector, seq_vectors
