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
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine',timesteps=self.timesteps,noise_type='uniform')
        self.transition_model = DiscreteUniformTransition(x_classes=20)
        
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

    def apply_noise(self, data, t_float):
        
        if self.noise_type== 'uniform':
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t_float, device=data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:20].unsqueeze(2)).squeeze()
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t, num_classes = 20)
        noise_data = data.clone()
        noise_data.x = noise_X
        return noise_data
    
    def forward(self, inputs, layer_output, dropout, graph):
        t = np.random.randint(1, self.timesteps)
        t_float = t / self.timesteps
        t_float = torch.tensor([[t_float]]).cuda()
        graph = self.apply_noise(graph, t_float)
        graph.x = graph.x.float()
        #graph.edge_index = graph.edge_index.float()
        time_emb = self.embed_time(t_float)
        
        fingerprints, smileadjacency, words, seqadjacency = inputs
        substrate_vectors = self.fingerprint_gcn(smileadjacency, fingerprints, dropout)
        substrate_vectors = torch.unsqueeze(torch.mean(substrate_vectors, 0), 0)
        
        seq_vectors = self.seq_transformer(words, dropout)
        seq_vectors = torch.unsqueeze(torch.mean(seq_vectors, 0), 0)

        # graph edge to adj tensor
        words = graph.x.cuda()
        edge = graph.edge_index #[2, nums]
        n = words.shape[0]
        adj = torch.zeros([n, n]).cuda()

        for index in range(edge.shape[1]):
            adj[edge[0, index].item(), edge[1, index].item()] = 1
        
        #protein_vectors = self.protein_gcn(seqadjacency, words, dropout)
        protein_vectors = self.protein_gcn(adj, words)
        #([n,n], [n])
        protein_vectors = torch.unsqueeze(torch.mean(protein_vectors, 0), 0)

        cat_vector = torch.cat((substrate_vectors, protein_vectors, seq_vectors, time_emb), 1)

        for j in range(layer_output):
            cat_vector = F.relu(cat_vector)
            cat_vector = F.dropout(cat_vector, dropout, training=self.training)
            cat_vector = self.W_out[j](cat_vector)

        cat_vector = F.relu(cat_vector)
        cat_vector = F.dropout(cat_vector, dropout, training=self.training)
        interaction = self.W_interaction(cat_vector)
        interaction = torch.squeeze(interaction, 0)
        return interaction


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in load_pickle(file_name)]

def load_proteinadjacencies(file_name, dtype):
    return [dtype(d.toarray()).to(device) for d in load_pickle(file_name)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def split_sequence(sequence, ngram, word_dict):
    sequence = '--' + sequence + '='

    words = list()
    for i in range(len(sequence) - ngram + 1):
        try:
            words.append(word_dict[sequence[i:i + ngram]])
        except:
            word_dict[sequence[i:i + ngram]] = 0
            words.append(word_dict[sequence[i:i + ngram]])

    return np.array(words)


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                try:
                    fingerprints.append(fingerprint_dict[fingerprint])
                except:
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    try:
                        edge = edge_dict[(both_side, edge)]
                    except:
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def get_ca_coords(pdb):
    with open(pdb, 'r') as file:
        lines = file.readlines()
        file.close()

    out = []

    for line in lines:
        if line.startswith('ATOM ') and line.split()[4] == 'A' and line.split()[2] == 'CA':
            res_num = line.split()[5]
            res_name = line.split()[3]
            x = line.split()[6]
            y = line.split()[7]
            z = line.split()[8]
            if len(x) > int(8):
                x = line.split()[6][:-8]
                y = line.split()[6][-8:]
                z = line.split()[7]
            elif len(y) > int(8):
                x = line.split()[6]
                y = line.split()[7][:-8]
                z = line.split()[7][-8:]
            elif len(res_num) > int(4):
                x = line.split()[5][-8:]
                y = line.split()[6]
                z = line.split()[7]
                res_num = line.split()[5][:-8]

            out.append([res_num, res_name, x, y, z])

    df = pd.DataFrame(out, columns=['res_num', 'res_name', 'x', 'y', 'z'])

    return df


def luciferase_contact_map(pdb,seq):
    ca_coords = get_ca_coords(pdb)
    dist_arr = pairwise_distances(ca_coords[['x', 'y', 'z']].values)  # distance
    dist_tensor = torch.from_numpy(dist_arr)
    dist_thres = 10
    cont_arr = (dist_arr < dist_thres).astype(int)
    cont_tensor = torch.from_numpy(cont_arr)
    if cont_arr.shape[0] == len(seq):
        proteinadjacency = sparse.csr_matrix(cont_arr)
    else:
        a = np.zeros((cont_arr.shape[0], len(seq) - cont_arr.shape[0]))
        cont_arr = np.column_stack((cont_arr, a))
        b = np.zeros((len(seq) - cont_arr.shape[0], len(seq)))
        cont_arr = np.row_stack((cont_arr, b))
        row, col = np.diag_indices_from(cont_arr)
        cont_arr[row, col] = 1
        proteinadjacency = sparse.csr_matrix(cont_arr)
    return proteinadjacency


def main():
    train_data_dir =  './dataset/dp_train/train/'
    val_data_dir = './dataset/dp_train/validation/'
    parser = argparse.ArgumentParser()    
    parser.add_argument('--train_dir', type = str, default=train_data_dir, help='path of training data')
    parser.add_argument('--val_dir', type = str, default=val_data_dir, help='path of val data')
    args = parser.parse_args()
    config = vars(args)
    # load training dataset
    train_label_path = './diffusion/results/example/train/kcat.pickle'
    with open(train_label_path, 'rb') as f:
        train_kcat = pickle.load(f)
    # load validation dataset
    val_label_path = './diffusion/results/example/val/kcat.pickle'
    with open(val_label_path, 'rb') as f:
        val_kcat = pickle.load(f)
    train_ID,val_ID= os.listdir(config['train_dir']),os.listdir(config['val_dir'])
    #print(f"train_id: {len(train_ID)}")
    #print(f"val_id: {len(val_ID)}")

    
    dim = 20
    layer_output = 3
    hidden_dim1 = 20
    hidden_dim2 = 20
    dropout = 0
    nhead = 4
    hid_size = 64
    layers_trans = 3
    radius = 2
    ngram = 4

    atom_dict = load_pickle('./diffusion/results/example/train/atom_dict.pickle')
    bond_dict = load_pickle('./diffusion/results/example/train/bond_dict.pickle')
    edge_dict = load_pickle('./diffusion/results/example/train/edge_dict.pickle')
    

    fingerprint_dict = load_pickle('./diffusion/results/example/train/fingerprint_dict.pickle')
    word_dict = load_pickle('./diffusion/results/example/train/sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)


    # load training data
    fingerprints_dict = './diffusion/results/example/train/compounds.pickle'
    train_fingerprints = load_tensor(fingerprints_dict, torch.LongTensor)

    smilesadjacency_dict = './diffusion/results/example/train/smilesadjacencies.pickle'
    train_smilesadjacencys = load_tensor(smilesadjacency_dict, torch.FloatTensor)

    words_dict = './diffusion/results/example/train/proteins.pickle'
    train_words = load_tensor(words_dict, torch.LongTensor)

    proteinadjacencys_dict = './diffusion/results/example/train/proteinadjacencies.pickle'
    train_proteinadjacencys = load_proteinadjacencies(proteinadjacencys_dict, torch.FloatTensor)


    # load validation data
    fingerprints_dict = './diffusion/results/example/val/compounds.pickle'
    val_fingerprints = load_tensor(fingerprints_dict, torch.LongTensor)

    smilesadjacency_dict = './diffusion/results/example/val/smilesadjacencies.pickle'
    val_smilesadjacencys = load_tensor(smilesadjacency_dict, torch.FloatTensor)

    words_dict = './diffusion/results/example/val/proteins.pickle'
    val_words = load_tensor(words_dict, torch.LongTensor)

    proteinadjacencys_dict = './diffusion/results/example/val/proteinadjacencies.pickle'
    val_proteinadjacencys = load_proteinadjacencies(proteinadjacencys_dict, torch.FloatTensor)
    train_dataset = Cath(train_ID,config['train_dir'], ori_data=[train_fingerprints, train_smilesadjacencys, train_words, train_proteinadjacencys, train_kcat])
    val_dataset = Cath(val_ID,config['val_dir'], ori_data=[val_fingerprints, val_smilesadjacencys, val_words, val_proteinadjacencys, val_kcat])

    #import pdb;pdb.set_trace()
    model = DeepKcat(n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size,
                     layers_trans).to(device)
    
    # Training loop
    lr = 1e-9
    num_epochs = 20 #100
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    for epoch in range(num_epochs):
        model.train()
        for index in range(train_dataset.len()):
            tmp, graph, label = train_dataset.get(index)
            graph.x = graph.x.cuda()
            graph.edge_index = graph.edge_index.cuda()
            label = torch.tensor(label).float().cuda()
            optimizer.zero_grad()
            #import pdb;pdb.set_trace()
            inputs = [train_fingerprints[index], train_smilesadjacencys[index], train_words[index], train_proteinadjacencys[index]]
            kcat = model.forward(inputs, layer_output, dropout, graph)
            #print(f"train_epoch: {index}")
            loss = criterion(kcat, label)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for index in range(val_dataset.len()):
                tmp, graph, label = val_dataset.get(index)
                label = torch.tensor(label).cuda()
                
                inputs = [val_fingerprints[index], val_smilesadjacencys[index], val_words[index], val_proteinadjacencys[index]]
                kcat = model.forward(inputs, layer_output, dropout, graph)
                print(f"val_epoch: {index}")
                loss = criterion(kcat, label)
                val_loss += loss.item()
            val_loss = val_loss/val_dataset.len()
            print(f"Epoch {epoch+1}:")
            print(f"Validation Loss: {val_loss}")

        scheduler.step()
        torch.save(model.state_dict(), 'deepkcat_epo20.pt')
        
if __name__ == '__main__':
    main()
