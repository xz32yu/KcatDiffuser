import torch
import math
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from rdkit import Chem
import scipy.sparse as sparse


word_dict = defaultdict(lambda: len(word_dict))
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

proteins = list()
compounds = list()
smilesadjacencies = list()
regression =list()
proteinadjacencies = list()

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)
    # return word_dict

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
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


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def dump_pickle(lists, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lists, f)

def get_info(filename):
    infos = []
    with open(filename, 'r') as file:
        for line in file:
            infos.append(line.strip())
    return infos

def main() :
    # radius = 3 # The initial setup, I suppose it is 2, but not 2.
    radius = 2
    ngram = 3

    
    dir = "./diffusion/results/"
    '''
    smiles_list = get_info(dir+"info/train_smiles.txt")
    sequence_list = get_info(dir+"info/train_seq.txt")
    kcat_list = get_info(dir+"info/train_kcat.txt")
    pdb_list = get_info(dir+"info/train_pdb.txt")
    '''
    smiles_list = get_info(dir+"info/val_smiles.txt")
    sequence_list = get_info(dir+"info/val_seq.txt")
    kcat_list = get_info(dir+"info/val_kcat.txt")
    pdb_list = get_info(dir+"info/val_pdb.txt")

    """Exclude data contains '.' in the SMILES format."""
    for i in range(len(smiles_list)):
        smiles = smiles_list[i]
        sequence = sequence_list[i]
        # print(smiles)
        Kcat = kcat_list[i]
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol)
        # print(atoms)
        i_jbond_dict = create_ijbonddict(mol)
        # print(i_jbond_dict)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        #print(fingerprints)
        compounds.append(fingerprints)

        smilesadjacency = create_adjacency(mol)
        #print(smilesadjacency)
        smilesadjacencies.append(smilesadjacency)

        words = split_sequence(sequence,ngram)
        #print(len(words))
        proteins.append(words)

        structure = pdb_list[i]
        proteinadjacency = luciferase_contact_map(structure,sequence)
        #proteinadjacency = luciferase_contact_map(structure)
        #print(proteinadjacency)
        proteinadjacencies.append(proteinadjacency)

        # print(float(Kcat))

        regression.append(np.array([float(Kcat)]))

        # regression.append(np.array([math.log10(float(Kcat))]))
    '''
    dump_pickle(compounds, dir+'example/train/compounds.pickle')
    dump_pickle(smilesadjacencies, dir+'example/train/smilesadjacencies.pickle')
    dump_pickle(regression, dir+'example/train/kcat.pickle')
    dump_pickle(proteins, dir+'example/train/proteins.pickle')
    dump_pickle(proteinadjacencies, dir+'example/train/proteinadjacencies.pickle')

    dump_dictionary(fingerprint_dict, dir+'example/train/fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir+'example/train/atom_dict.pickle')
    dump_dictionary(bond_dict, dir+'example/train/bond_dict.pickle')
    dump_dictionary(edge_dict, dir+'example/train/edge_dict.pickle')
    dump_dictionary(word_dict, dir+'example/train/sequence_dict.pickle')
    '''
    dump_pickle(compounds, dir+'example/val/compounds.pickle')
    dump_pickle(smilesadjacencies, dir+'example/val/smilesadjacencies.pickle')
    dump_pickle(regression, dir+'example/val/kcat.pickle')
    dump_pickle(proteins, dir+'example/val/proteins.pickle')
    dump_pickle(proteinadjacencies, dir+'example/val/proteinadjacencies.pickle')

    dump_dictionary(fingerprint_dict, dir+'example/val/fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir+'example/val/atom_dict.pickle')
    dump_dictionary(bond_dict, dir+'example/val/bond_dict.pickle')
    dump_dictionary(edge_dict, dir+'example/val/edge_dict.pickle')
    dump_dictionary(word_dict, dir+'example/val/sequence_dict.pickle')
    


if __name__ == '__main__' :
    main()
