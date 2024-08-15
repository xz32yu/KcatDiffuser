import os
import re
import torch
import esm
import numpy as np
from ema_pytorch import EMA
from kcatdiffuser import EGNN_NET, KcatDiffuser
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch
from dataset_src.generate_graph import prepare_graph
from esm.pretrained import load_model_and_alphabet
from model.regressor.example_model import load_pickle
from model.regressor.deepkcat import DeepKcat

def predict_sequence(dirname):
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
    filename = 'result_0.pt'
    amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('./diffusion/results/weight/June20_result_lr=0.0005_dp=0.1_clip=1.0_timestep=500_depth=6_hidden=128_embedding=True_embed_dim=128_ss=-1_noise=blosum.pt', map_location=device)
    config = ckpt['config']
    config['noise_type'] = 'blosum'#uniform
    gnn = EGNN_NET(input_feat_dim=config['input_feat_dim'], hidden_channels=config['hidden_dim'], edge_attr_dim=config['edge_attr_dim'], dropout=config['drop_out'], n_layers=config['depth'], update_edge = config['update_edge'], embedding=config['embedding'], embedding_dim=config['embedding_dim'], embed_ss=config['embed_ss'], norm_feat=config['norm_feat'])
    diffusion = KcatDiffuser(model=gnn, config=config)
    diffusion = EMA(diffusion)
    diffusion.load_state_dict(ckpt['ema'])
    pdb = './dataset/raw/test/result_0.pdb'
    seq = 'MAAFSLSAKQILSPSTHRPSLSKTTTADSSLRFRNPHSLSLRCSSLSSSSNVGRTRLMRASASSTAPVMDTSPTKAVSSAPTIVDVDLGDRSYPIYIGSGLLDQPDLLQRHVHGKRVLVVTNSTVAPIYLDKVVGALTNGNPNVSVESVILPDGEKYKNMDTLMKVFDKAIESRLDRRCTFVALGGGVIGDMCGYAAASFLRGVNFIQIPTTVMAQVDSSVGGKTGINHRLGKNLIGAFYQPQCVLIDTDTLNTLPDRELASGLAEVVKYGLIRDANFFEWQEKNMPALMARDPSALAYAIKRSCENKAEVVSLDEKESGLRATLNLGHTFGHAIETGFGYGQWLHGEAVAAGMVMAVDMSYRLGWIDESIVNRAHNILQQAKLPTAPPETMTVEMFKSVMAVDKKVADGLLRLILLKGPLGNCVFTGDYDRKALDETLHAFCKS'
    smiles = 'C(C(C(C(COP(=O)(O)O)O)O)O)C(=O)C(=O)O'
    kcat = '4.329123596291566'
    
    graph = torch.load(dirname+filename)
    data = prepare_graph(graph)
    input_graph = Batch.from_data_list([data])
    fingerprint_dict = load_pickle('./diffusion/results/dict/fingerprint_dict.pickle')
    word_dict = load_pickle('./diffusion/results/dict/sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    model = DeepKcat(n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size, layers_trans).to(device)
    model.load_state_dict(torch.load('./diffusion/results/model/deepkcat_epo20.pt'), strict=False)
    model.train(False)
    file = filenames[i].replace(".pt","")
    
    prob,sample_graph = diffusion.ema_model.ddim_sample_regressor(input_graph, smiles, seq, pdb, cond=False, diverse=False, stop=0, step=50, regressor=model, input_properties=data)
    sequence = ''.join([amino_acids_type[i] for i in sample_graph.argmax(dim=1).tolist()])

    '''
    graph = torch.load(dirname+filename)
    input_graph = Batch.from_data_list([prepare_graph(graph)])
    prob,sample_graph = diffusion.ema_model.ddim_sample(input_graph)
    sequence = ''.join([amino_acids_type[i] for i in sample_graph.argmax(dim=1).tolist()])
    '''
    return sequence

if __name__ == "__main__":
    dirname = "./dataset/process/test/"
    sequence = predict_sequences(dirname)
    print(sequence)

