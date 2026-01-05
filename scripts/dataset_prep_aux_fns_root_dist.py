import subprocess
import torch
from aux_msa_functions import *
from Bio import Phylo
import numpy as np
import networkx as nx
from ete3 import Tree
import torch
from Bio import Phylo
import os
from Bio import Phylo
from tqdm import tqdm
from select_gpu import get_free_gpu
import esm

def init_tree_processing(tree):

    leaf_lengths_dict = {}
    internal_lengths_dict = {}
    internal_distances_from_root = {}
    leaf_distances_from_root = {}

    for node in tree.traverse("preorder"):    
        if node.is_root():
            internal_distances_from_root[node.name] = 0
            internal_lengths_dict[node.name] = 0
            continue
        else:
            branch_length = node.get_distance(node.up)
            
            if node.is_leaf():
                leaf_lengths_dict[node.name] = branch_length
                leaf_distances_from_root[node.name] = node.get_distance(tree)
            else:
                internal_lengths_dict[node.name] = branch_length
                internal_distances_from_root[node.name] = node.get_distance(tree)
            
    return leaf_lengths_dict, leaf_distances_from_root, internal_lengths_dict, internal_distances_from_root

def prepare_branch_lengths_and_root_distances(true_tree_path, MSA_path):

    sequences = read_msa(MSA_path)
    seqs_order = [seq[0] for seq in sequences]

    true_tree, node_taxa_mapping, taxa_node_mapping = newick_to_graph(true_tree_path)
    leaf_lengths_dict, leaf_distances_from_root_dict, internal_lengths_dict, internal_distances_from_root_dict  = init_tree_processing(true_tree)

    leaf_branch_lengths = [leaf_lengths_dict[taxa_node_mapping[seq_name]] for seq_name in seqs_order]
    leaf_distances_from_root = [leaf_distances_from_root_dict[taxa_node_mapping[seq_name]] for seq_name in seqs_order]

    sorted_internal_distances_from_root_indices = np.argsort(list(internal_distances_from_root_dict.values()))
    sorted_internal_nodes = np.array(list(internal_distances_from_root_dict.keys()))[sorted_internal_distances_from_root_indices]

    internal_branch_lengths = [internal_lengths_dict[node] for node in sorted_internal_nodes]
    internal_distances_from_root = [internal_distances_from_root_dict[node] for node in sorted_internal_nodes]

    if len(internal_branch_lengths) != len(leaf_branch_lengths) - 1:
        return None, None, None

    all_branch_lengths =  internal_branch_lengths + leaf_branch_lengths
    all_branch_lengths = torch.tensor(all_branch_lengths)

    all_root_distances = internal_distances_from_root + leaf_distances_from_root
    all_root_distances = torch.tensor(all_root_distances)

    seqs_order_dict = {seq:i for i,seq in enumerate(seqs_order)}

    return all_branch_lengths, all_root_distances, seqs_order_dict

def prepare_initial_leaf_embeddings(data_ids, Large_D = 1000, model = None, batch_converter = None, device = "cuda", use_transf = True, alphabet = None):

    torch.cuda.empty_cache()
    
    all_embeddings = []

    for i in tqdm(range(len(data_ids))):

        # submsa_path = f"../data/submsa-seed-simulations-equal-size/4/{data_ids[i][0]}/submsa-{data_ids[i][1]}.fasta"
        submsa_path = f"../data/simulated_msas_phyloformer/size-4-len-100/{data_ids[i][0]}_4_tips.fa"
        submsa = read_msa(submsa_path) 


        _, _, tokens = batch_converter([submsa])
        tokens = tokens.to(device)

        if use_transf:
            with torch.no_grad():
                embeddings = model(tokens, need_head_weights = False, return_contacts = False, repr_layers = [12])["representations"][12][0]
                embeddings = embeddings.mean(dim=1).cpu()
                R, H = embeddings.shape
                padding_tensor = torch.zeros((R, Large_D - H))
                embeddings = torch.concat((embeddings, padding_tensor), dim=1)


                del padding_tensor, tokens
            
                all_embeddings.append(embeddings)

                del embeddings
                torch.cuda.empty_cache()
        else:
            alphabet_dict = alphabet.to_dict()

            embeddings = torch.zeros(len(submsa), len(submsa[0][1]), len(alphabet_dict))

            for k in range(len(submsa)): 
                for j, tok in enumerate(tokens[0][k][1:]):
                    embeddings[k, j, tok] = 1

            embeddings = embeddings.mean(dim = 1)
            # R, H = embeddings.shape
            # padding_tensor = torch.zeros((R, Large_D - H))
            # embeddings = torch.concat((embeddings, padding_tensor), dim=1)

            del tokens
        
            all_embeddings.append(embeddings)

            del embeddings



    return all_embeddings


def make_datasets(data_id, leaf_embeddings = None, Large_D = 768):
    
    # true_tree_path = f"../data/msa-seed-simulations-subtrees-equal-size/4/{data_id[0]}/subtree-{data_id[1]}.newick"
    # MSA_path = f"../data/submsa-seed-simulations-equal-size/4/{data_id[0]}/submsa-{data_id[1]}.fasta"

    true_tree_path = f"../data/simulated_trees_phyloformer/size-4/{data_id[0]}_4_tips.newick"
    MSA_path = f"../data/simulated_msas_phyloformer/size-4-len-100/{data_id[0]}_4_tips.fa"

    all_branch_lengths, all_root_distances, seqs_order = prepare_branch_lengths_and_root_distances(true_tree_path, MSA_path)

    return all_branch_lengths, all_root_distances

def newick_to_graph(newick_str):
    
    t = Tree(newick_str, format=1)
    # R = t.get_midpoint_outgroup()
    # t.set_outgroup(R)

    if len(t.children) == 3:
        print("x")
        R = t.get_midpoint_outgroup()
        t.set_outgroup(R)

    counter = [0]
    node_taxa_mapping = {}
    taxa_node_mapping = {}
        
    for node in t.traverse("preorder"):

        if node.is_leaf():
            node_taxa_mapping[counter[0]] = node.name.replace("'","")
            taxa_node_mapping[node.name] = counter[0]

        node.name = counter[0]
        counter[0] += 1

    return t, node_taxa_mapping, taxa_node_mapping

def node_embedding(tree, seqs_order, node_taxa_mapping, rooted = False, leaf_embeddings = None):

    ntips = len(tree.get_leaves()) 

    if leaf_embeddings == None:   
        leaf_embeddings = torch.eye(ntips)

        
    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.c = 0
            taxa = node_taxa_mapping[node.name]
            leaf_emb_ind = seqs_order[taxa]
            node.d = leaf_embeddings[leaf_emb_ind]
        else:
            child_c, child_d = 0., 0.
            for child in node.children:
                child_c += child.c
                child_d += child.d
            if node.is_root() and rooted:
                node.c = 1./(2. - child_c)
                node.d = node.c * child_d
            else:
                node.c = 1./(3. - child_c)
                node.d = node.c * child_d
        
    node_features, edge_index, leaf_indices = [], [], []    

    for node in tree.traverse('preorder'):
        neigh_idx_list = []
        if not node.is_root():
            node.d = node.c * node.up.d + node.d
            neigh_idx_list.append(node.up.name)
            
            if not node.is_leaf():
                neigh_idx_list.extend([child.name for child in node.children])
            else:
                neigh_idx_list.extend([-1, -1])              
        else:
            if rooted:
                neigh_idx_list.extend([-1] + [child.name for child in node.children])
            else:
                neigh_idx_list.extend([child.name for child in node.children])
              
        edge_index.append(neigh_idx_list)                
        node_features.append(node.d)

        if node.is_leaf():
            leaf_indices.append(node.name)

    edge_index = torch.LongTensor(edge_index)
    
    return torch.stack(node_features), edge_index, leaf_indices
