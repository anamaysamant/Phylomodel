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

torch.cuda.empty_cache()

def init_tree_processing(tree):

    leaf_lengths_dict = {}
    internal_lengths_dict = {}
    internal_distances_from_root = {}
    child_parent_rel = {}

    for node in tree.traverse("preorder"):    

        if node.is_root():
            internal_distances_from_root[node.name] = 0
            internal_lengths_dict[node.name] = 0
            child_parent_rel[node.name] = -1
        else:
            branch_length = node.get_distance(node.up)
            
            if node.is_leaf():
                leaf_lengths_dict[node.name] = branch_length
            else:
                internal_lengths_dict[node.name] = branch_length
                internal_distances_from_root[node.name] = node.get_distance(tree)

            child_parent_rel[node.name] = node.up.name
        
    return leaf_lengths_dict, internal_lengths_dict, internal_distances_from_root, child_parent_rel

def prepare_branch_lengths_and_relationships(true_tree_path, MSA_path):

    sequences = read_msa(MSA_path)
    true_tree = Phylo.read(true_tree_path, format="newick")
    true_tree.root_at_midpoint()

    sequences = reorder_seqs(true_tree.clade, dict(sequences))
    seqs_order = [seq[0] for seq in sequences]

    true_tree, node_taxa_mapping, taxa_node_mapping = newick_to_graph(true_tree_path)
    leaf_lengths_dict, internal_lengths_dict, internal_distances_from_root, child_parent_rel  = init_tree_processing(true_tree)

    leaf_branch_lengths = [leaf_lengths_dict[taxa_node_mapping[seq_name]] for seq_name in seqs_order]

    sorted_internal_distances_from_root_indices = np.argsort(list(internal_distances_from_root.values()))
    sorted_internal_nodes= np.array(list(internal_distances_from_root.keys()))[sorted_internal_distances_from_root_indices]

    internal_branch_lengths = [internal_lengths_dict[node] for node in sorted_internal_nodes]

    if len(internal_branch_lengths) != len(leaf_branch_lengths) - 1:
        return None, None, None
    
    all_branch_lengths =  internal_branch_lengths + leaf_branch_lengths
    all_branch_lengths = torch.tensor(all_branch_lengths)

    leaf_node_parents = [child_parent_rel[taxa_node_mapping[taxa]] for taxa in seqs_order]
    int_node_parents = [child_parent_rel[node] for node in sorted_internal_nodes]

    all_parents = int_node_parents + leaf_node_parents
    parents_mapping = {node:ind for ind, node in enumerate(sorted_internal_nodes)}
    parents_mapping[-1] = -2

    all_parents_mapped = torch.tensor([int(parents_mapping[node]) for node in all_parents])

    seqs_order_dict = {seq:i for i,seq in enumerate(seqs_order)}

    return all_branch_lengths, all_parents_mapped, seqs_order_dict

def prepare_initial_int_node_embeddings(MSA_file_path, seqs_order, Large_D = 1000, leaf_embeddings = None, delete_init_tree = True, fam = None):

    temp_tree_file = f"{MSA_file_path}-temp.newick"

    subprocess.run(["rapidnj",MSA_file_path,"-i","fa","-x",temp_tree_file])

    init_tree, node_taxa_mapping, _ = newick_to_graph(temp_tree_file, fam = fam) 

    init_tree.set_outgroup(init_tree.get_midpoint_outgroup())
    
    subprocess.run(['rm', temp_tree_file])
        
    node_embeddings, edges, leaf_indices = node_embedding(init_tree, seqs_order, node_taxa_mapping, rooted=True, leaf_embeddings=leaf_embeddings)
    _, _, init_distances_from_root, _  = init_tree_processing(init_tree)

    sorted_init_distances_from_root_indices = np.argsort(list(init_distances_from_root.values()))
    
    sorted_init_nodes = np.array(list(init_distances_from_root.keys()))[sorted_init_distances_from_root_indices]
    
    sorted_internal_node_embeddings = node_embeddings[sorted_init_nodes]
    R, C = sorted_internal_node_embeddings.shape
    
    padding_tensor = torch.zeros((R, Large_D - C)).to(leaf_embeddings.device)
    sorted_internal_node_embeddings = torch.concat((sorted_internal_node_embeddings, padding_tensor), dim=1)

    return sorted_internal_node_embeddings


def prepare_initial_leaf_embeddings(data_ids, Large_D = 1000, model = None, batch_converter = None, device = "cuda"):

    torch.cuda.empty_cache()
    
    all_embeddings = []

    for i in tqdm(range(len(data_ids))):

        true_tree = Phylo.read(f"../data/msa-seed-simulations-subtrees-equal-size/4/{data_ids[i][0]}/subtree-{data_ids[i][1]}.newick", format="newick")
        true_tree.root_at_midpoint()

        submsa_path = f"../data/submsa-seed-simulations-equal-size/4/{data_ids[i][0]}/submsa-{data_ids[i][1]}.fasta"

        submsa = read_msa(submsa_path) 
        submsa = reorder_seqs(true_tree.clade, dict(submsa))

        _, _, tokens = batch_converter([submsa])
        tokens = tokens.to(device)

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

    return all_embeddings


def make_datasets(data_id, leaf_embeddings = None, Large_D = 1000):
    
    true_tree_path = f"../data/msa-seed-simulations-subtrees-equal-size/4/{data_id[0]}/subtree-{data_id[1]}.newick"
    MSA_path = f"../data/submsa-seed-simulations-equal-size/4/{data_id[0]}/submsa-{data_id[1]}.fasta"

    all_branch_lengths, mapped_parents, seqs_order = prepare_branch_lengths_and_relationships(true_tree_path, MSA_path)

    if all_branch_lengths == None:
        return None, None, None

    internal_node_embeddings = prepare_initial_int_node_embeddings(MSA_path, seqs_order, Large_D = Large_D, leaf_embeddings = leaf_embeddings)

    return internal_node_embeddings, all_branch_lengths, mapped_parents

def newick_to_graph(newick_str, fam = None):
    
    t = Tree(newick_str, format=1)

    R = t.get_midpoint_outgroup()
    t.set_outgroup(R)
 
    # t.write(format=1, outfile=f"../tree-tests/{fam}-init-tree.newick")

    counter = [0]
    node_taxa_mapping = {}
    taxa_node_mapping = {}
        
    for node in t.traverse("preorder"):

        if node.is_leaf():
            node.name = node.name.replace("'","")
            node_taxa_mapping[counter[0]] = node.name
            taxa_node_mapping[node.name] = counter[0]

        node.name = counter[0]
        counter[0] += 1

    return t, node_taxa_mapping, taxa_node_mapping

def node_embedding(tree, seqs_order, node_taxa_mapping, rooted = False, leaf_embeddings = None):

    ntips = len(tree.get_leaves()) 

    if leaf_embeddings == None:   
        leaf_embeddings = torch.eye(ntips).to(leaf_embeddings.device)

        
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
