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

gpu = str(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()

def make_branch_lengths(tree):

    leaf_lengths_dict = {}
    internal_lengths_dict = {}
    internal_distances_from_root = {}
    child_parent_rel = {}


    for node in tree.traverse("preorder"):    
        if node.is_root():
            internal_distances_from_root[node.name] = 0
            internal_lengths_dict[node.name] = 0
            child_parent_rel[node.name] = -1
            continue
        else:
            branch_length = node.get_distance(node.up)
            
            if node.is_leaf():
                leaf_lengths_dict[node.name] = branch_length
            else:
                internal_lengths_dict[node.name] = branch_length
                internal_distances_from_root[node.name] = node.get_distance(tree)

            child_parent_rel[node.name] = node.up.name
            
    return leaf_lengths_dict, internal_lengths_dict, internal_distances_from_root, child_parent_rel


def prepare_branch_lengths_and_relationships(true_tree_path, nat_MSA_path):

    nat_sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(nat_MSA_path, "fasta")]
    true_tree = Phylo.read(true_tree_path, format="newick")

    nat_sequences = reorder_seqs(true_tree.clade, dict(nat_sequences))
    nat_seqs_order = [seq[0] for seq in nat_sequences]

    _, true_tree, node_taxa_mapping, taxa_node_mapping = newick_to_graph(true_tree_path)
    leaf_lengths_dict, internal_lengths_dict, internal_distances_from_root, child_parent_rel  = make_branch_lengths(true_tree)

    leaf_branch_lengths = [leaf_lengths_dict[taxa_node_mapping[seq_name]] for seq_name in nat_seqs_order]

    sorted_internal_distances_from_root_indices = np.argsort(list(internal_distances_from_root.values()))
    sorted_internal_nodes= np.array(list(internal_distances_from_root.keys()))[sorted_internal_distances_from_root_indices]

    internal_branch_lengths = [internal_lengths_dict[node] for node in sorted_internal_nodes]

    all_branch_lengths =  internal_branch_lengths + leaf_branch_lengths
    all_branch_lengths = torch.tensor(all_branch_lengths)

    leaf_node_parents = [child_parent_rel[taxa_node_mapping[taxa]] for taxa in nat_seqs_order]
    int_node_parents = [child_parent_rel[node] for node in sorted_internal_nodes]

    all_parents = int_node_parents + leaf_node_parents
    parents_mapping = {node:ind for ind, node in enumerate(sorted_internal_nodes)}
    parents_mapping[-1] = -2

    all_parents_mapped = torch.tensor([int(parents_mapping[node]) for node in all_parents])

    return all_branch_lengths, all_parents_mapped, sorted_internal_nodes

def prepare_initial_int_node_embeddings(MSA_file_path, true_tree_path, Large_D = 1000, leaf_embeddings = None):

    # true_tree = Phylo.read(true_tree_path, format="newick")

    # sequences = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(MSA_file_path, "fasta")]
    # sequences = reorder_seqs(true_tree.clade, dict(sequences))
        
    # _,_,batch_tokens = batch_converter([sequences])
    # batch_tokens = batch_tokens.to(device)

    # with torch.no_grad():
    #     leaf_embeddings = model(batch_tokens, need_head_weights = False, return_contacts = False, repr_layers = [12])["representations"][12][0]
    #     leaf_embeddings = leaf_embeddings.mean(dim=1).cpu()
    #     R, C = leaf_embeddings.shape
    #     padding_tensor = torch.zeros((R, Large_D - C))
    #     leaf_embeddings = torch.concat((leaf_embeddings, padding_tensor), dim=1)

    # del(batch_tokens)
    # torch.cuda.empty_cache()

    temp_tree_file = f"{MSA_file_path}-temp.newick"

    subprocess.run(["rapidnj",MSA_file_path,"-i","fa","-x",temp_tree_file])

    _, init_tree, _, _ = newick_to_graph(temp_tree_file) 
 
    subprocess.run(['rm', temp_tree_file])
    # subprocess.run(['rm', temp_matrix_file])

    internal_node_embeddings, edges, leaf_indices = node_embedding(init_tree, rooted=True, leaf_embeddings=leaf_embeddings)
    _, _, init_distances_from_root, _  = make_branch_lengths(init_tree)

    sorted_init_distances_from_root_indices = np.argsort(list(init_distances_from_root.values()))
    
    sorted_init_nodes = np.array(list(init_distances_from_root.keys()))[sorted_init_distances_from_root_indices]

    # if task == "bl_pred":
    #     sorted_init_nodes = sorted_init_nodes[1:]
    
    sorted_internal_node_embeddings = internal_node_embeddings[sorted_init_nodes]
    R, C = sorted_internal_node_embeddings.shape
    
    padding_tensor = torch.zeros((R, Large_D - C))
    sorted_internal_node_embeddings = torch.concat((sorted_internal_node_embeddings, padding_tensor), dim=1)

    # all_embeddings = torch.concat((sorted_internal_node_embeddings, leaf_embeddings), dim = 0)

    return sorted_internal_node_embeddings

def prepare_initial_leaf_embeddings(families, Large_D = 1000, batch_size = 10):

    torch.cuda.empty_cache()
    
    all_embeddings = []

    msa_folder = "../data/protein-families-msa-seed/"
    tree_folder ="../data/seed-trees/"
    # Process in batches
    for batch_start in tqdm(range(0, len(families), batch_size)):

        current_batch = range(batch_start, batch_start + batch_size)

        true_trees = [Phylo.read(os.path.join(tree_folder, f"{families[i]}_seed.newick"), format="newick") for i in current_batch]
        batch_paths = [os.path.join(msa_folder, f"{families[i]}_seed.fasta") for i in current_batch]

        # Load MSAs
        batch_data = [read_msa(p) for p in batch_paths]
        batch_data = [reorder_seqs(true_trees[i].clade, dict(batch_data[i])) for i in range(len(batch_data))]

        # Store sizes for unbatching
        msa_depths = [len(msa) for msa in batch_data]
        msa_lengths = [len(msa[0][1]) for msa in batch_data]

        # Convert to tokens
        _, _, batch_tokens = batch_converter(batch_data)  # shape: (B, N_max, L_max)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            batch_embeddings = model(batch_tokens, need_head_weights = False, return_contacts = False, repr_layers = [12])["representations"][12]
            batch_embeddings = batch_embeddings.mean(dim=2)
            B, R, H = batch_embeddings.shape
            padding_tensor = torch.zeros((B,R, Large_D - H)).to(device)
            batch_embeddings = torch.concat((batch_embeddings, padding_tensor), dim=2)

            del padding_tensor

            for i, (depth, length, path) in enumerate(zip(msa_depths, msa_lengths, batch_paths)):
                
                leaf_emb = batch_embeddings[i, :depth, :].cpu()
                all_embeddings.append(leaf_emb)

            del batch_tokens, batch_embeddings
            torch.cuda.empty_cache()


    return all_embeddings

def make_datasets(family, leaf_embeddings = None, Large_D = 1000):
    
    true_tree_path = f"../data/seed-trees/{family}_seed.newick"
    nat_MSA_path = f"../data/protein-families-msa-seed/{family}_seed.fasta"
    
    all_branch_lengths, mapped_parents, sorted_internal_nodes = prepare_branch_lengths_and_relationships(true_tree_path, nat_MSA_path)

    all_embeddings = prepare_initial_int_node_embeddings(nat_MSA_path, true_tree_path, Large_D = Large_D, leaf_embeddings = leaf_embeddings)

    return all_embeddings, all_branch_lengths, mapped_parents

def newick_to_graph(newick_str):
    
    t = Tree(newick_str, format=1)
    R = t.get_midpoint_outgroup()
    t.set_outgroup(R)
    G = nx.Graph()
    counter = [0]
    node_taxa_mapping = {}
    taxa_node_mapping = {}
        
    for node in t.traverse("preorder"):

        if node.is_leaf():
            node_taxa_mapping[counter[0]] = node.name
            taxa_node_mapping[node.name] = counter[0]

        node.name = counter[0]
        counter[0] += 1

        if not node.is_root():
            G.add_edge(node.up.name, node.name)

    return G, t, node_taxa_mapping, taxa_node_mapping

def node_embedding(tree, rooted = False, leaf_embeddings = None):

    ntips = len(tree.get_leaves()) 

    if leaf_embeddings == None:   
        leaf_embeddings = torch.eye(ntips)
        
    counter = [0]

    if rooted:
        total_nodes = 2 * ntips - 1
    else:
        total_nodes = 2 * ntips - 2

    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.c = 0
            node.d = leaf_embeddings[counter[0]]
            counter[0] += 1
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