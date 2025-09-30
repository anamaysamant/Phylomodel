from io import StringIO
import numpy as np
from phylomodel_models_root_distance import MainPhyloModel
from dataset_prep_aux_fns import *
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import torch.nn as nn
import pickle as pkl

from tqdm import tqdm

def adj_mat_from_distances(branch_lengths, root_distances):

    n_nodes = len(branch_lengths)
    n_int_nodes = int((len(branch_lengths) + 1)/2 - 1)
    n_leaf_nodes = n_int_nodes + 1

    A = np.zeros((n_nodes, n_nodes))

    queue = [0]
    used = set([0])

    total_leaf_count = 0

    while queue:

        cur_node = queue.pop(0)

        if cur_node > n_int_nodes - 1:
            continue

        diff =  root_distances - branch_lengths - root_distances[cur_node]
        diff_ind_order = np.argsort(diff)

        n_children = 0
        leaf_children = 0

        for i in diff_ind_order:

            if len(queue) == 1 and leaf_children == 1 and i > n_int_nodes - 1 and total_leaf_count < n_leaf_nodes - 1:
                continue

            if n_children == 2:
                break
            
            if i not in used:

                A[cur_node, i] = branch_lengths[i]
                queue.append(i)
                used.add(i)
                n_children += 1

                if i > n_int_nodes - 1:
                    leaf_children += 1
                    total_leaf_count += 1

    return A

def prepare_initial_leaf_embeddings(msa, Large_D = 1000, model = None, batch_converter = None, device = None):

    torch.cuda.empty_cache()

    _, _, tokens = batch_converter([msa])
    tokens = tokens.to(device)

    with torch.no_grad():
        embeddings = model(tokens, need_head_weights = False, return_contacts = False, repr_layers = [12])["representations"][12][0]
        embeddings = embeddings.mean(dim=1)
        R, H = embeddings.shape
        padding_tensor = torch.zeros((R, Large_D - H)).to(device)
        embeddings = torch.concat((embeddings, padding_tensor), dim=1)

        del padding_tensor, tokens
        torch.cuda.empty_cache()

    return embeddings

def to_newick(G, node):
    children = list(G.successors(node))  # for directed graph
    if not children:
        # Leaf: just return name with branch length if available
        if G.in_edges(node):
            parent = list(G.predecessors(node))[0]
            return f"{node}:{G[parent][node].get('weight', 1.0)}"
        else:
            return node  # root leaf (edge case)
    else:
        # Internal node: recurse over children
        subtrees = [to_newick(G, c) for c in children]
        if G.in_edges(node):
            parent = list(G.predecessors(node))[0]
            return f"({','.join(subtrees)}){node}:{G[parent][node].get('weight', 1.0)}"
        else:
            # Root node (no incoming edge) → no branch length
            return f"({','.join(subtrees)}){node}"

gpu = str(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
msa_transf, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transf = msa_transf.to(device)
batch_converter = alphabet.get_batch_converter()
msa_transf.eval()

checkpoint_path = "../models/bl-fit-200-PF00004-size-4-50-epochs.pt"

checkpoint = torch.load(checkpoint_path, weights_only=False)

model = MainPhyloModel(**checkpoint["model_hparams"]).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open("../data/families_under_200_over_10.pkl","rb") as f:

    all_families_init = pkl.load(f)

all_families = []

for family in all_families_init:

    if os.path.exists(f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"):
        all_families.append(family)

all_families = all_families[:1]

np.random.seed(42)

data_ids = list(range(1,20000))

train_split = 0.9
train_size = int(np.ceil(train_split * len(data_ids)))

train_ids_ind = np.random.choice(range(len(data_ids)), train_size, replace = False)
train_ids = [data_ids[ind] for ind in train_ids_ind]

test_ids_ind = list(set(range(len(data_ids))) - set(train_ids_ind))
test_ids = [data_ids[ind] for ind in test_ids_ind]

# all_families = all_families[:100]

inferred_trees = {}
# train_ids= ["4745"]

for data_id in tqdm(train_ids):

    input_MSA_path = f"../data/submsa-seed-simulations-equal-size/4/PF00004/submsa-{data_id}.fasta"
    input_MSA = read_msa(input_MSA_path)

    n_int_nodes = len(input_MSA) - 1
    total_nodes = 2 * len(input_MSA) - 1
    leaf_nodes = len(input_MSA)

    input_MSA_order = {seq[0]:i for i,seq in enumerate(input_MSA)}
    X = prepare_initial_leaf_embeddings(input_MSA, Large_D=768, model = msa_transf, batch_converter = batch_converter, device = device)

    with torch.no_grad():
        branch_lengths, root_distances = model(X, attn_mask = None)
        branch_lengths = branch_lengths.squeeze(-1).cpu().numpy()
        root_distances = root_distances.squeeze(-1).cpu().numpy()

    hard_adj_mat = adj_mat_from_distances(branch_lengths, root_distances)
    tree_graph = nx.from_numpy_array(hard_adj_mat, create_using=nx.DiGraph)

    labels = list(map(str,range(n_int_nodes))) + list(input_MSA_order.keys())
    mapping = {i: labels[i] for i in range(len(labels))}
    tree_graph = nx.relabel_nodes(tree_graph, mapping)

    # cycles = list(nx.simple_cycles(tree_graph))
    # print("Cycles found:", cycles)

    # cycle_nodes = set(n for c in cycles for n in c)
    # node_colors = ["red" if n in cycle_nodes else "lightblue" for n in tree_graph.nodes]

    # pos = graphviz_layout(tree_graph, prog="dot")
    # plt.figure(figsize=(30, 30))
    # nx.draw(tree_graph, pos,
    #         with_labels=True,
    #         node_color=node_colors,
    #         node_size=500,
    #         arrowsize=20,
    #         font_size=8)

    # plt.title("Cycles highlighted in red")
    # plt.show()

    newick_str = to_newick(tree_graph, "0") + ";"

    handle = StringIO(newick_str)
    tree = Phylo.read(handle, "newick")
    Phylo.write(tree, "inferred-tree-test.newick", "newick")
    # Phylo.draw_ascii(tree)
    
    # newick_str = to_newick(tree_graph, "0") + ";"
    # tree = Tree(newick=newick_str, format = 1)
    # inferred_trees[data_id] = tree




    