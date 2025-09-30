from io import StringIO
import numpy as np
from phylomodel_models import ParentPredictor
from dataset_prep_aux_fns import *
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import torch.nn as nn
import pickle as pkl

from tqdm import tqdm


def greedy_phylo_tree_bottom_up(P):

    n = P.shape[0]
    A = np.zeros((n, n), dtype=int)

    P[0,:] = 0.0
    
    root = 0
    parents = {root: None}
    parent_count = {}
    assigned_children = {root: []}

    n_int_nodes = P.shape[1]
    leaf_nodes = list(range(n_int_nodes, n))

    used = set(leaf_nodes)

    assert (np.tril(P, k = -1) == P).all()

    for u in range(1, n):

        probs = [(float(P[u, v]), v) for v in range(n_int_nodes) if v not in used and P[u,v] > 0]
        probs.sort(reverse=True)

        parent_candidate = probs[0][1]

        if parent_candidate not in parent_count.keys():
            parent_count[parent_candidate] = 1
            parents[u] = parent_candidate
            assigned_children[parent_candidate] = [u]
            A[u, parent_candidate] = 1
        
        elif parent_count[parent_candidate] < 2:
            parent_count[parent_candidate] += 1
            parents[u] = parent_candidate
            assigned_children[parent_candidate].append(u)
            used.add(parent_candidate)
            A[u, parent_candidate] = 1
        
    return A

def will_create_cycle(adj, u, v):

    n = adj.shape[0]

    # DFS starting from v to see if we can reach u
    visited = set()
    stack = [v]

    while stack:
        node = stack.pop()
        if node == u:
            return True  # cycle detected
        if node not in visited:
            visited.add(node)
            neighbors = np.where(adj[:, node] != 0)[0]  # outgoing edges
            stack.extend(neighbors)

    return False

def construct_phylogenetic_tree(prob_matrix, binary=True):
    n = prob_matrix.shape[0]

    edges = [
        (i, j, prob_matrix[i, j])
        for i in range(n)
        for j in range(n)
        if i != j and prob_matrix[i,j] > 0
    ]


    edges.sort(key=lambda x: x[2], reverse=True)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u, v, p in edges:
        if G.in_degree(v) >= 1:
            continue
        if binary and G.out_degree(u) >= 2:
            continue

        G.add_edge(u, v, weight=p)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
            continue

        if G.number_of_edges() == n - 1:
            break

    # Final validation
    roots = [node for node in G.nodes if G.in_degree(node) == 0]

    if len(roots) != 1:
        invalid_trees[0] += 1
        invalid_roots.append(len(roots))
        # raise ValueError(f"Tree not valid: found roots {roots}")
    # if not nx.is_weakly_connected(G):
        # raise ValueError("Tree not connected")

    return G, roots[0]

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

checkpoint_path = "../models/fit-200.pt"

checkpoint = torch.load(checkpoint_path, weights_only=True)

model = ParentPredictor(**checkpoint["model_hparams"]).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open("../data/families_under_200_over_10.pkl","rb") as f:

    all_families_init = pkl.load(f)

all_families = []

for family in all_families_init:

    if os.path.exists(f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"):
        all_families.append(family)


invalid_trees = [0]
invalid_roots = []

all_families = all_families[:1]

for family in tqdm(all_families):

    input_MSA_path = f"../data/protein-families-msa-seed/{family}_seed.fasta"
    input_MSA = read_msa(input_MSA_path)

    n_int_nodes = len(input_MSA) - 1
    total_nodes = 2 * len(input_MSA) - 1
    leaf_nodes = len(input_MSA)

    input_MSA_order = {seq[0]:i for i,seq in enumerate(input_MSA)}


    leaf_embeddings = prepare_initial_leaf_embeddings(input_MSA, Large_D=768, model = msa_transf, batch_converter = batch_converter, device = device)

    int_node_embeddings = prepare_initial_int_node_embeddings(input_MSA_path, input_MSA_order, Large_D=768, leaf_embeddings=leaf_embeddings)

    X = torch.concat((int_node_embeddings, leaf_embeddings), dim=0)

    with torch.no_grad():
        soft_adj_mat = model(X, attn_mask = None).squeeze(-1)[...,:n_int_nodes]
        size = soft_adj_mat.shape[0]
        mask = torch.triu(torch.ones(size, size), diagonal=0).bool()[...,:n_int_nodes].to(device)
        soft_adj_mat = soft_adj_mat.masked_fill(mask, float('-inf'))
        soft_adj_mat = nn.Softmax(dim = 1)(soft_adj_mat).cpu().numpy()

    hard_adj_mat = greedy_phylo_tree_bottom_up(soft_adj_mat)
    hard_adj_mat = np.transpose(hard_adj_mat)

    # assert (np.triu(hard_adj_mat, k = 1) == hard_adj_mat).all()
    tree_graph = nx.from_numpy_array(hard_adj_mat, create_using=nx.DiGraph)

    # soft_adj_mat[0] = 0.0
    # soft_adj_mat = np.concat((soft_adj_mat, np.zeros((total_nodes,leaf_nodes))), axis = 1)

    # tree_graph, _ = construct_phylogenetic_tree(soft_adj_mat.transpose())

    labels = list(map(str,range(n_int_nodes))) + list(input_MSA_order.keys())
    mapping = {i: labels[i] for i in range(len(labels))}
    tree_graph = nx.relabel_nodes(tree_graph, mapping)

    cycles = list(nx.simple_cycles(tree_graph))
    print("Cycles found:", cycles)

    cycle_nodes = set(n for c in cycles for n in c)

    node_colors = ["red" if n in cycle_nodes else "lightblue" for n in tree_graph.nodes]

    pos = graphviz_layout(tree_graph, prog="dot")
    plt.figure(figsize=(30, 30))
    nx.draw(tree_graph, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            arrowsize=20,
            font_size=8)

    plt.title("Cycles highlighted in red")
    plt.show()

    # newick_str = to_newick(tree_graph, "0") + ";"

    # handle = StringIO(newick_str)
    # tree = Phylo.read(handle, "newick")
    # Phylo.write(tree, "inferred-tree-test.newick", "newick")
    # Phylo.draw_ascii(tree)




