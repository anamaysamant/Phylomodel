from io import StringIO
import numpy as np
from phylomodel_models import ParentPredictor
from dataset_prep_aux_fns import *
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import torch.nn as nn


def greedy_phylo_tree_bottom_up(P):

    n = P.shape[0]
    A = np.zeros((n, n), dtype=int)

    P[0,:] = 0.0
    
    # Choose root = node with highest total outgoing probability
    root = 0
    parents = {root: None}
    parent_count = {}
    assigned_children = {root: []}

    n_int_nodes = P.shape[1]
    leaf_nodes = list(range(n_int_nodes, n))


    used = set(leaf_nodes)

    max_probs_nodes = P.max(axis = 1)
    nodes_prob_order = np.argsort(max_probs_nodes)[::-1]

    # reordered_nodes = [all_nodes[1:][i] for i in nodes_prob_order]
    queue = list(nodes_prob_order)[:-1]

    counter = 0

    while queue:

        print(len(queue))

        u = queue.pop(0)

        probs = [(float(P[u, v]), v) for v in range(n) if v not in used and v != u and not will_create_cycle(A, v, u)]
        print(probs)
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
        
        counter += 1

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
        raise ValueError(f"Tree not valid: found roots {roots}")
    if not nx.is_weakly_connected(G):
        raise ValueError("Tree not connected")

    return G, roots[0]

def prepare_initial_leaf_embeddings(msa, Large_D = 1000):

    torch.cuda.empty_cache()

    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

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

checkpoint_path = "../models/subtree-fit-1.pt"

checkpoint = torch.load(checkpoint_path, weights_only=True)

checkpoint["model_hparams"]["input_dim"] = checkpoint["model_hparams"]["large_D"]
del checkpoint["model_hparams"]["large_D"]

model = ParentPredictor(**checkpoint["model_hparams"]).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_MSA_path = "../data/protein-families-msa-seed/PF00005_seed.fasta"
input_MSA = read_msa(input_MSA_path)

n_int_nodes = len(input_MSA) - 1
total_nodes = 2 * len(input_MSA) - 1
leaf_nodes = len(input_MSA)

input_MSA_order = {seq[0]:i for i,seq in enumerate(input_MSA)}

leaf_embeddings = prepare_initial_leaf_embeddings(input_MSA, Large_D=768)
int_node_embeddings = prepare_initial_int_node_embeddings(input_MSA_path, input_MSA_order, Large_D=768, leaf_embeddings=leaf_embeddings)

X = torch.concat((int_node_embeddings, leaf_embeddings), dim=0)

with torch.no_grad():
    soft_adj_mat = model(X, attn_mask = None).squeeze(-1)[...,:n_int_nodes]
    soft_adj_mat = nn.Softmax(dim = 1)(soft_adj_mat).cpu().numpy()

# hard_adj_mat = greedy_phylo_tree_bottom_up(soft_adj_mat)
# hard_adj_mat = np.transpose(hard_adj_mat)

# tree_graph = nx.from_numpy_array(hard_adj_mat, create_using=nx.DiGraph)

soft_adj_mat[0] = 0.0
soft_adj_mat = np.concat((soft_adj_mat, np.zeros((total_nodes,leaf_nodes))), axis = 1)

print(soft_adj_mat.shape)

tree_graph, _ = construct_phylogenetic_tree(soft_adj_mat)


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

newick_str = to_newick(tree_graph, "0") + ";"

handle = StringIO(newick_str)
tree = Phylo.read(handle, "newick")
Phylo.write(tree, "inferred-tree-test.newick", "newick")
# Phylo.draw_ascii(tree)




