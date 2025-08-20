import os
from ete3 import Tree
import numpy as np
from aux_msa_functions import *

seed = 42
subtree_size = snakemake.params["subtree_size"]

np.random.seed(seed)

tree_path = snakemake.input["tree"]
msa_path = snakemake.input["MSA"]

# output_subtree_folder = f"./data/seed-subtrees/{family}"
# output_MSA_folder = f"./data/protein-families-msa-seed-sub/{family}"

output_subtree_paths = snakemake.output["subtrees"]
output_MSA_paths = snakemake.output["subMSAs"]

num_subtrees = len(output_subtree_paths)

seed_seqs = read_msa(msa_path)
names_to_seq = dict(seed_seqs)

tree = Tree(tree_path, format=1)
leaf_names = tree.get_leaf_names()
num_leaves = len(leaf_names)

if num_leaves < subtree_size:
    subtree_size = int(max(5,num_leaves/2))

for i in range(num_subtrees):

    output_subtree_path = output_subtree_paths[i]
    output_MSA_path = output_MSA_paths[i]

    subtree_leaves_ind = np.random.choice(range(num_leaves), subtree_size, replace = False)
    subtree_leaves = [leaf_names[j] for j in subtree_leaves_ind]

    sub_tree = tree.copy()
    sub_tree.prune(subtree_leaves, preserve_branch_length=True)
    sub_tree.write(format=1, outfile=output_subtree_path)

    preordered_subtree_leaves =  sub_tree.get_leaf_names()
    sub_MSA = [(seq_name, names_to_seq[seq_name]) for seq_name in preordered_subtree_leaves]

    Seq_tuples_to_fasta(sub_MSA, output_MSA_path)
    

