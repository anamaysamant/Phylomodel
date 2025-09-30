import os
from ete3 import Tree
from Bio import Phylo
import numpy as np
from aux_msa_functions import *

seed = 42

np.random.seed(seed)



msa_type = "seed"
fam = "PF00271"
msa_path = f"../data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-0/logits-proposal/static-context/10/{fam}-1.fasta"
tree_path = f"../data/{msa_type}-trees/{fam}_{msa_type}.newick"

subtree_size = 4
num_subtrees = 20000

os.makedirs(f"../data/msa-seed-simulations-subtrees-equal-size/{subtree_size}/{fam}", exist_ok=True)
os.makedirs(f"../data/submsa-seed-simulations-equal-size/{subtree_size}/{fam}", exist_ok=True)

output_subtree_paths = [f"../data/msa-seed-simulations-subtrees-equal-size/{subtree_size}/{fam}/subtree-{sim_ind}.newick" for sim_ind in range(1,num_subtrees + 1)]
output_MSA_paths = [f"../data/submsa-seed-simulations-equal-size/{subtree_size}/{fam}/submsa-{sim_ind}.fasta" for sim_ind in range(1,num_subtrees + 1)]

# tree_path = snakemake.input["tree"]
# msa_path = snakemake.input["MSA"]

# subtree_size = int(snakemake.wildcards["nseqs"])

# output_subtree_paths = snakemake.output["subtrees"]
# output_MSA_paths = snakemake.output["subMSAs"]

# num_subtrees = len(output_subtree_paths)


sim_seqs = read_msa(msa_path)
names_to_seq = dict(sim_seqs)

tree = Tree(tree_path, format=1)
leaf_names = tree.get_leaf_names()
num_leaves = len(leaf_names)

tree.set_outgroup(tree.get_midpoint_outgroup())

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
    

