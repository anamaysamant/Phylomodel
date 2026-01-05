import os
from ete3 import Tree
from Bio import Phylo
import numpy as np
from aux_msa_functions import *
import pickle as pkl
from tqdm import tqdm

seed = 42

np.random.seed(seed)

subtree_size = 50
num_subtrees_train = 18000
num_subtrees_test = 2000
initial_train_test_split = 0.7
msa_type = "seed"

with open("../data/families_under_200_over_50.pkl","rb") as f:

    all_families_init = pkl.load(f)

all_families = []

for family in all_families_init:

    if os.path.exists(f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"):
        all_families.append(family)

all_families = ["PF00004"]

for fam in tqdm(all_families):
    
    msa_path = f"../data/msa-{msa_type}-simulations/MSA-1b/{fam}/init-seq-0/logits-proposal/static-context/10/{fam}-1.fasta"
    tree_path = f"../data/{msa_type}-trees/{fam}_{msa_type}.newick"

    subtree_dir_train = f"../data/msa-seed-simulations-subtrees-equal-size/{subtree_size}/train/{fam}"
    submsa_dir_train = f"../data/submsa-seed-simulations-equal-size/{subtree_size}/train/{fam}"

    if os.path.exists(subtree_dir_train):

        num_existing_subtrees = len(os.listdir(subtree_dir_train))
        if num_subtrees_train < num_existing_subtrees:
            continue
    else:
        os.makedirs(subtree_dir_train, exist_ok=True)
        os.makedirs(submsa_dir_train, exist_ok=True)

    output_subtree_paths_train = [f"../data/msa-seed-simulations-subtrees-equal-size/{subtree_size}/train/{fam}/subtree-{sim_ind}.newick" for sim_ind in range(1,num_subtrees_train + 1)]
    output_MSA_paths_train = [f"../data/submsa-seed-simulations-equal-size/{subtree_size}/train/{fam}/submsa-{sim_ind}.fasta" for sim_ind in range(1,num_subtrees_train + 1)]

    subtree_dir_test = f"../data/msa-seed-simulations-subtrees-equal-size/{subtree_size}/test/{fam}"
    submsa_dir_test = f"../data/submsa-seed-simulations-equal-size/{subtree_size}/test/{fam}"

    if os.path.exists(subtree_dir_test):

        num_existing_subtrees = len(os.listdir(subtree_dir_test))
        if num_subtrees_test < num_existing_subtrees:
            continue
    else:
        os.makedirs(subtree_dir_test, exist_ok=True)
        os.makedirs(submsa_dir_test, exist_ok=True)

    output_subtree_paths_test = [f"../data/msa-seed-simulations-subtrees-equal-size/{subtree_size}/test/{fam}/subtree-{sim_ind}.newick" for sim_ind in range(1,num_subtrees_test + 1)]
    output_MSA_paths_test = [f"../data/submsa-seed-simulations-equal-size/{subtree_size}/test/{fam}/submsa-{sim_ind}.fasta" for sim_ind in range(1,num_subtrees_test + 1)]

    sim_seqs = read_msa(msa_path)
    names_to_seq = dict(sim_seqs)

    tree = Tree(tree_path, format=1)
    leaf_names = tree.get_leaf_names()
    num_leaves = len(leaf_names)

    train_leaf_inds = np.random.choice(range(num_leaves), int(initial_train_test_split * num_leaves), replace = False)
    test_leaf_inds = list(set(range(num_leaves)) - set(train_leaf_inds))

    tree.set_outgroup(tree.get_midpoint_outgroup())

    for i in tqdm(range(num_subtrees_train)):

        output_subtree_path = output_subtree_paths_train[i]
        output_MSA_path = output_MSA_paths_train[i]

        subtree_leaves_ind = np.random.choice(train_leaf_inds, subtree_size, replace = False)
        subtree_leaves = [leaf_names[j] for j in subtree_leaves_ind]

        sub_tree = tree.copy()
        sub_tree.prune(subtree_leaves, preserve_branch_length=True)
        sub_tree.write(format=1, outfile=output_subtree_path)

        preordered_subtree_leaves =  sub_tree.get_leaf_names()
        sub_MSA = [(seq_name, names_to_seq[seq_name]) for seq_name in preordered_subtree_leaves]

        Seq_tuples_to_fasta(sub_MSA, output_MSA_path)

    for i in tqdm(range(num_subtrees_test)):

        output_subtree_path = output_subtree_paths_test[i]
        output_MSA_path = output_MSA_paths_test[i]

        subtree_leaves_ind = np.random.choice(test_leaf_inds, subtree_size, replace = False)
        subtree_leaves = [leaf_names[j] for j in subtree_leaves_ind]

        sub_tree = tree.copy()
        sub_tree.prune(subtree_leaves, preserve_branch_length=True)
        sub_tree.write(format=1, outfile=output_subtree_path)

        preordered_subtree_leaves =  sub_tree.get_leaf_names()
        sub_MSA = [(seq_name, names_to_seq[seq_name]) for seq_name in preordered_subtree_leaves]

        Seq_tuples_to_fasta(sub_MSA, output_MSA_path)
    



