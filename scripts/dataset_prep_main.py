import pickle as pkl
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
import numpy as np

from dataset_prep_aux_fns import *

# torch.set_num_threads(1) 

np.random.seed(42)

Large_D = 768

X_train = []
X_test = []

y_train_bl = []
y_test_bl = []

y_train_pc = []
y_test_pc = []

with open("../data/families_under_500_over_10.pkl","rb") as f:

    all_families = pkl.load(f)

train_split = 0.9
train_size = int(np.ceil(train_split * len(all_families)))

train_families_ind = np.random.choice(range(len(all_families)), train_size, replace = False)
train_families = [all_families[ind] for ind in train_families_ind]

test_families_ind = list(set(range(len(all_families))) - set(train_families_ind))
test_families = [all_families[ind] for ind in test_families_ind]

train_leaf_embeds = prepare_initial_leaf_embeddings(train_families, Large_D = Large_D, batch_size = 1)
test_leaf_embeds = prepare_initial_leaf_embeddings(test_families, Large_D = Large_D, batch_size = 1)

# train_res = Parallel(n_jobs = 30, backend="multiprocessing")(delayed(make_datasets)(family) for family in train_families)
# test_res = Parallel(n_jobs = 30, backend="multiprocessing")(delayed(make_datasets)(family) for family in test_families)

train_res = list(
    tqdm(
        Parallel(return_as="generator", n_jobs=30)(
            delayed(make_datasets)(family, train_leaf_embeds[i], Large_D) for i,family in enumerate(train_families)
        ),
        total=len(train_families),
    )
)

test_res = list(
    tqdm(
        Parallel(return_as="generator", n_jobs=30)(
            delayed(make_datasets)(family, test_leaf_embeds[i], Large_D) for i, family in enumerate(test_families)
        ),
        total=len(test_families),
    )
)

X_train = [torch.concat((train_res[i][0], train_leaf_embeds[i]), dim=0) for i in range(len(train_res))]
y_train_bl = [item[1] for item in train_res]
y_train_pc = [item[2] for item in train_res]

X_test = [torch.concat((test_res[i][0], test_leaf_embeds[i]), dim=0) for i in range(len(test_res))]
y_test_bl = [item[1] for item in test_res]
y_test_pc = [item[2] for item in test_res]

with open("train_test_sets_MSA_transf_dirichlet.pkl","wb") as f:
    pkl.dump([X_train, X_test, y_train_bl, y_test_bl, y_train_pc, y_test_pc], f)


# for i, family in enumerate(train_families):

#     true_tree_path = f"./data/seed-trees/{family}_seed.newick"
#     nat_MSA_path = f"./data/protein-families-msa-seed/{family}_seed.fasta"

#     if (i % 100) == 0:
#         print(i)
    
#     all_branch_lengths, mapped_parents, sorted_internal_nodes = prepare_branch_lengths_and_relationships(true_tree_path, nat_MSA_path)

#     all_embeddings = prepare_initial_node_embeddings(nat_MSA_path, true_tree_path, Large_D = Large_D)

#     X_train.append(all_embeddings)
#     y_train_bl.append(all_branch_lengths)
#     y_train_pc.append(mapped_parents)


# for family in test_families:

#     true_tree_path = f"./data/seed-trees/{family}_seed.newick"
#     nat_MSA_path = f"./data/protein-families-msa-seed/{family}_seed.fasta"
    
#     all_branch_lengths, mapped_parents, sorted_internal_nodes = prepare_branch_lengths_and_relationships(true_tree_path, nat_MSA_path)

#     all_embeddings = prepare_initial_node_embeddings(nat_MSA_path, true_tree_path, Large_D = Large_D)

#     X_test.append(all_embeddings)
#     y_test_bl.append(all_branch_lengths)
#     y_test_pc.append(mapped_parents)

                                  

# for family in families:

#     true_tree_path = f"./data/seed-trees/{family}_seed.newick"
#     nat_MSA_path = f"./data/protein-families-msa-seed/{family}_seed.fasta"
#     MSA_folder = f"./data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/"
    
#     all_branch_lengths, mapped_parents, sorted_internal_nodes = prepare_branch_lengths_and_relationships(true_tree_path, nat_MSA_path)

#     for file in os.listdir(MSA_folder)[-70:]:

#         file_path = os.path.join(MSA_folder, file)
        
#         all_embeddings = prepare_initial_node_embeddings(file_path, Large_D = Large_D)

#         X_train.append(all_embeddings)
#         y_train_bl.append(all_branch_lengths)
#         y_train_pc.append(mapped_parents)

#     for file in os.listdir(MSA_folder)[:-70]:

#         file_path = os.path.join(MSA_folder, file)

#         all_embeddings = prepare_initial_node_embeddings(file_path, Large_D = Large_D)

#         X_test.append(all_embeddings)
#         y_test_bl.append(all_branch_lengths)
#         y_test_pc.append(mapped_parents)

    