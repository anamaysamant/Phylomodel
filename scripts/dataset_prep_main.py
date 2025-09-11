import pickle as pkl
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
import numpy as np

method = "root_distance"

if method == "root_distance":
    from dataset_prep_aux_fns import *
else:
    from dataset_prep_aux_fns_alternate import *

# torch.set_num_threads(1) 

np.random.seed(42)

Large_D = 768

X_train = []
X_test = []

y_train_bl = []
y_test_bl = []

y_train_pc = []
y_test_pc = []

with open("../data/families_under_200_over_50.pkl","rb") as f:

    all_families_init = pkl.load(f)

all_families = []

for family in all_families_init:

    if os.path.exists(f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"):
        all_families.append(family)

# all_families = all_families[:10]

num_subtrees = len(os.listdir(f"../data/msa-seed-simulations-subtrees-equal-size/50/{all_families[0]}/"))

train_split = 0.9
train_size = int(np.ceil(train_split * len(all_families)))

train_families_ind = np.random.choice(range(len(all_families)), train_size, replace = False)
train_families = [all_families[ind] for ind in train_families_ind]

test_families_ind = list(set(range(len(all_families))) - set(train_families_ind))
test_families = [all_families[ind] for ind in test_families_ind]

train_fam_reps = []

for family in train_families:
    train_fam_reps += [family] * num_subtrees

ind_reps = list(range(1, num_subtrees + 1)) * len(train_families)

train_ids = [(train_fam_reps[i], ind_reps[i]) for i in range(len(ind_reps))]

test_fam_reps = []

for family in test_families:
    test_fam_reps += [family] * num_subtrees

ind_reps = list(range(1, num_subtrees + 1)) * len(test_families)

test_ids = [(test_fam_reps[i], ind_reps[i]) for i in range(len(ind_reps))]

train_leaf_embeds = prepare_initial_leaf_embeddings(train_ids, Large_D = Large_D)
test_leaf_embeds = prepare_initial_leaf_embeddings(test_ids, Large_D = Large_D)

train_res = list(
    tqdm(
        Parallel(return_as="generator", n_jobs=30)(
            delayed(make_datasets)(train_id, train_leaf_embeds[i], Large_D) for i,train_id in enumerate(train_ids)
        ),
        total=len(train_ids),
    )
)

test_res = list(
    tqdm(
        Parallel(return_as="generator", n_jobs=30)(
            delayed(make_datasets)(test_id, test_leaf_embeds[i], Large_D) for i, test_id in enumerate(test_ids)
        ),
        total=len(test_ids),
    )
)

X_train = [torch.concat((train_res[i][0], train_leaf_embeds[i]), dim=0) for i in range(len(train_res)) if train_res[i][0] != None]
y_train_bl = [item[1] for item in train_res if item[1] != None]
y_train_pc = [item[2] for item in train_res if item[2] != None] 

X_test = [torch.concat((test_res[i][0], test_leaf_embeds[i]), dim=0) for i in range(len(test_res)) if test_res[i][0] != None]
y_test_bl = [item[1] for item in test_res if item[1] != None]
y_test_pc = [item[2] for item in test_res if item[2] != None]

with open(f"{method}_MSA_train_test_sets_MSA_transf_dirichlet_under_200_equal.pkl","wb") as f:
    pkl.dump([X_train, X_test, y_train_bl, y_test_bl, y_train_pc, y_test_pc], f)

    