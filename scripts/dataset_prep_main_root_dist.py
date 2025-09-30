import pickle as pkl
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
import numpy as np

method = "root_distance"

if method == "root_distance":
    from dataset_prep_aux_fns_root_dist import *
else:
    from dataset_prep_aux_fns_alternate import *

# torch.set_num_threads(1) 

np.random.seed(42)

Large_D = 768

X_train = []
X_test = []

y_train_bl = []
y_test_bl = []

y_train_rd = []
y_test_rd = []

with open("../data/families_under_200_over_50.pkl","rb") as f:

    all_families_init = pkl.load(f)

all_families = []

for family in all_families_init:

    if os.path.exists(f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"):
        all_families.append(family)

# all_families = all_families[:10]

gpu = str(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
msa_transf, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transf = msa_transf.to(device)
batch_converter = alphabet.get_batch_converter()
msa_transf.eval()

# num_subtrees = len(os.listdir(f"../data/msa-seed-simulations-subtrees-equal-size/50/{all_families[0]}/"))

# train_split = 0.9
# train_size = int(np.ceil(train_split * len(all_families)))

# train_families_ind = np.random.choice(range(len(all_families)), train_size, replace = False)
# train_families = [all_families[ind] for ind in train_families_ind]

# test_families_ind = list(set(range(len(all_families))) - set(train_families_ind))
# test_families = [all_families[ind] for ind in test_families_ind]

# train_fam_reps = []

# for family in train_families:
#     train_fam_reps += [family] * num_subtrees

# ind_reps = list(range(1, num_subtrees + 1)) * len(train_families)

# train_ids = [(train_fam_reps[i], ind_reps[i]) for i in range(len(ind_reps))]

# test_fam_reps = []

# for family in test_families:
#     test_fam_reps += [family] * num_subtrees

# ind_reps = list(range(1, num_subtrees + 1)) * len(test_families)

# test_ids = [(test_fam_reps[i], ind_reps[i]) for i in range(len(ind_reps))]

all_families = ["PF00004"]

num_subtrees = len(os.listdir(f"../data/msa-seed-simulations-subtrees-equal-size/50/{all_families[0]}/"))

data_ids = [(all_families[0], i) for i in list(range(1,num_subtrees))]

train_split = 0.9
train_size = int(np.ceil(train_split * len(data_ids)))

train_ids_ind = np.random.choice(range(len(data_ids)), train_size, replace = False)
train_ids = [data_ids[ind] for ind in train_ids_ind]

test_ids_ind = list(set(range(len(data_ids))) - set(train_ids_ind))
test_ids = [data_ids[ind] for ind in test_ids_ind]

train_leaf_embeds = prepare_initial_leaf_embeddings(train_ids, Large_D = Large_D, model = msa_transf, batch_converter = batch_converter, device = device)
test_leaf_embeds = prepare_initial_leaf_embeddings(test_ids, Large_D = Large_D, model = msa_transf, batch_converter = batch_converter, device = device)

train_res = list(
    tqdm(
        Parallel(return_as="generator", n_jobs=30)(
            delayed(make_datasets)(train_id) for i,train_id in enumerate(train_ids)
        ),
        total=len(train_ids),
    )
)

test_res = list(
    tqdm(
        Parallel(return_as="generator", n_jobs=30)(
            delayed(make_datasets)(test_id) for i, test_id in enumerate(test_ids)
        ),
        total=len(test_ids),
    )
)

X_train = [train_leaf_embeds[i] for i in range(len(train_res)) if train_res[i][0] != None]
y_train_bl = [item[0] for item in train_res if item[0] != None]
y_train_rd = [item[1] for item in train_res if item[1] != None] 

X_test = [test_leaf_embeds[i] for i in range(len(test_res)) if test_res[i][0] != None]
y_test_bl = [item[0] for item in test_res if item[0] != None]
y_test_rd = [item[1] for item in test_res if item[1] != None]

with open(f"train_test_sets_leaves_PF00004_size_50_rd.pkl","wb") as f:
    pkl.dump([X_train, X_test, y_train_bl, y_test_bl, y_train_rd, y_test_rd], f)

    