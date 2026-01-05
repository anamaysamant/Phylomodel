import pickle as pkl
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
import numpy as np

from dataset_prep_aux_fns_root_dist_proto import *

np.random.seed(42)

Large_D = 768
few_families = False
num_subtrees = 10

with open("../data/families_under_200_over_50.pkl","rb") as f:

    all_families_init = pkl.load(f)

all_families = []

for family in all_families_init:

    if os.path.exists(f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"):
        all_families.append(family)


family_subsets = {}

gpu = str(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
msa_transf, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transf = msa_transf.to(device)
batch_converter = alphabet.get_batch_converter()
msa_transf.eval()

train_split = 0.9
train_size = int(np.ceil(train_split * len(all_families)))

train_families_ind = np.random.choice(range(len(all_families)), train_size, replace = False)
train_families = [all_families[ind] for ind in train_families_ind]

family_subsets["train"] = train_families

test_families_ind = list(set(range(len(all_families))) - set(train_families_ind))
test_families = [all_families[ind] for ind in test_families_ind]

family_subsets["test"] = test_families

 
datasets = {}

for dataset_type in ["train","test"]:

    cur_families = family_subsets[dataset_type]
    # cur_families = ["PF00004"]

    X_all_fams = []
    y_bl_all_fams = []
    y_rd_all_fams = []

    for fam in tqdm(cur_families):
        
        if few_families:
            msas_folder = f"../data/submsa-seed-simulations-equal-size/4/{dataset_type}/{fam}/"
        else:
            msas_folder = f"../data/submsa-seed-simulations-equal-size/4/{fam}/"

        all_msa_names = sorted(os.listdir(msas_folder))
        num_subtrees = num_subtrees if num_subtrees else len(all_msa_names)
        msa_names = all_msa_names[:num_subtrees]
        msa_paths = [os.path.join(msas_folder, msa) for msa in msa_names]

        if few_families:
            trees_folder = f"../data/msa-seed-simulations-subtrees-equal-size/4/{dataset_type}/{fam}/"
        else:
            trees_folder = f"../data/msa-seed-simulations-subtrees-equal-size/4/{fam}/"

        all_tree_names = sorted(os.listdir(trees_folder))
        num_subtrees = num_subtrees if num_subtrees else len(all_tree_names)
        tree_names = sorted(os.listdir(trees_folder))[:num_subtrees]
        tree_paths = [os.path.join(trees_folder, tree) for tree in tree_names]

        assert [x.split(".")[-2].split("-")[-1] for x in tree_names] == [x.split(".")[-2].split("-")[-1] for x in msa_names]

        leaf_embeds = prepare_initial_leaf_embeddings(msa_paths, Large_D = Large_D, model = msa_transf, batch_converter = batch_converter, device = device, use_transf = True, alphabet = alphabet)

        data_res = list(
            tqdm(
                Parallel(return_as="generator", n_jobs=30)(
                    delayed(make_datasets)(msa_path, tree_path) for msa_path,tree_path in zip(msa_paths, tree_paths)
                ),
                total=len(msa_paths),
            )
        )

        
        X = [leaf_embeds[i] for i in range(len(data_res)) if data_res[i][0] != None]
        y_bl = [item[0] for item in data_res if item[0] != None]
        y_rd = [item[1] for item in data_res if item[1] != None] 

        X_all_fams.extend(X)
        y_bl_all_fams.extend(y_bl)
        y_rd_all_fams.extend(y_rd)

    datasets[f"X_{dataset_type}"] = X_all_fams
    datasets[f"y_{dataset_type}_bl"] = y_bl_all_fams
    datasets[f"y_{dataset_type}_rd"] = y_rd_all_fams


with open(f"train_test_sets_leaves_under_200_10_reps_size_4_rd.pkl","wb") as f:
    pkl.dump(datasets, f)



    