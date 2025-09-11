from aux_msa_functions import *
from Bio import Phylo
from joblib import Parallel, delayed
import os
from tqdm import tqdm

def rename_sim_seqs(tree_root, sim_seqs):

    output = []
    counter = [0]
    
    def rename_sim_seqs_recur(tree_root, sim_seqs):
    
        b = tree_root.clades
        
        if len(b)>0:
            for clade in b:
               rename_sim_seqs_recur(clade, sim_seqs) 
        else:
            output.append((tree_root.name,sim_seqs[counter[0]][1]))
            counter[0] += 1

    rename_sim_seqs_recur(tree_root, sim_seqs)

    return output

def rename_sim_seqs_iterator(family):

    sim_MSA_path = f"../data/msa-seed-simulations/MSA-1b/{family}/init-seq-0/logits-proposal/static-context/10/{family}-1.fasta"

    if os.path.exists(sim_MSA_path):

        sim_MSA = read_msa(sim_MSA_path)
        tree_path = f"../data/seed-trees/{family}_seed.newick"
        tree = Phylo.read(tree_path, format="newick")
        tree.root_at_midpoint()

        renamed_sim_seqs = rename_sim_seqs(tree.clade, sim_MSA)

        Seq_tuples_to_fasta(renamed_sim_seqs, sim_MSA_path)


if __name__ == "__main__":

    families = os.listdir("../data/msa-seed-simulations/MSA-1b/")

    tqdm(
        Parallel(n_jobs=50)(delayed(rename_sim_seqs_iterator)(family) for family in families),
        total = len(families)
    )






