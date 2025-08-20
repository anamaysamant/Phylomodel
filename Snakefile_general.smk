import numpy as np
import pickle as pkl
import os

FAMILIES = ["PF00271","PF00005","PF00004","PF01535","PF00595","PF00397","PF07679",
            "PF00072","PF00096","PF00041",
            "PF01356","PF03440","PF04008","PF06351","PF06355", "PF16747","PF18648"]
FAMILIES = ["PF00271","PF00005","PF00004","PF01535","PF00595","PF00397","PF00153","PF07679",
            "PF00076","PF00072","PF00096","PF00512","PF00041","PF13354","PF02518",
            "PF01356","PF03440","PF04008","PF06351","PF06355", "PF16747","PF18648"]
FAMILIES = ["PF00271"]
MSA_TYPES = ["seed"]
N_MUTATIONS = ["500"]
N_SEQUENCES = ["50"]
INIT_SEQS = ["0",]

num_simulations = 10

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

SEQ_LEN_LIM = 500

SEQ_DEPTH_LIM = 500

with open("./data/families_under_500_over_10.pkl","rb") as f:

    FAMILIES = pkl.load(f)

FAMILIES = ["PF00004"]

rule all:
    input:
        # expand("scores/protein-families-msa-{msa_type}/{fam}_{msa_type}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES)
        # expand("data/{msa_type}-trees/{fam}_{msa_type}.newick",
        #         msa_type = MSA_TYPES, fam= FAMILIES)
        # expand("data/bootstrap-{msa_type}-trees/{fam}/bootstrap-{sim_ind}.newick",
        #       msa_type = MSA_TYPES, fam= FAMILIES, sim_ind = SIM_INDS),
        # expand("scores/bootstrap-{msa_type}-trees/{fam}/bootstrap-scores.tsv",
        #         msa_type = MSA_TYPES, fam= FAMILIES, sim_ind = SIM_INDS),
        # expand("scores/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, fam= FAMILIES, sim_ind = SIM_INDS),
        # expand("data/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.fasta",
        #         msa_type = MSA_TYPES, fam= FAMILIES, sim_ind = SIM_INDS)
        expand("data/{msa_type}-subtrees/{fam}/subtree-{sim_ind}.newick",
                sim_ind = SIM_INDS, msa_type = MSA_TYPES, fam = FAMILIES), 
        expand("data/protein-families-submsa-{msa_type}/{fam}/submsa-{sim_ind}.fasta",
                sim_ind = SIM_INDS, msa_type = MSA_TYPES, fam = FAMILIES)
        
rule generate_subtrees:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        subtrees=expand("data/{msa_type}-subtrees/{fam}/subtree-{sim_ind}.newick",
                sim_ind = SIM_INDS, allow_missing = True), 
        subMSAs=expand("data/protein-families-submsa-{msa_type}/{fam}/submsa-{sim_ind}.fasta",
                sim_ind = SIM_INDS, allow_missing = True),
    params:
        subtree_size=20
    script:
        "scripts/dataset_subtree_augment.py"

rule generate_tree:
    input:
        "data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta"
    output:
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    log:
        "logs/generate-{msa_type}-trees/{fam}.log"
    shell:
        "fasttree {input} > {output}"

rule generate_bootstrap_MSA:
    input:
        "data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.fasta"
    script:
        "scripts/make_bootstrap_trees.py"

rule generate_bootstrap_tree:
    input:
        "data/bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.fasta"
    output:
        "data/bootstrap-{msa_type}-trees/{fam}/bootstrap-{sim_ind}.newick"
    shell:
        "fasttree {input} > {output}"

rule generate_sequences_bootstrap:
    input:
        "data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
    output:
        "data/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.fasta"
    script:
        "scripts/make_sequence_bootstraps.py"

rule generate_scores_sequences_bootstrap_MSA:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path = "data/pdb-ref-structures/{fam}-ref.pdb" 

    output:
        ungapped_seq=temp("data/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.tbl"),
        score_table="scores/sequences-bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --pdb_path {input.pdb_path}
        """

rule generate_tree_metrics_bootstrap:
    input:
        simulated_trees=expand("data/bootstrap-{msa_type}-trees/{fam}/bootstrap-{sim_ind}.newick",
                sim_ind = SIM_INDS, msa_type = MSA_TYPES, allow_missing = True), 
        simulated_MSAs=expand("data/bootstrap-{msa_type}-msa/{fam}/bootstrap-{sim_ind}.fasta",
                sim_ind = SIM_INDS, msa_type = MSA_TYPES, allow_missing = True),
        seed_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        seed_tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"     
    output:
        "scores/bootstrap-{msa_type}-trees/{fam}/bootstrap-scores.tsv"
    script:
        "scripts/tree_metrics_generator.py"

rule root_tree:
    input:
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/{msa_type}-trees/{fam}_{msa_type}_rooted.newick"
    shell:
        "python scripts/root_tree.py --input {input} --output {output}"

rule generate_scores_natural:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy",
        pdb_path = "data/pdb-ref-structures/{fam}-ref.pdb" 

    output:
        ungapped_seq=temp("data/protein-families-msa-{msa_type}/{fam}_{msa_type}-ungapped.fasta"),
        hmm_table=temp("scores/protein-families-msa-{msa_type}/{fam}_{msa_type}.tbl"),
        score_table="scores/protein-families-msa-{msa_type}/{fam}_{msa_type}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --pdb_path {input.pdb_path}
        """