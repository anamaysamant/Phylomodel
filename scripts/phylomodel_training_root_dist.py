from typing import Sequence, TypeVar
import pandas as pd
from aux_msa_functions import *
import numpy as np
import torch

from torch.utils.data import Dataset
import pickle as pkl
from phylomodel_models_root_distance import *
from select_gpu import get_free_gpu
import torch.nn as nn
import re
from phylomodel_training_aux_fns import *

torch.set_grad_enabled(True)

TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)

def generate_model_outputs(data, branch_length_labels, root_distance_labels):

    msa_index_vector = [torch.tensor([ind] * len(branch_length_labels[0])) for ind in range(len(data))]
    msa_index_vector = torch.concat(msa_index_vector)

    y_mask = (branch_length_labels != -1)

    x_mask = (data != 0).all(dim=2)

    data = data[x_mask, :].to(device)

    if y_mask.sum().item() != 2 * data.shape[0] - 1:
        return None, None, None, None
    
    branch_length_labels = branch_length_labels[y_mask].to(device)
    root_distance_labels = root_distance_labels[y_mask].to(device)

    msa_index_vector = msa_index_vector[y_mask.flatten()].to(device)

    attn_mask = (msa_index_vector.unsqueeze(0) == msa_index_vector.unsqueeze(1)).float() 
    attn_mask = (1 - attn_mask) * -1e9

    branch_length_preds, root_distance_preds = model(data, attn_mask = attn_mask)

    branch_length_preds = branch_length_preds.squeeze(-1)
    root_distance_preds = root_distance_preds.squeeze(-1)

    root_mask = branch_length_labels != 0

    return branch_length_preds[root_mask], root_distance_preds[root_mask], branch_length_labels[root_mask], root_distance_labels[root_mask]

with open("../data/processed-train-test-data/train_test_sets_leaves_PF00004_size_50_rd.pkl","rb") as f:
   data = pkl.load(f)

X_train = data[0]
X_test = data[1]
y_train_bl = data[2]
y_test_bl = data[3]
y_train_rd = data[4]
y_test_rd = data[5]

train_dataset = TreeDatasetRD((X_train, y_train_bl, y_train_rd))
test_dataset = TreeDatasetRD((X_test, y_test_bl, y_test_rd))

criterion = nn.MSELoss()
criterion_sum = nn.MSELoss(reduction="sum")
gpu = int(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

checkpoint_path = None

if checkpoint_path != None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model = MainPhyloModel(**checkpoint["model_hparams"]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters()) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    cur_epochs = checkpoint['epoch']
    batch_size = checkpoint['batch_size']
    lr = checkpoint['learning_rate']
    bl_weight = checkpoint['bl_weight']

    input_dim = checkpoint['model_hparams']['input_dim']
    hidden_dim_base = checkpoint['model_hparams']['hidden_dim_base']
    attn_dim_main = checkpoint['model_hparams']['hidden_dim_base']
    n_heads_main = checkpoint['model_hparams']['n_heads_main']
    n_layers_main = checkpoint['model_hparams']['n_layers_main']
    attn_dim_int = checkpoint['model_hparams']['attn_dim_int']
    n_heads_int = checkpoint['model_hparams']['n_heads_int']
    n_layers_int = checkpoint['model_hparams']['n_layers_int']
    output_dim = checkpoint['model_hparams']['output_dim']

    hidden_dim_head = checkpoint['model_hparams']['hidden_dim_head']
    n_layers_bl = checkpoint['model_hparams']['n_layers_bl']
    n_layers_rd = checkpoint['model_hparams']['n_layers_rd']

   

else:
    lr = 0.0001
    batch_size = 1
    cur_epochs = 0

    input_dim = 768
    hidden_dim_base = 64
    attn_dim_main = 64
    n_heads_main = 4
    n_layers_main = 2
    attn_dim_int = 32
    n_heads_int = 2
    n_layers_int = 1
    output_dim = 768

    hidden_dim_head = 64
    n_layers_bl = 1
    n_layers_rd = 1

    bl_weight = 0.91

    model = MainPhyloModel(input_dim, hidden_dim_base, attn_dim_main, n_heads_main, n_layers_main,
                  attn_dim_int, n_heads_int, n_layers_int, output_dim, 
                  hidden_dim_head, n_layers_bl, n_layers_rd).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        collate_fn=collate_tensors_rd,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=1, 
                                        collate_fn=collate_tensors_rd,
                                        shuffle=False)

n_total_steps = len(train_loader)

train_epoch_losses = []
test_epoch_losses = []

num_epochs = 50

for epoch in range(cur_epochs, num_epochs):

    train_epoch_outputs = []
    train_branch_length_labels = []

    model.train()

    for i, (data, branch_length_labels, root_distance_labels) in enumerate(train_loader):

        branch_length_preds, root_distance_preds, branch_length_labels, root_distance_labels = generate_model_outputs(data, branch_length_labels, root_distance_labels)

        if branch_length_preds == None:
            continue

        bl_loss = criterion_sum(branch_length_preds, branch_length_labels)
        rd_loss = criterion_sum(root_distance_preds, root_distance_labels)

        total_loss = bl_weight * bl_loss + (1 - bl_weight) * rd_loss
        n_train_nodes = len(branch_length_labels)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], mean MSE Loss: {(total_loss/n_train_nodes).item():.4f}')

    scheduler.step()

    model.eval()

    with torch.no_grad():

        train_epoch_loss = torch.tensor(0, dtype=torch.float32).to(device)
        n_train_nodes = 0
       
        for i, (data, branch_length_labels, root_distance_labels) in enumerate(train_loader):

            branch_length_preds, root_distance_preds, branch_length_labels, root_distance_labels = generate_model_outputs(data, branch_length_labels, root_distance_labels)

            if branch_length_preds == None:
                continue

            bl_loss = criterion_sum(branch_length_preds, branch_length_labels)
            rd_loss = criterion_sum(root_distance_preds, root_distance_labels)

            total_loss = bl_weight * bl_loss + (1 - bl_weight) * rd_loss
            train_epoch_loss += total_loss

            n_train_nodes += len(branch_length_labels)

        train_epoch_losses.append((train_epoch_loss/n_train_nodes).item())

    with torch.no_grad():

        test_epoch_loss = torch.tensor(0, dtype=torch.float32).to(device)
        n_test_nodes = 0

        for i, (data, branch_length_labels, root_distance_labels) in enumerate(test_loader):

            branch_length_preds, root_distance_preds, branch_length_labels, root_distance_labels = generate_model_outputs(data, branch_length_labels, root_distance_labels)

            if branch_length_preds == None:
                continue

            bl_loss = criterion_sum(branch_length_preds, branch_length_labels)
            rd_loss = criterion_sum(root_distance_preds, root_distance_labels)

            total_loss = bl_weight * bl_loss + (1 - bl_weight) * rd_loss
            test_epoch_loss += total_loss
            n_test_nodes += len(branch_length_labels)

        test_epoch_losses.append((test_epoch_loss/n_test_nodes).item())

    if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:

        with open(f"train_losses_{epoch + 1}epochs_{lr}lr_{batch_size}batch_under_200_eq.pkl", "wb") as f:
            pkl.dump(train_epoch_losses, f)

        with open(f"test_losses_{epoch + 1}epochs_{lr}lr_{batch_size}batch_under_200_eq.pkl", "wb") as f:
            pkl.dump(test_epoch_losses,f)
            
        torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'batch_size': batch_size,
                    'learning_rate':lr,
                    'bl_weight': bl_weight,
                    'model_hparams': {
                    'input_dim': input_dim,
                    'hidden_dim_base': hidden_dim_base,
                    'attn_dim_main': attn_dim_main,
                    'n_heads_main':n_heads_main,
                    'n_layers_main':n_layers_main,
                    'attn_dim_int': attn_dim_int,
                    'n_layers_int': n_layers_int,
                    'n_heads_int': n_heads_int,
                    'output_dim':output_dim,
                    'hidden_dim_head':hidden_dim_head,
                    'n_layers_bl':n_layers_bl,
                    'n_layers_rd':n_layers_rd,
                    }
                    }, f"fit-200-equal-{epoch + 1}-epochs.pt")

