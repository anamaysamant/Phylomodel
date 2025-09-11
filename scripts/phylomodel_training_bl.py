from typing import Sequence, TypeVar
import pandas as pd
from aux_msa_functions import *
import numpy as np
import torch

from torch.utils.data import Dataset
import pickle as pkl
from phylomodel_models import *
from select_gpu import get_free_gpu
import torch.nn as nn
import re

torch.set_grad_enabled(True)

TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)

class TreeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        X, y = self.data[0][idx], self.data[1][idx]
        return X, y

def collate_tensors(
sequences: Sequence[TensorLike], constant_value=0, dtype=None
) -> TensorLike:
    
    batch_size = len(sequences)

    X_batch, y_batch = zip(*sequences)
    X_batch, y_batch = list(X_batch), list(y_batch)
    
    shape_X = [batch_size] + np.max([mat.shape for mat in X_batch], 0).tolist()
    shape_y = [batch_size] + [shape_X[1]]

    if dtype is None:
        dtype = X_batch[0].dtype

    if isinstance(X_batch[0], np.ndarray):
        X_array = np.full(shape_X, constant_value, dtype=dtype)
    elif isinstance(X_batch[0], torch.Tensor):
        X_array = torch.full(shape_X, constant_value, dtype=dtype)

    if isinstance(y_batch[0], np.ndarray):
        y_array = np.full(shape_y, -1, dtype=y_batch[0][0].dtype)
    elif isinstance(y_batch[0], torch.Tensor):
        y_array = torch.full(shape_y, -1, dtype=y_batch[0][0].dtype)
        
    for arr, mat in zip(X_array, X_batch):
        arrslice = tuple(slice(dim) for dim in mat.shape)
        arr[arrslice] = mat

    for arr, mat in zip(y_array, y_batch):
        arrslice = tuple(slice(dim) for dim in mat.shape)
        arr[arrslice] = mat


    return X_array, y_array

with open("../data/root_distance_MSA_train_test_sets_MSA_transf_dirichlet_under_200_equal.pkl","rb") as f:
   data = pkl.load(f)

X_train = data[0]
X_test = data[1]
y_train_bl = data[2]
y_test_bl = data[3]
y_train_pc = data[4]
y_test_pc = data[5]

train_dataset = TreeDataset((X_train, y_train_bl))
test_dataset = TreeDataset((X_test, y_test_bl))


criterion = nn.MSELoss()
criterion_sum = nn.MSELoss(reduction="sum")
gpu = int(get_free_gpu())
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"


checkpoint_path = None
Large_D = 768

if checkpoint_path != None:
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    del checkpoint["model_hparams"]["output_dim"]

    model = BranchLengthPredictor(**checkpoint["model_hparams"]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters()) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    cur_epochs = checkpoint['epoch']
    batch_size = checkpoint['batch_size']
    hidden_dim = checkpoint['model_hparams']['hidden_dim']
    embed_dim = checkpoint['model_hparams']['embed_dim']
    n_heads = checkpoint['model_hparams']['n_heads']
    n_layers = checkpoint['model_hparams']['n_layers']
    Large_D = checkpoint['model_hparams']['input_dim']
    lr = checkpoint['learning_rate']

else:
    lr = 0.0001
    hidden_dim = 64
    embed_dim = 64
    n_heads = 4
    n_layers = 2
    batch_size = 5
    output_dim = Large_D
    cur_epochs = 0

    model = BranchLengthPredictor(Large_D, hidden_dim=hidden_dim, embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        collate_fn=collate_tensors,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=1, 
                                        collate_fn=collate_tensors,
                                        shuffle=False)

n_total_steps = len(train_loader)

train_epoch_losses = []
test_epoch_losses = []

num_epochs = 2

for epoch in range(cur_epochs, num_epochs):

    train_epoch_outputs = []
    train_labels = []

    model.train()

    for i, (data, labels) in enumerate(train_loader):

        n_train_nodes = 0

        total_loss = torch.tensor(0, dtype=torch.float32).to(device)

        n_int_nodes_list = []

        for j in range(len(labels)):

            total_nodes = (labels[j] != -1).sum()
            n_int_nodes = int((total_nodes + 1)/2 - 1)

            n_int_nodes_list.append(n_int_nodes) 

        n_int_nodes_vector = [torch.tensor([n_nodes] * len(labels[0])) for n_nodes in n_int_nodes_list]
        n_int_nodes_vector = torch.concat(n_int_nodes_vector)

        n_int_nodes_list = torch.tensor(n_int_nodes_list)

        for n_nodes in n_int_nodes_list.unique():

            data_j = data[n_int_nodes_list == n_nodes]
            labels_j = labels[n_int_nodes_list == n_nodes]

            msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
            msa_index_vector_j = torch.concat(msa_index_vector_j)

            seq_mask = (labels_j != -1)

            data_j = data_j[seq_mask, :].to(device)
            labels_j = labels_j[seq_mask].to(device)
            msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()].to(device)

            attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
            attn_mask = (1 - attn_mask) * -1e9

            outputs = model(data_j, attn_mask = attn_mask).squeeze(-1)

            root_mask = labels_j != 0

            cur_mse_loss = criterion_sum(outputs[root_mask], labels_j[root_mask])
            total_loss += cur_mse_loss 

            n_train_nodes += len(labels_j[root_mask])
        
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
       
        for i, (data, labels) in enumerate(train_loader):
            
            n_int_nodes_list = []

            for j in range(len(labels)):

                total_nodes = (labels[j] != -1).sum()
                n_int_nodes = int((total_nodes + 1)/2 - 1)

                n_int_nodes_list.append(n_int_nodes) 

            n_int_nodes_vector = [torch.tensor([n_nodes] * len(labels[0])) for n_nodes in n_int_nodes_list]
            n_int_nodes_vector = torch.concat(n_int_nodes_vector)

            n_int_nodes_list = torch.tensor(n_int_nodes_list)

            for n_nodes in n_int_nodes_list.unique():

                data_j = data[n_int_nodes_list == n_nodes]
                labels_j = labels[n_int_nodes_list == n_nodes]

                cur_n_train = len(data_j)

                msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
                msa_index_vector_j = torch.concat(msa_index_vector_j)

                seq_mask = (labels_j != -1)

                data_j = data_j[seq_mask, :].to(device)
                labels_j = labels_j[seq_mask].to(device)
                msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()].to(device)

                attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
                attn_mask = (1 - attn_mask) * -1e9

                outputs = model(data_j, attn_mask = attn_mask).squeeze(-1)

                root_mask = labels_j != 0

                train_epoch_loss += criterion_sum(outputs[root_mask], labels_j[root_mask])
                n_train_nodes += len(labels_j[root_mask])

        train_epoch_losses.append((train_epoch_loss/n_train_nodes).item())

    with torch.no_grad():

        test_epoch_loss = torch.tensor(0, dtype=torch.float32).to(device)
        n_test_nodes = 0

        for i, (data, labels) in enumerate(test_loader):

            n_int_nodes_list = []

            for j in range(len(labels)):

                total_nodes = (labels[j] != -1).sum()
                n_int_nodes = int((total_nodes + 1)/2 - 1)

                n_int_nodes_list.append(n_int_nodes) 

            n_int_nodes_vector = [torch.tensor([n_nodes] * len(labels[0])) for n_nodes in n_int_nodes_list]
            n_int_nodes_vector = torch.concat(n_int_nodes_vector)

            n_int_nodes_list = torch.tensor(n_int_nodes_list)

            for n_nodes in n_int_nodes_list.unique():

                data_j = data[n_int_nodes_list == n_nodes]
                labels_j = labels[n_int_nodes_list == n_nodes]

                cur_n_test = len(data_j)

                msa_index_vector_j = [torch.tensor([ind] * len(labels_j[0])) for ind in range(len(data_j))]
                msa_index_vector_j = torch.concat(msa_index_vector_j)

                seq_mask = (labels_j != -1)

                data_j = data_j[seq_mask, :].to(device)
                labels_j = labels_j[seq_mask].to(device)
                msa_index_vector_j = msa_index_vector_j[seq_mask.flatten()].to(device)

                attn_mask = (msa_index_vector_j.unsqueeze(0) == msa_index_vector_j.unsqueeze(1)).float() 
                attn_mask = (1 - attn_mask) * -1e9

                outputs = model(data_j, attn_mask = attn_mask).squeeze(-1)

                root_mask = labels_j != 0

                test_epoch_loss += criterion_sum(outputs[root_mask], labels_j[root_mask])
                n_test_nodes += len(labels_j[root_mask])

        test_epoch_losses.append((test_epoch_loss/n_test_nodes).item())

    if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:

        with open(f"bl_train_losses_{epoch + 1}epochs_{lr}lr_{batch_size}batch_under_200_eq.pkl", "wb") as f:
            pkl.dump(train_epoch_losses, f)

        with open(f"bl_test_losses_{epoch + 1}epochs_{lr}lr_{batch_size}batch_under_200_eq.pkl", "wb") as f:
            pkl.dump(test_epoch_losses,f)
            
        torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict,
                    'model_hparams': {
                    'input_dim': Large_D,
                    'hidden_dim': hidden_dim,
                    'embed_dim': embed_dim,
                    'n_heads':n_heads,
                    'n_layers':n_layers,
                    },
                    'batch_size': batch_size,
                    'learning_rate':lr
                    }, f"bl-fit-200-equal-{epoch + 1}-epochs.pt")

