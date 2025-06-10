"""Train a ChemGCN model."""

import numpy as np
import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from chem_gcn.model import ChemGCN
from chem_gcn.utils import (
    train_model,
    test_model,
    parity_plot,
    loss_curve,
    Standardizer,
)
from chem_gcn.graphs import GraphData, collate_graph_dataset

#### Fix seeds
np.random.seed(0)
torch.manual_seed(0)
use_GPU = torch.cuda.is_available()

#### Inputs
max_atoms = 200
node_vec_len = 60
train_size = 0.8
batch_size = 64
hidden_nodes = 60
n_conv_layers = 4
n_hidden_layers = 2
learning_rate = 0.001
n_epochs = 106

#### Start by creating dataset
main_path = Path(__file__).resolve().parent
data_path = main_path / "data" / "pkb_short_30000.csv"
dataset = GraphData(
    dataset_path=data_path, max_atoms=max_atoms, node_vec_len=node_vec_len
)

#### Split data into training and test sets
dataset_indices = np.arange(0, len(dataset), 1)
train_size = int(np.round(train_size * len(dataset)))
test_size = len(dataset) - train_size

train_indices = np.random.choice(dataset_indices, size=train_size, replace=False)
test_indices = np.array(list(set(dataset_indices) - set(train_indices)))

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_graph_dataset,
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=collate_graph_dataset,
)

#### Initialize model, standardizer, optimizer, and loss function
model = ChemGCN(
    node_vec_len=node_vec_len,
    node_fea_len=hidden_nodes,
    hidden_fea_len=hidden_nodes,
    n_conv=n_conv_layers,
    n_hidden=n_hidden_layers,
    n_outputs=1,
    p_dropout=0.1,
)
if use_GPU:
    model.cuda()

outputs = [dataset[i][1] for i in range(len(dataset))]
standardizer = Standardizer(torch.Tensor(outputs))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

#### Train the model with loss saving
loss = []
mae = []
epoch = []
plot_dir = main_path / "plots"
os.makedirs(plot_dir, exist_ok=True)  # 确保plots目录存在

for i in range(n_epochs):
    epoch_loss, epoch_mae = train_model(
        i,
        model,
        train_loader,
        optimizer,
        loss_fn,
        standardizer,
        use_GPU,
        max_atoms,
        node_vec_len,
        save_dir=plot_dir,  # 传递保存目录以保存每个epoch的损失
        output_file_path = main_path / "pkb_train_predictions.csv"
    )
    loss.append(epoch_loss)
    mae.append(epoch_mae)
    epoch.append(i)

#### Test the model and save predictions to CSV
output_file_path = main_path / "pkb_test_prediction.csv"
test_loss, test_mae = test_model(
    model, test_loader, loss_fn, standardizer, use_GPU, max_atoms, node_vec_len, output_file_path
)

#### Print final results
print(f"Training Loss: {loss[-1]:.2f}")
print(f"Training MAE: {mae[-1]:.2f}")
print(f"Test Loss: {test_loss:.2f}")
print(f"Test MAE: {test_mae:.2f}")

#### Generate plots and save loss data
parity_plot(plot_dir, model, test_loader, standardizer, use_GPU, max_atoms, node_vec_len)
loss_curve(plot_dir, epoch, loss, maes=mae)  # 同时保存损失和MAE曲线