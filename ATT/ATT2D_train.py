import os
import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss  # Using mean squared error loss
import numpy as np
from torch.nn import Linear, BatchNorm1d, Dropout, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from ATT2D_data import PkaDataset  # Replace with actual dataset class path
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Define model
class ATT(torch.nn.Module):
    def __init__(self, n_heads=4, edge_dim=11, num_features=30, embedding_size=128, self_attention=False,
                 multihead_attention=False, return_attention=False):
        super(ATT, self).__init__()
        torch.manual_seed(42)

        self.self_attention = self_attention
        self.multihead_attention = multihead_attention
        self.return_attention = return_attention
        self.dropout_layer = Dropout(p=0.1)

        # GAT layers
        self.initial_conv = GATConv(num_features, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        self.conv1 = GATConv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform2 = Linear(embedding_size * n_heads, embedding_size)
        self.bn2 = BatchNorm1d(embedding_size)

        self.conv2 = GATConv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform3 = Linear(embedding_size * n_heads, embedding_size)
        self.bn3 = BatchNorm1d(embedding_size)

        self.conv3 = GATConv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=0)
        self.head_transform4 = Linear(embedding_size * n_heads, embedding_size)
        self.bn4 = BatchNorm1d(embedding_size)

        if self.self_attention:
            self.W_a = Linear(embedding_size, embedding_size, bias=False)
            self.W_b = Linear(embedding_size, embedding_size)

        if self.multihead_attention:
            self.mhat = MultiAtomAttention(embedding_size)

        # Output layer
        self.out1 = Linear(embedding_size * 2, embedding_size * 2)
        self.out2 = Linear(embedding_size * 2, 1)  # Output layer produces a scalar value (regression task)

    def forward(self, x, edge_index, edge_attr, batch_index):
        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform1(hidden)
        hidden = self.bn1(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform2(hidden)
        hidden = self.bn2(hidden)

        hidden = self.conv2(hidden, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform3(hidden)
        hidden = self.bn3(hidden)

        hidden = self.conv3(hidden, edge_index, edge_attr)
        hidden = F.tanh(hidden)
        hidden = self.head_transform4(hidden)
        hidden = self.bn4(hidden)

        if self.self_attention:
            graph_length = [0] * (batch_index[-1] + 1)
            for i in range(len(batch_index)):
                graph_length[batch_index[i]] += 1

            mol_vecs = []
            attention_weights = []
            start = 0
            for length in graph_length:
                current_hidden = torch.narrow(hidden, 0, start, length)
                att_w = torch.matmul(self.W_a(current_hidden), current_hidden.T)
                att_w = F.softmax(att_w, dim=1)
                att_hiddens = torch.matmul(att_w, current_hidden)
                att_hiddens = F.relu(self.W_b(att_hiddens))
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = current_hidden + att_hiddens
                mol_vecs.append(mol_vec)
                attention_weights.append(att_w)
                start += length
            hidden = torch.cat(mol_vecs, dim=0)

        if self.multihead_attention:
            graph_length = [0] * (batch_index[-1] + 1)
            for i in range(len(batch_index)):
                graph_length[batch_index[i]] += 1

            mol_vecs = []
            attention_weights = []
            start = 0
            for length in graph_length:
                current_hidden = torch.narrow(hidden, 0, start, length)
                att_hiddens, multi_att_w = self.mhat(current_hidden)
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = current_hidden + att_hiddens
                mol_vecs.append(mol_vec)
                attention_weights.append(multi_att_w)
                start += length
            hidden = torch.cat(mol_vecs, dim=0)

        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        out = F.relu(self.out1(hidden))
        out = self.out2(out)  # Output a scalar value

        if (self.self_attention or self.multihead_attention) and self.return_attention:
            return out, hidden, attention_weights

        return out, hidden


class MultiAtomAttention(torch.nn.Module):
    def __init__(self, embedding_size):
        super(MultiAtomAttention, self).__init__()
        self.embedding_size = embedding_size
        self.dropout_layer = Dropout(p=0.2)
        self.num_heads = 4
        self.att_size = self.embedding_size // self.num_heads
        self.scale_factor = self.att_size ** -0.5

        self.W_a_q = Linear(self.embedding_size, self.num_heads * self.att_size, bias=False)
        self.W_a_k = Linear(self.embedding_size, self.num_heads * self.att_size, bias=False)
        self.W_a_v = Linear(self.embedding_size, self.num_heads * self.att_size, bias=False)
        self.W_a_o = Linear(self.num_heads * self.att_size, self.embedding_size)
        self.norm = LayerNorm(self.embedding_size, elementwise_affine=True)

    def forward(self, x):
        cur_embedding_size = x.size()

        a_q = self.W_a_q(x).view(cur_embedding_size[0], self.num_heads, self.att_size)
        a_k = self.W_a_k(x).view(cur_embedding_size[0], self.num_heads, self.att_size)
        a_v = self.W_a_v(x).view(cur_embedding_size[0], self.num_heads, self.att_size)

        a_q = a_q.transpose(0, 1)
        a_k = a_k.transpose(0, 1).transpose(1, 2)
        a_v = a_v.transpose(0, 1)

        att_a_w = torch.matmul(a_q, a_k)
        att_a_w = F.softmax(att_a_w * self.scale_factor, dim=2)
        att_a_h = torch.matmul(att_a_w, a_v)
        att_a_h = F.relu(att_a_h)
        att_a_h = self.dropout_layer(att_a_h)

        att_a_h = att_a_h.transpose(0, 1).contiguous()
        att_a_h = att_a_h.view(cur_embedding_size[0], self.num_heads * self.att_size)
        att_a_h = self.W_a_o(att_a_h)
        assert att_a_h.size() == cur_embedding_size

        att_a_h = att_a_h.unsqueeze(dim=0)
        att_a_h = self.norm(att_a_h)
        mol_vec = att_a_h.squeeze(dim=0)

        return mol_vec, torch.mean(att_a_w, axis=0)


# Fine-tuning component
class ATT_fine(torch.nn.Module):
    def __init__(self, pretrain_net):
        super(ATT_fine, self).__init__()
        torch.manual_seed(42)

        self.class_embedding_size = 32
        self.pretrain_net = pretrain_net

        # For regression
        self.linear1 = Linear(256, self.class_embedding_size)
        self.dropout = Dropout(0.2)
        self.bnL1 = BatchNorm1d(self.class_embedding_size)
        self.linear2 = Linear(self.class_embedding_size, 1)  # Output a scalar value

    def forward(self, x, edge_attr, edge_index, batch_index):
        pretrain_net_outputs = self.pretrain_net(x, edge_attr, edge_index, batch_index)
        if len(pretrain_net_outputs) == 2:
            pretrain_pred, pretrain_embedding = pretrain_net_outputs
        elif len(pretrain_net_outputs) == 3:
            pretrain_pred, pretrain_embedding, _ = pretrain_net_outputs
        else:
            raise Exception("Wrong outputs from pre-trained network")

        pretrain_embedding = self.dropout(pretrain_embedding)

        # Regression head
        class_hidden = self.linear1(pretrain_embedding).relu()
        class_hidden = self.bnL1(class_hidden)
        class_hidden = self.dropout(class_hidden)
        class_out = self.linear2(class_hidden)  # Output a scalar value

        return class_out, pretrain_pred


# Data loading
def load_data(root, filename, batch_size=64, train_ratio=0.8, test_ratio=0.1):
    """
    Load dataset and split according to specified ratios
    Args:
        root: Data root directory
        filename: Data filename
        batch_size: Batch size
        train_ratio: Training set ratio (default 0.8)
        test_ratio: Test set ratio (default 0.1)
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = PkaDataset(root=root, filename=filename)

    # Calculate dataset sizes
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    val_size = len(dataset) - train_size - test_size

    # Ensure test set is the last 10% of data
    train_dataset = dataset[:train_size]  # First 80%
    val_dataset = dataset[train_size:train_size + val_size]  # Middle 10%
    test_dataset = dataset[-test_size:]  # Last 10% (forced as test set)

    print(f"Dataset split: Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model_with_metrics(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_true = []
    all_smiles = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = loss_fn(out.squeeze(), data.y.float())
            total_loss += loss.item()

            all_preds.extend(out.squeeze().cpu().numpy())
            all_true.extend(data.y.cpu().numpy())
            all_smiles.extend(data.smiles)  # Assuming dataset returns smiles attribute

    avg_loss = total_loss / len(loader)
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    r2 = r2_score(all_true, all_preds)

    return avg_loss, rmse, r2, all_preds, all_true, all_smiles


def train_model(model, train_loader, val_loader, test_loader, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'val_rmse': [],
        'val_r2': [],
        'test_rmse': [],
        'test_r2': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training phase
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = loss_fn(out.squeeze(), data.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        val_loss, val_rmse, val_r2, _, _, _ = evaluate_model_with_metrics(
            model, val_loader, loss_fn, device)

        # Test phase (every 5 epochs or at the end)
        if epoch % 5 == 0 or epoch == epochs - 1:
            test_loss, test_rmse, test_r2, _, _, _ = evaluate_model_with_metrics(
                model, test_loader, loss_fn, device)
        else:
            test_loss, test_rmse, test_r2 = float('nan'), float('nan'), float('nan')

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['test_rmse'].append(test_rmse)
        history['test_r2'].append(test_r2)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
        print(f"Test Loss: {test_loss:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model_temp.pth')

    # Load best model and generate final predictions
    model.load_state_dict(torch.load('best_model_temp.pth'))
    os.remove('best_model_temp.pth')

    # Save training history
    pd.DataFrame(history).to_csv('training_log_bpka_2d.csv', index=False)

    # Generate and save all predictions
    def save_predictions(loader, save_path):
        _, _, _, preds, trues, smiles = evaluate_model_with_metrics(
            model, loader, loss_fn, device)
        df = pd.DataFrame({
            'SMILES': smiles,
            'True': trues,
            'Predicted': preds
        })
        df.to_csv(save_path, index=False)

    save_predictions(train_loader, 'train_predictions_2d_bpka.csv')
    save_predictions(val_loader, 'val_predictions_2d_bpka.csv')
    save_predictions(test_loader, 'test_predictions_2d_bpka.csv')

    # Save final model
    torch.save(model.pretrain_net.state_dict(), 'best_model_bpka_2d.pth')
    print("Model and predictions saved")


if __name__ == "__main__":
    # Data paths
    root = "/home/lxy/ToxPred_nitrification-main/data/bpka_2d"
    filename = "bpka_30000.csv"

    # Load data
    # Explicitly specify ratios (first 80% train, middle 10% val, last 10% test)
    train_loader, val_loader, test_loader = load_data(
        root, filename,
        batch_size=64,
        train_ratio=0.8,
        test_ratio=0.1
    )

    # Initialize model
    pretrain_net = ATT(
        n_heads=4,
        edge_dim=11,
        num_features=30,
        multihead_attention=True,
        return_attention=True
    )
    model = ATT_fine(pretrain_net)

    # Train model
    train_model(model, train_loader, val_loader, test_loader, epochs=100, lr=0.001)