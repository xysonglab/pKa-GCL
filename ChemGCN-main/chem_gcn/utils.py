"""Utility functions."""

import os
import pandas as pd  # 导入pandas库
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Standardizer:
    def __init__(self, X):
        """
        Class to standardize ChemGCN outputs

        Parameters
        ----------
        X : torch.Tensor
            Tensor of outputs
        """
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        """
        Convert a non-standardized output to a standardized output

        Parameters
        ----------
        X : torch.Tensor
            Tensor of non-standardized outputs

        Returns
        -------
        Z : torch.Tensor
            Tensor of standardized outputs

        """
        Z = (X - self.mean) / (self.std)
        return Z

    def restore(self, Z):
        """
        Restore a standardized output to the non-standardized output

        Parameters
        ----------
        Z : torch.Tensor
            Tensor of standardized outputs

        Returns
        -------
        X : torch.Tensor
            Tensor of non-standardized outputs

        """
        X = self.mean + Z * self.std
        return X

    def state(self):
        """
        Return dictionary of the state of the Standardizer

        Returns
        -------
        dict
            Dictionary with the mean and std of the outputs

        """
        return {"mean": self.mean, "std": self.std}

    def load(self, state):
        """
        Load a dictionary containing the state of the Standardizer and assign mean and std

        Parameters
        ----------
        state : dict
            Dictionary containing mean and std
        """
        self.mean = state["mean"]
        self.std = state["std"]


# Utility functions to train, test model
def train_model(
        epoch,
        model,
        training_dataloader,
        optimizer,
        loss_fn,
        standardizer,
        use_GPU,
        max_atoms,
        node_vec_len,
        save_dir,
        output_file_path=None,
        target_epoch=105
):
    """
    Execute training of one epoch for the ChemGCN model and optionally save predictions.
    """
    avg_loss = 0
    avg_mae = 0
    count = 0

    if epoch == target_epoch:
        smiles_list = []
        predictions_list = []
        true_values_list = []
    else:
        smiles_list = predictions_list = true_values_list = None

    model.train()

    for i, dataset in enumerate(training_dataloader):
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        # Store SMILES if this is the target epoch
        if epoch == target_epoch and len(dataset) > 2:
            smiles_list.extend([dataset[2][j] for j in range(len(dataset[2]))])

        first_dim = int(torch.numel(node_mat) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        output_std = standardizer.standardize(output)

        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
            nn_output = output_std.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output_std

        nn_prediction = model(*nn_input)
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss.detach().cpu().item()

        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        avg_mae += mae

        if epoch == target_epoch:
            predictions_list.extend(prediction.numpy().flatten().tolist())
            true_values_list.extend(output.numpy().flatten().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

    avg_loss /= count
    avg_mae /= count

    print(f"Epoch: [{epoch}]\tTraining Loss: [{avg_loss:.2f}]\tTraining MAE: [{avg_mae:.2f}]")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        loss_file = os.path.join(save_dir, "training_loss_0.1.csv")

        if not os.path.exists(loss_file):
            with open(loss_file, 'w') as f:
                f.write("epoch,loss,mae\n")

        with open(loss_file, 'a') as f:
            f.write(f"{epoch},{avg_loss:.4f},{avg_mae:.4f}\n")

    if epoch == target_epoch and output_file_path and smiles_list:
        results_df = pd.DataFrame({
            'smiles': smiles_list,
            'predict': predictions_list,
            'true': true_values_list,
            'epoch': [epoch] * len(smiles_list)
        })
        results_df.to_csv(output_file_path, index=False)

    return avg_loss, avg_mae


def test_model(
    model,
    test_dataloader,
    loss_fn,
    standardizer,
    use_GPU,
    max_atoms,
    node_vec_len,
    output_file_path=None,  # 添加输出文件路径参数
):
    """
    Test the ChemGCN model and optionally save predictions to a CSV file.

    Parameters
    ----------
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    loss_fn : like nn.MSELoss()
        Model loss function
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph
    output_file_path: str or None
        If provided, the predictions will be saved to a CSV file at this path.

    Returns
    -------
    test_loss : float
        Test loss
    test_mae : float
        Test MAE
    """
    test_loss = 0
    test_mae = 0
    count = 0

    # Initialize lists to store predictions and SMILES
    smiles_list = []
    predictions_list = []
    true_values_list = []

    model.eval()

    for i, dataset in enumerate(test_dataloader):
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)
        # Standardize output
        output_std = standardizer.standardize(output)

        smiles_list.extend([dataset[2][i] for i in range(len(dataset[2]))])

        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
            nn_output = output_std.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output_std

        nn_prediction = model(*nn_input)

        prediction = standardizer.restore(nn_prediction.detach().cpu()).numpy()
        true_values_list.extend(output.numpy())
        predictions_list.extend(prediction)

        loss = loss_fn(nn_output, nn_prediction)
        test_loss += loss

        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        test_mae += mae

        count += 1

    test_loss = test_loss.detach().cpu().numpy() / count
    test_mae = test_mae / count

    print(f"Test Loss: {test_loss:.2f}")
    print(f"Test MAE: {test_mae:.2f}")

    # If an output file path is provided, save predictions to a CSV file
    if output_file_path:
        results_df = pd.DataFrame({
            'smiles': smiles_list,
            'predict': predictions_list,
            'true': true_values_list
        })
        results_df.to_csv(output_file_path, index=False)

    return test_loss, test_mae


def parity_plot(
    save_dir,
    model,
    test_dataloader,
    standardizer,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Create a parity plot for the ChemGCN model.

    Parameters
    ----------
    save_dir: str
        Name of directory to store the parity plot in
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph
    """
    outputs = []
    predictions = []

    model.eval()

    for i, dataset in enumerate(test_dataloader):
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
        else:
            nn_input = (node_mat, adj_mat)

        nn_prediction = model(*nn_input)

        prediction = standardizer.restore(nn_prediction.detach().cpu())

        outputs.append(output)
        predictions.append(prediction)

    outputs_arr = np.concatenate(outputs)
    preds_arr = np.concatenate(predictions)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
    ax.scatter(outputs_arr, preds_arr, marker="o", s=30, c="r")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.plot([outputs_arr.min(), outputs_arr.max()], [outputs_arr.min(), outputs_arr.max()], color="black", lw=2)
    ax.set_title("Parity Plot")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plot_file_path = os.path.join(save_dir, "parity_plot_pkb.png")
    plt.savefig(plot_file_path)
    plt.close()


def loss_curve(save_dir, epochs, losses, maes=None):
    """
    Make a loss curve and save data to CSV.

    Parameters
    ----------
    save_dir: str
        Name of directory to store plot in
    epochs: list
        List of epochs
    losses: list
        List of losses
    maes: list or None
        List of MAEs (optional)
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存数据到CSV
    data = {'epoch': epochs, 'loss': losses}
    if maes is not None:
        data['mae'] = maes

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, "loss_data_pkb.csv"), index=False)

    # 绘制损失曲线
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=500)
    ax.plot(epochs, losses, marker="o", linestyle="--", color="royalblue", label='Loss')
    if maes is not None:
        ax.plot(epochs, maes, marker="s", linestyle="-.", color="green", label='MAE')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curve_pkb.png"))
    plt.close()
