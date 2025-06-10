import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem


class PkaDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(PkaDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        valid_indices = []  # To keep track of valid indices

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                print("Invalid SMILES:", row["smiles"])
                continue

            try:
                f = featurizer._featurize(mol)
            except Exception as e:
                print("Problematic SMILES:", row["smiles"], "Error:", e)
                continue

            # Get node features, edge features and edge indices
            node_feats = torch.tensor(f.node_features, dtype=torch.float)
            edge_feats = torch.tensor(f.edge_features, dtype=torch.float)
            edge_index = torch.tensor(f.edge_index, dtype=torch.long)

            # Get pKa label
            label = self._get_labels(row["pka"])

            # Build graph data
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label,
                smiles=row["smiles"]
            )

            # Save graph data
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{len(valid_indices)}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{len(valid_indices)}.pt'))

            valid_indices.append(index)  # Track valid indices

        # Save the valid indices for later use
        torch.save(valid_indices, os.path.join(self.processed_dir, 'valid_indices.pt'))

    def _get_labels(self, label):
        """Convert label to tensor"""
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)

    def len(self):
        """Return dataset size"""
        valid_indices = torch.load(os.path.join(self.processed_dir, 'valid_indices.pt'))
        return len(valid_indices)

    def get(self, idx):
        """Load single graph data"""
        valid_indices = torch.load(os.path.join(self.processed_dir, 'valid_indices.pt'))
        if idx >= len(valid_indices):
            raise IndexError(f"Index {idx} is out of bounds for the dataset with {len(valid_indices)} valid samples.")

        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


# Main function
if __name__ == "__main__":
    # Dataset paths
    root = "/home/lxy/ToxPred_nitrification-main/data/"  # Dataset root directory
    filename = "pka_30000.csv"  # CSV filename

    # Create dataset
    dataset = PkaDataset(root=root, filename=filename)

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print("First sample:")
    sample = dataset.get(0)
    print(f"Node features: {sample.x.shape}")
    print(f"Edge indices: {sample.edge_index.shape}")
    print(f"Edge features: {sample.edge_attr.shape}")
    print(f"Label (pKa): {sample.y}")
    print(f"SMILES: {sample.smiles}")