import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem


class PkaDataset3D(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(PkaDataset3D, self).__init__(root, transform, pre_transform)

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

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            try:
                # Attempt to generate a 3D conformer
                embed_result = AllChem.EmbedMolecule(mol)
                if embed_result == -1:
                    print(f"Failed to generate 3D conformer for SMILES: {row['smiles']}")
                    continue

                # Try MMFF optimization first
                try:
                    optimize_result = AllChem.MMFFOptimizeMolecule(mol)
                    if optimize_result != 0:
                        print(f"MMFF optimization failed for SMILES: {row['smiles']}, trying UFF...")
                        # If MMFF fails, try UFF
                        optimize_result = AllChem.UFFOptimizeMolecule(mol)
                        if optimize_result != 0:
                            print(f"UFF optimization also failed for SMILES: {row['smiles']}")
                            continue
                except Exception as e:
                    print(f"Error optimizing SMILES: {row['smiles']}, Error: {e}")
                    continue
            except Exception as e:
                print(f"Error processing SMILES: {row['smiles']}, Error: {e}")
                continue

            # Get 3D coordinates
            conf = mol.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

            try:
                f = featurizer._featurize(mol)
            except Exception as e:
                print("Problematic SMILES:", row["smiles"], "Error:", e)
                continue

            # Get node features, edge features, and edge index
            node_feats = torch.tensor(f.node_features, dtype=torch.float)
            edge_feats = torch.tensor(f.edge_features, dtype=torch.float)
            edge_index = torch.tensor(f.edge_index, dtype=torch.long)

            # Ensure edge_index has the correct shape [2, num_edges]
            if edge_index.size(0) != 2:
                edge_index = edge_index.t()  # Transpose to [2, num_edges]

            # Add 3D coordinates to node features
            node_feats = torch.cat([node_feats, torch.tensor(coords, dtype=torch.float)], dim=-1)

            # Calculate pairwise distances as edge features
            dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            distances = dist_matrix[edge_index[0], edge_index[1]]  # Extract distances for edges
            distances = torch.tensor(distances, dtype=torch.float).unsqueeze(-1)  # Convert to tensor and add dimension

            # Concatenate distances to edge features
            edge_feats = torch.cat([edge_feats, distances], dim=-1)

            # Ensure edge_attr has the correct shape [num_edges, edge_dim]
            edge_attr = edge_feats  # Shape of edge_attr is [num_edges, edge_dim]

            # Get pKa label
            label = self._get_labels(row["pka"])

            # Build graph data
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
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
        """Load a single graph data"""
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
    root = "/home/lxy/ToxPred_nitrification-main/data"  # Dataset root directory
    filename = "pka_30000.csv"  # CSV filename

    # Create dataset
    dataset = PkaDataset3D(root=root, filename=filename)

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print("First sample:")
    sample = dataset.get(0)
    print(f"Node features: {sample.x.shape}")
    print(f"Edge indices: {sample.edge_index.shape}")
    print(f"Edge features: {sample.edge_attr.shape}")
    print(f"Label (pKa): {sample.y}")
    print(f"SMILES: {sample.smiles}")