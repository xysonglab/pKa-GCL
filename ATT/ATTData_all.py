import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors


class PkaHybridDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        Hybrid molecular dataset combining 2D graph features and 3D geometric features
        Args:
            root: Directory where dataset should be saved
            filename: CSV file containing SMILES and pKa values
            test: Whether this is a test set
        """
        self.test = test
        self.filename = filename
        super(PkaHybridDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)  # Initialize DeepChem featurizer
        valid_indices = []

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # ========== Basic molecular processing ==========
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue

            # Use hydrogenated molecule consistently
            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()

            # ========== Extract 2D features using DeepChem ==========
            try:
                f = featurizer._featurize(mol)  # Use previously saved 2D feature extraction
                node_attrs_2d = torch.tensor(f.node_features, dtype=torch.float)
                edge_attrs_2d = torch.tensor(f.edge_features, dtype=torch.float)
                edge_index = torch.tensor(f.edge_index, dtype=torch.long)
            except Exception as e:
                print(f"DeepChem 2D feature extraction failed: {e}")
                continue

            # ========== 3D feature generation ==========
            try:
                # Conformer generation
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)

                # Get 3D coordinates
                conf = mol.GetConformer()
                coords = torch.tensor([conf.GetAtomPosition(i) for i in range(n_atoms)],
                                      dtype=torch.float)

                # Calculate 3D edge features (based on same edge index)
                dist_matrix = torch.cdist(coords, coords)
                edge_attrs_3d = dist_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            except Exception as e:
                print(f"3D feature extraction failed: {e}")
                continue

            # ========== Feature fusion ==========
            # Node features: [n_atoms, 2D features + 3D coordinates]
            node_feats = torch.cat([
                node_attrs_2d,  # 2D features from DeepChem
                coords  # 3D coordinates
            ], dim=-1)

            # Edge features: [n_edges, 2D edge features + 3D distances]
            edge_feats = torch.cat([
                edge_attrs_2d,  # 2D edge features from DeepChem
                edge_attrs_3d  # 3D distance features
            ], dim=-1)

            # ========== Construct graph data ==========
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=torch.tensor([row["pka"]], dtype=torch.float),
                smiles=row["smiles"],
                pos=coords  # Save 3D coordinates
            )

            # Save data file
            file_name = f'data_test_{index}.pt' if self.test else f'data_{index}.pt'
            torch.save(data, os.path.join(self.processed_dir, file_name))
            valid_indices.append(index)

        # Save valid indices
        torch.save(valid_indices, os.path.join(self.processed_dir, 'valid_indices.pt'))

    def _get_rdkit_2d_descriptors(self, mol):
        """Calculate RDKit 2D molecular descriptors"""
        desc_list = [
            rdMolDescriptors.CalcExactMolWt(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Chem.Crippen.MolLogP(mol),
            Chem.Crippen.MolMR(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol)
        ]
        return np.array(desc_list)

    def len(self):
        return len(torch.load(os.path.join(self.processed_dir, 'valid_indices.pt')))

    def get(self, idx):
        valid_indices = torch.load(os.path.join(self.processed_dir, 'valid_indices.pt'))
        if idx >= len(valid_indices):
            raise IndexError(f"Invalid index {idx} (max: {len(valid_indices) - 1})")

        file_name = f'data_test_{valid_indices[idx]}.pt' if self.test else f'data_{valid_indices[idx]}.pt'
        try:
            return torch.load(os.path.join(self.processed_dir, file_name))
        except FileNotFoundError:
            available_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
            raise FileNotFoundError(
                f"File {file_name} not found. Available files: {available_files[:5]}...")


if __name__ == "__main__":
    # Example usage
    dataset = PkaHybridDataset(root="/home/lxy/ToxPred_nitrification-main/data/23d", filename="pka_short_30000.csv")
    sample = dataset[0]

    print(f"Combined node features shape: {sample.x.shape}")
    print(f"Edge features shape: {sample.edge_attr.shape}")
    print(f"3D positions shape: {sample.pos.shape}")
    print(f"pKa value: {sample.y.item()}")