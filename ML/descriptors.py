import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import seaborn as sns
from tqdm import tqdm  # Import tqdm

# Read data
dataset = pd.read_csv("D:\geomgcl\data\pka_datasize\\1\pka_1.csv")

# Generate canonical SMILES
def canonical_smiles(smiles_list):
    canonical_smiles_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical_smiles_list.append(Chem.MolToSmiles(mol))
        else:
            print(f"Warning: Invalid SMILES encountered: {smi}")
            canonical_smiles_list.append(None)  # Or skip this entry
    return canonical_smiles_list

# Generate Canonical SMILES
dataset['SMILES'] = canonical_smiles(dataset['smiles'])
dataset_new = dataset.drop_duplicates(subset=['SMILES'])

# Visualize pKa distribution
sns.displot(dataset_new['pka'], kde=True)
sns.set(style="whitegrid")

# Calculate RDKit descriptors
def RDkit_descriptors(smiles_list):
    # Create molecule objects, skip invalid SMILES
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi is not None]

    # Add hydrogens
    mols = [Chem.AddHs(mol) for mol in mols]

    # Use RDKit descriptor calculator
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    # Use tqdm to show progress bar
    descriptors = []
    for mol in tqdm(mols, desc="Calculating Descriptors", unit="mol"):
        descriptors.append(calc.CalcDescriptors(mol))

    return descriptors, desc_names

# Calculate molecular descriptors
Mol_descriptors, desc_names = RDkit_descriptors(dataset_new['SMILES'])

# Create dataframe with descriptors
df_with_descriptors = pd.DataFrame(Mol_descriptors, columns=desc_names)

# Add SMILES and corresponding pKa values to the descriptor dataframe
df_with_descriptors['SMILES'] = dataset_new['SMILES'].reset_index(drop=True)
df_with_descriptors['pka'] = dataset_new['pka'].reset_index(drop=True)

# Export to CSV file
df_with_descriptors.to_csv("D:\geomgcl\data\pka_datasize\des\pka_1_des.csv", index=False)

# Display first few rows of dataframe
print(df_with_descriptors.head())