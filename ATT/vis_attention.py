from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import torch
import os


def visualize_attention(smiles, attention_weight, output_path=None, colorbar_path=None):
    """
    Visualize attention weights on molecular structure with enhanced styling.
    Optionally saves colorbar as separate image with thick border.

    Args:
        smiles (str): Input SMILES string
        attention_weight (torch.Tensor): Attention weights tensor
        output_path (str, optional): Path to save the main image. If None, displays interactively.
        colorbar_path (str, optional): Path to save the colorbar image. If None, colorbar is not saved separately.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Convert attention weights
    num_atoms = len(mol.GetAtoms())
    atomSum_weights = attention_weight.sum(axis=0).cpu().numpy()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [[0, 0.4, 0.8], [1, 1, 1], [0.97, 0.46, 0.43]])

    Amean_weight = atomSum_weights / num_atoms
    nanMean = np.nanmean(Amean_weight)
    weights_to_plot = Amean_weight - nanMean

    # Set up figure with enhanced styling
    plt.rcParams.update({
        'font.size': 24,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # ==================== Main Figure ====================
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # Draw molecule with attention weights
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        weights_to_plot,
        alpha=0.1,
        size=(200, 200),
        colorMap=cmap,
        contourLines=10,
        linewidth=100
    )

    # Adjust main plot position
    ax = plt.gca()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.85, pos.height])

    # Make atoms and bonds more prominent
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabelBold", "true")

    # Save or display main figure
    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Main visualization saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)

    # ==================== Colorbar Figure ====================
    if colorbar_path:
        # Create figure just for colorbar with thick border
        cb_fig = plt.figure(figsize=(1.2, 6), dpi=300)  # Slightly wider for border
        cb_ax = cb_fig.add_axes([0.15, 0.1, 0.25, 0.65])  # Adjusted position

        # Create normalized colorbar
        norm = plt.Normalize(vmin=weights_to_plot.min(), vmax=weights_to_plot.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add colorbar with enhanced styling
        cbar = plt.colorbar(sm, cax=cb_ax)
        cbar.set_label('Relative Attention',
                       rotation=270,
                       labelpad=30,
                       weight='bold',
                       fontsize=28)

        # ====== Border Customization ======
        # Set thick border around colorbar
        for spine in cb_ax.spines.values():
            spine.set_linewidth(3)  # 3-point thick border
            spine.set_color('black')  # Black border color

        # Make colorbar ticks and labels bold
        cb_ax.tick_params(width=2, length=6, labelsize=24)

        # Save colorbar with tight layout
        cb_fig.savefig(colorbar_path, format='svg', bbox_inches='tight', dpi=300,
                       facecolor='white', edgecolor='none')  # Transparent background
        print(f"Colorbar with thick border saved to {colorbar_path}")
        plt.close(cb_fig)


def convert_att(att, n_atom):
    """
    Convert the attention to a matrix, while each row representing an atom, and each column representing a
    contribution. For example, the location [0,1] represents the contribution of atom 1 to atom 0
    :param att: attention weights from gat_conv_layer
    :param n_atom: number of atoms in the molecule
    :return: contribution matrix
    """

    att_avg = torch.mean(att[1], axis=1)

    converted_att = torch.zeros((n_atom, n_atom))

    for i in range(len(att[0][0])):
        converted_att[att[0][1][i]][att[0][0][i]] += att_avg[i]

    return converted_att


def calculate_gat_att(x, edge_index, edge_attr, gat_conv_layer, transform_layer, batch_norm_layer, n_atom):
    """
    calculate the attention matrix for a GATConv layer, and keep track of the input (hidden) to the next
    convolution layer
    :param x: input of the convolution layer
    :param edge_index
    :param edge_attr
    :param gat_conv_layer
    :param transform_layer
    :param batch_norm_layer
    :param n_atom
    :return: hidden to the next convolution layer, contribution matrix
    """

    forward_res = gat_conv_layer.forward(x, edge_index, edge_attr, return_attention_weights=True)
    att = convert_att(forward_res[1], n_atom)
    hidden = forward_res[0]
    hidden = F.tanh(hidden)
    hidden = transform_layer(hidden)
    hidden = batch_norm_layer(hidden)

    return hidden, att


def extract_layers(model):
    """
    Extract GATConv layers, head transform layers (transform dimensions back to the embedding size), and batch
    normalization layers.
    :param model
    :return: convolution layers, transform layers, and normalization layers
    """

    conv_layers = [model.initial_conv, model.conv1, model.conv2, model.conv3]
    transform_layers = [model.head_transform1, model.head_transform2, model.head_transform3, model.head_transform4]
    norm_layers = [model.bn1, model.bn2, model.bn3, model.bn4]

    return conv_layers, transform_layers, norm_layers


def extract_multi_head_attention(x, edge_index, edge_attr, batch, model):
    # Calculate the multi-head attention
    # Only accept one sample at a time

    layer_att = model(x, edge_index, edge_attr, batch)[2][0].detach()
    return layer_att


def calculate_overall_att(x, edge_index, edge_attr, batch, model, n_atom):
    # Calculate the corrected attentions of inputs with a model.
    # Only accept one sample at a time

    conv_layers, transform_layers, norm_layers = extract_layers(model)

    assert len(conv_layers) == len(transform_layers) and len(transform_layers) == len(norm_layers)

    n_layers = len(conv_layers)
    atts = []
    hidden = x

    for i in range(n_layers):
        hidden, att = calculate_gat_att(hidden, edge_index, edge_attr, conv_layers[i], transform_layers[i],
                                        norm_layers[i], n_atom)
        atts.append(att)

    # The attention (distribution) of atoms before multi-head attention layer.
    # In the method section - Attention Correction of paper: C1 x C2 x C3 x C4
    pre_att = atts[0]
    for i in range(1, n_layers):
        pre_att = torch.matmul(atts[i], pre_att)

    # The attention from multi-head attention layer.
    # The next two lines should be equivalent.

    # layer_att = extract_multi_head_attention(x, edge_index, edge_attr, batch, model)
    layer_att = model(x, edge_index, edge_attr, batch)[2][0]

    # The attention after multiplying multi-head attention with embeddings (hidden)
    post_att = torch.round(torch.matmul(layer_att, pre_att))

    # The attention is added back to embeddings, so the overall attention is
    overall_att = pre_att + post_att
    overall_att = overall_att.detach()

    # Normalize for each atom
    overall_att = overall_att / overall_att.sum(dim=-1).unsqueeze(-1)

    return overall_att
