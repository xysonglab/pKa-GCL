# PKa-GCL

This is the code for "Dual-Modality Graph Contrastive Learning (GCL) for Accurate and Interpretable Molecular Property Prediction: Unveiling pKa Insights for Drug Discovery" paper.

## Directory Structure

```shell
pKa-GCL/
├── ATT/            
├── ChemGCN-main/  
├── data/  
├── ML/         
├──output/        
├──results/
├──runs/
├──convert_to_sdf.py
├──dataloader.py
├──dataset.py
├──environment.yml
├──layers.py
├──layers_mean.py
├──model.py
├──model_mean.py
├──predict.py
├──preprocess.py
├──README.md
├──train_finetune.py
├──train_pkagcl.py
├──traindatasize.py
├──trainfinesize.py
└──utils.py  
```

## Installation

### Environment Setup

    # Create and activate conda environment
    conda create --name pkagcl python=3.7
    conda activate pkagcl
    
    # Installing PaddlePaddle (based on your CUDA version)
    python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    conda install pgl
    
    # Install additional dependencies
    conda install -c rdkit rdkit
    pip install -U scikit-learn

### Optional Set-up

    # Set up the environment using the provided file
    conda env create --name pkagcl --file environment.yml
    conda activate pkagcl

## Data Preparation

This dataset, obtained from the publication at [https://doi.org/10.1021/acs.jctc.4c00328], serves as the benchmark data for training and evaluating models in acid dissociation constant (pKa) prediction.

### 1. 3D Structure Generation

    # Convert CSV to 3D SDF using RDKit
    python convert_to_sdf.py --dataset [Dataset_name] --algo [Optimization_algorithm] --process [Number_of_processes]

### 2. Data Preprocessing

    # Molecular Feature Extraction
    python convert_to_sdf.py --dataset [Dataset_name] --algo [Optimization_algorithm] --process [Number_of_processes]

## Model Training Process

### 1. Pretraining Stage

```
# This code represents the model training and validation step, implementing a contrastive learning framework (pKaGCL) for molecular property prediction using both 2D and 3D molecular representations.

python train_pkagcl.py --cuda [0] --dataset [Dataset_name] --num_dist 4 --cut_dist 5 --model_dir [Model_path_to_save]
```

### 2. Fine-tuning Stage

```
# The model implements k-fold cross-validation, integrates both 2D and 3D molecular features, supports pKa regression tasks, includes a complete training, validation, and testing pipeline, and saves model checkpoints and evaluation metrics.

python train_finetune.py --cuda [0] --dataset [Dataset_name]  --model_dir [Model_path_to_load] --task_dim [1] --num_dist 2 --num_angle 4 --cut_dist 4 --output_dir [Outpath_path_to_save] --results_dir [Resultspath_path_to_save]
```

### 3. Model Evaluation

```
# Load the trained model, perform prediction and evaluation on the test set, and output the regression performance score and prediction results file.

python predict.py --cuda [0] --dataset [Dataset_name]  --model_dir [Model_path_to_load] --task_dim [1] --num_dist 2 --num_angle 4 --cut_dist 4 --output_dir [Outpath_path_to_save] --results_dir [Resultspath_path_to_save]
```

## Visualization

To explore model interpretability through attention visualization or Shapley value analysis, see the provided examples in DT.py, RF.py, XGBoost.py, examples.ipynb, viz_attention.py, and shapley.py.
