import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import shap
from joblib import dump
import numpy as np
from sklearn.manifold import TSNE
import optuna
from sklearn.metrics import mean_squared_error

# Set global plot parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2

SEED = 42
np.random.seed(SEED)

# Load data
df = pd.read_csv(r"D:\pKa-GCL\data\pka_30000\pka_30000.csv")
smiles = df['SMILES']
y = df['pka']
X = df.drop(columns=['SMILES', 'pka'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp, smiles_train, smiles_temp = train_test_split(
    X_scaled, y, smiles, test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test, smiles_val, smiles_test = train_test_split(
    X_temp, y_temp, smiles_temp, test_size=0.5, random_state=SEED)


# Hyperparameter optimization with Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    }

    model = DecisionTreeRegressor(random_state=SEED, **params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, val_pred))


study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50)

# Train best model
best_params = study.best_params
print("Best hyperparameters:", best_params)

best_model = DecisionTreeRegressor(random_state=SEED, **best_params)
with tqdm(total=1, desc="Training Best Model") as pbar:
    best_model.fit(X_train, y_train)
    pbar.update(1)

# Save model and scaler
dump(best_model, r'D:\pKa-GCL\data\pka_30000\DT\pka_DT_1_model.joblib')
dump(scaler, r'D:\pKa-GCL\data\pka_30000\des\DT\scaler_1.joblib')

# SHAP analysis
X_test_scaled = scaler.transform(X_test)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

# 1. SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title("SHAP Summary Plot", fontsize=28, fontweight='bold')
plt.xlabel('SHAP Value', fontsize=28, fontweight='bold')
plt.ylabel('Feature', fontsize=28, fontweight='bold')
plt.savefig(r'D:\pKa-GCL\data\pka_30000\pka_DT_30000_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. SHAP Feature Importance (Bar Plot)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar")
plt.title("SHAP Feature Importance", fontsize=28, fontweight='bold')
plt.xlabel('Mean |SHAP Value|', fontsize=28, fontweight='bold')
plt.savefig(r'D:\pKa-GCL\data\pka_30000\pka_DT_30000_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. SHAP Dependence Plot
plt.figure(figsize=(10, 8))
shap.dependence_plot(0, shap_values, X_test_scaled, feature_names=X.columns)
plt.title("SHAP Dependence Plot", fontsize=28, fontweight='bold')
plt.xlabel(X.columns[0], fontsize=28, fontweight='bold')
plt.ylabel('SHAP Value', fontsize=28, fontweight='bold')
plt.savefig(r'D:\pKa-GCL\data\pka_30000\pka_DT_30000_shap_dependence_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. SHAP Force Plot (HTML only - can't be saved as PNG directly)
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test_scaled[0],
    feature_names=X.columns,
    matplotlib=False
)
shap.save_html(r'D:\pKa-GCL\data\pka_30000\pka_DT_30000_shap_force_plot.html', force_plot)

# Feature importance analysis
importance = np.abs(shap_values).mean(0)
sorted_idx = importance.argsort()[-20:]  # Top 20 features
for idx in sorted_idx:
    print(f"{X.columns[idx]}: {importance[idx]}")
shap_df = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': importance
}).sort_values('mean_abs_shap', ascending=False)

shap_df.to_csv('D:\pKa-GCL\data\pka_30000\\top_20_shap_features.csv', index=False)
# Save SHAP values
shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df['TRUE'] = y_test.values
shap_df['predict'] = best_model.predict(X_test_scaled)
shap_df['SMILES'] = smiles_test.values
shap_df.to_csv(r'D:\pKa-GCL\data\pka_30000\pka_DT_1_shap_values.csv', index=False)

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=SEED)
shap_values_tsne = tsne.fit_transform(shap_values)

plt.figure(figsize=(10, 8))
sc = plt.scatter(
    shap_values_tsne[:, 0],
    shap_values_tsne[:, 1],
    c=y_test,
    cmap='GnBu',
    alpha=1,
    s=5,
    vmin=0  # Set colorbar minimum to 0
)

cbar = plt.colorbar(sc)
cbar.set_label('True pKa', fontsize=20, fontweight='bold')
plt.xlabel('t-SNE 1', fontsize=20, fontweight='bold')
plt.ylabel('t-SNE 2', fontsize=20, fontweight='bold')
plt.savefig(r'D:\pKa-GCL\data\pka_30000\pka_DT_1_shap_tsne_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved in PNG format with 300 DPI")