import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump
import numpy as np
import shap
import optuna
from optuna.samplers import TPESampler
import random
from tqdm import tqdm
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2
# 设置全局随机种子，确保每次运行结果一致
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 读取计算出的描述符数据
df = pd.read_csv("D:\pKa-GCL\data\pka_30000\pka_30000.csv")

# 提取 SMILES 和 pKa 数据
smiles = df['SMILES']
y = df['pka']  # pKa 作为目标变量
X = df.drop(columns=['SMILES', 'pka'])

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集为训练、验证和测试集
X_train, X_temp, y_train, y_temp, smiles_train, smiles_temp = train_test_split(
    X_scaled, y, smiles, test_size=0.2, random_state=SEED
)
X_val, X_test, y_val, y_test, smiles_val, smiles_test = train_test_split(
    X_temp, y_temp, smiles_temp, test_size=0.5, random_state=SEED
)

# 定义贝叶斯优化的目标函数
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'random_state': SEED,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return val_rmse

# 使用指定随机种子的 TPE sampler 进行贝叶斯优化
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 输出最佳超参数
best_params = study.best_params
print("Best hyperparameters:", best_params)

# 使用最佳超参数训练最终模型
best_model = RandomForestRegressor(**best_params, random_state=SEED)
best_model.fit(X_train, y_train)

# 保存模型与Scaler
model_path = 'D:\pKa-GCL\data\pka_30000\RF\pka_RF_1_model.joblib'
scaler_path = 'D:\pKa-GCL\data\pka_30000\RF\scaler_RF_1.joblib'
dump(best_model, model_path)
dump(scaler, scaler_path)

# 测试集标准化
X_test_scaled = scaler.transform(X_test)

# SHAP 特征重要性分析
explainer = shap.Explainer(best_model, X_train, check_additivity=False, approximate=True)
shap_values = explainer(X_test_scaled)

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test_scaled, feature_names=X.columns)
plt.title("SHAP Summary Plot", fontsize=28, fontweight='bold')
plt.xlabel('SHAP Value', fontsize=28, fontweight='bold')
plt.ylabel('Feature', fontsize=28, fontweight='bold')
plt.savefig('D:\pKa-GCL\data\pka_30000\pka_RF_30000_shap_unique_summary.png', bbox_inches='tight', dpi=300)
plt.close()

# SHAP Feature Importance Bar Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_test_scaled, feature_names=X.columns, plot_type="bar")
plt.title("SHAP Feature Importance", fontsize=28, fontweight='bold')
plt.xlabel('Mean |SHAP Value|', fontsize=28, fontweight='bold')
plt.savefig('D:\pKa-GCL\data\pka_30000\pka_RF_30000_shap_unique_summary_bar.png', bbox_inches='tight', dpi=300)
plt.close()

# SHAP Dependence Plot (第一个特征)
plt.figure(figsize=(10, 8))
shap.dependence_plot(0, shap_values.values, X_test_scaled, feature_names=X.columns)
plt.title("SHAP Dependence Plot", fontsize=28, fontweight='bold')
plt.xlabel(X.columns[0], fontsize=28, fontweight='bold')
plt.ylabel('SHAP Value', fontsize=28, fontweight='bold')
plt.savefig('D:\pKa-GCL\data\pka_30000\pka_RF_30000_shap_unique_dependence_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# SHAP Waterfall Plot（展示第一条样本）
plt.figure(figsize=(10, 8))
shap.waterfall_plot(shap_values[0])
plt.title("SHAP Waterfall Plot", fontsize=28, fontweight='bold')
plt.xlabel('Feature Value', fontsize=28, fontweight='bold')
plt.ylabel('SHAP Value', fontsize=28, fontweight='bold')
plt.savefig('D:\pKa-GCL\data\pka_30000\pka_shap_RF_waterfall_uniqeue_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# 获取 SHAP 特征重要性排名前 20
importance = shap_values.abs.mean(0).values
feature_names = X.columns
sorted_idx = importance.argsort()[-20:]
top_features = feature_names[sorted_idx]
shap_df = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': importance
}).sort_values('mean_abs_shap', ascending=False)

shap_df.to_csv('D:\pKa-GCL\data\pka_30000\\top_1_shap_features.csv', index=False)
# 打印排名前20的特征
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {importance[idx]}")

# 展示前 20 特征的测试集
X_display_top = pd.DataFrame(X_test[:, sorted_idx], columns=top_features)

# 保存 SHAP 值和相关数据
shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
shap_df['TRUE'] = y_test.values
shap_df['predict'] = best_model.predict(X_test_scaled)
shap_df['SMILES'] = smiles_test.values
shap_df.to_csv('D:\pKa-GCL\data\pka_30000\pka_RF_1_shap_values.csv', index=False)
