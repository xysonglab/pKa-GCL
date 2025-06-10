import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import optuna
import numpy as np
from rdkit import Chem
#46
# 读取计算出的描述符数据
df = pd.read_csv("D:\pKa-GCL\data\pka_30000\pka_30000.csv")

# 提取 SMILES 和 pKa 数据
smiles = df['SMILES']
y = df['pka']  # pKa 作为目标变量

# 去除 SMILES 列，得到特征矩阵
X = df.drop(columns=['SMILES', 'pka'])

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 修改数据拆分部分（替换原来的train_test_split代码）
# 将数据集拆分为训练集（前80%）、验证集（中间10%）和测试集（最后10%）
total_samples = len(X_scaled)
train_end = int(0.8 * total_samples)
val_end = int(0.9 * total_samples)

# 训练集（前80%）
X_train = X_scaled[:train_end]
y_train = y[:train_end]
smiles_train = smiles[:train_end]

# 验证集（中间10%）
X_val = X_scaled[train_end:val_end]
y_val = y[train_end:val_end]
smiles_val = smiles[train_end:val_end]

# 测试集（最后10%）
X_test = X_scaled[val_end:]
y_test = y[val_end:]
smiles_test = smiles[val_end:]

print(f"数据集划分：训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")


# 定义贝叶斯优化的目标函数
def objective(trial):
    # 定义超参数搜索空间
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
    }

    # 初始化模型
    model = xgb.XGBRegressor(**params)

    # 训练模型
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

    # 在验证集上评估模型
    val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)  # 计算 RMSE

    return val_rmse


# 使用 Optuna 进行贝叶斯优化
sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 输出最佳超参数
best_params = study.best_params
print("Best hyperparameters:", best_params)

# 使用最佳超参数训练最终模型
best_model = xgb.XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

# 保存模型和标准化器
model_path = 'D:\pKa-GCL\data\pka_30000\pka_30000.csv\XGB\\xgboost_pka_model_1.joblib'
scaler_path = 'D:\pKa-GCL\data\pka_30000\pka_30000.csv\XGB\scaler_1.joblib'
dump(best_model, model_path)
dump(scaler, scaler_path)

# 使用 SHAP 进行特征重要性分析
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
# 计算特征重要性
importance = np.abs(shap_values.values).mean(0)
sorted_idx = np.argsort(importance)[-20:]  # 取前20个最重要的特征
top_features = X.columns[sorted_idx]
top_importance = importance[sorted_idx]

# 输出前20个重要特征及其SHAP值
print("\nTop 20重要特征及其平均SHAP值:")
for feature, imp in zip(top_features[::-1], top_importance[::-1]):  # 从最重要到最不重要排序
    print(f"{feature}: {imp:.4f}")

# 保存SHAP值到CSV
shap_df = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': importance
}).sort_values('mean_abs_shap', ascending=False)

shap_df.to_csv('D:\pKa-GCL\data\pka_30000\pka_30000.csv\\top_20_shap_features.csv', index=False)
# 设置全局字体和样式
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'font.weight': 'bold',
    'text.color': 'black',          # 全局文本颜色
    'axes.labelcolor': 'black',     # 坐标轴标签颜色
    'xtick.color': 'black',         # x轴刻度颜色
    'ytick.color': 'black',         # y轴刻度颜色
    'axes.edgecolor': 'black'       # 坐标轴颜色
})

# 创建图形和轴
plt.figure(figsize=(20, 10))  # 加宽图形

# 计算特征重要性
importance = np.abs(shap_values.values).mean(0)
sorted_idx = np.argsort(importance)[-15:]
top_features = X.columns[sorted_idx]
top_importance = importance[sorted_idx]

# 创建主坐标轴网格
gs = plt.GridSpec(1, 2, width_ratios=[3, 1.5], wspace=0.15)

# 左侧蜂巢图
ax1 = plt.subplot(gs[0])
shap.summary_plot(
    shap_values.values,
    X_test,
    feature_names=X.columns,
    plot_type="dot",
    show=False,
    color_bar=False,
    alpha=1,
    max_display=15,
    cmap='RdBu'
)

for line in ax1.lines:
    if line.get_xdata()[0] == 0 and line.get_xdata()[1] == 0:
        line.set_visible(False)

# 边框和坐标轴设置
for spine in ['top', 'bottom', 'left', 'right']:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_edgecolor('black')
    ax1.spines[spine].set_linewidth(1.5)

ax1.tick_params(axis='x', which='both', labelsize=20, bottom=True, length=5, width=1.5)
ax1.tick_params(axis='y', which='both', labelsize=20, length=5, width=1.5)
ax1.set_xlabel("")  # 去除x轴标题

# 调整散点大小
for col in ax1.collections:
    if hasattr(col, '_sizes'):
        col.set_sizes([30])

# 蜂巢图设置虚线
y_min, y_max = ax1.get_ylim()
#ax1.grid(axis='y', linestyle='-', linewidth=1.5, alpha=0.5, color='black')

# 设置蜂巢图纵轴刻度字体大小和颜色，并缩小与坐标轴的间距
for label in ax1.get_yticklabels():
    label.set_fontsize(19)
    label.set_color('black')
    label.set_horizontalalignment('right')
    label.set_x(0.01)

# 添加中间虚线（去掉实线的手段是确保只添加我们手动设定的虚线）
#ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

# 添加 colorbar
norm = plt.Normalize(shap_values.values.min(), shap_values.values.max())
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
sm.set_array([])
cax = ax1.inset_axes([1.8, 0.1, 0.05, 1], transform=ax1.transAxes)
cbar = plt.colorbar(sm, cax=cax)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')

cbar.ax.tick_params(labelsize=16)

# 右侧条形图
ax2 = plt.subplot(gs[1])

# 移除不必要边框（包括左边框）
for spine in ['top', 'left', 'right']:
    ax2.spines[spine].set_visible(False)

# 保留下边框但去除 x=0 竖线（y轴主线）
ax2.spines['bottom'].set_visible(True)
ax2.spines['bottom'].set_edgecolor('black')
ax2.spines['bottom'].set_linewidth(1.5)

# 彻底隐藏 y 轴（避免显示 x=0 的竖线）
ax2.yaxis.set_visible(False)

x_max = max(top_importance) * 1.1

bars = ax2.barh(
    np.arange(len(top_features)),
    top_importance,
    height=0.6,
    color='#1f77b4',
    alpha=0.8,
    edgecolor='white',
    linewidth=0.5,
    align='center'
)

ax2.tick_params(axis='x',
                which='both',
                bottom=True,
                length=5,
                width=1.5,
                labelsize=20)

ax2.set_ylim(y_min, y_max)
ax2.set_xlim(0, x_max)

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(min(width + 0.02, x_max * 0.95),
             bar.get_y() + bar.get_height() / 2,
             f'{width:.2f}',
             va='center',
             ha='left',
             fontsize=10,
             bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=0.5))

# 条形图仅显示x轴的网格线
#ax2.grid(axis='x', linestyle='--', linewidth=1.5, alpha=0.6, color='black')

# 保存图形
plt.savefig(
    'D:\pKa-GCL\data\pka_30000\pka_30000.csv\\pka_shap_combined_plot.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()


# After this line:
# After calculating SHAP values
shap_values = explainer(X_test)

# Convert SHAP values to DataFrame and include SMILES
shap_df = pd.DataFrame(shap_values.values, columns=[f"SHAP_{col}" for col in X.columns])

# Combine all data - 确保SMILES列正确添加
shap_df = pd.concat([
    pd.DataFrame({'SMILES': smiles_test.reset_index(drop=True)}),  # 已包含SMILES
    pd.DataFrame(X_test, columns=X.columns),                      # 原始特征值
    shap_df,                                                     # SHAP值
    pd.DataFrame({'TRUE': y_test.reset_index(drop=True)}),    # 真实pKa值
    pd.DataFrame({'predict': best_model.predict(X_test)})        # 预测pKa值
], axis=1)

# 保存时确保路径存在
import os
os.makedirs('D:\pKa-GCL\data\pka_30000\pka_30000.csv\XGB/', exist_ok=True)
shap_csv_path = 'D:\pKa-GCL\data\pka_30000\pka_30000.csv/XGB/shap_1_values.csv'
shap_df.to_csv(shap_csv_path, index=False)
print(f"SHAP values with SMILES saved to: {shap_csv_path}")
# ========== 新增: 根据指定SMILES生成SHAP力图 ==========
def generate_shap_force_plot_for_smiles(target_smiles, smiles_list, X_data, shap_values, feature_names, output_dir):
    """为指定SMILES生成SHAP力图（显示红色前1和蓝色前3重要特征）"""
    try:
        # 查找SMILES对应的索引
        sample_idx = np.where(smiles_list == target_smiles)[0][0]
        print(f"找到SMILES: {target_smiles}，索引: {sample_idx}")

        # 获取对应的特征和SHAP值
        sample_features = X_data[sample_idx]
        sample_shap = shap_values[sample_idx]

        # 分离正向（红色）和负向（蓝色）影响特征
        positive_mask = sample_shap.values > 0
        negative_mask = sample_shap.values < 0

        # 获取红色（正向）前1特征
        pos_indices = np.argsort(-sample_shap.values[positive_mask])[:4]
        pos_features = sample_features[positive_mask][pos_indices]
        pos_shap_values = sample_shap.values[positive_mask][pos_indices]
        pos_feature_names = [feature_names[i] for i in np.where(positive_mask)[0][pos_indices]]

        # 获取蓝色（负向）前3特征
        neg_indices = np.argsort(sample_shap.values[negative_mask])[:4]  # 取最小的3个负值
        neg_features = sample_features[negative_mask][neg_indices]
        neg_shap_values = sample_shap.values[negative_mask][neg_indices]
        neg_feature_names = [feature_names[i] for i in np.where(negative_mask)[0][neg_indices]]

        # 合并特征
        top_features = np.concatenate([pos_features, neg_features])
        top_shap_values = np.concatenate([pos_shap_values, neg_shap_values])
        top_feature_names = pos_feature_names + neg_feature_names

        # 创建分子名称
        mol = Chem.MolFromSmiles(target_smiles)
        compound_name = mol.GetProp("_Name") if mol and mol.HasProp("_Name") else target_smiles
        safe_name = "".join(c for c in compound_name if c.isalnum() or c in " _-")
        filename = f"shap_force_plot_{safe_name}.html"
        save_path = f"{output_dir}/{filename}"

        # 创建并保存力图
        force_plot = shap.force_plot(
            explainer.expected_value,
            top_shap_values,
            top_features,
            feature_names=top_feature_names,
            matplotlib=False,
            contribution_threshold=0.05  # 过滤微小贡献
        )

        # 保存HTML文件
        shap.save_html(save_path, force_plot)
        print(f"SHAP力图已保存至: {save_path}")

    except Exception as e:
        print(f"无法为SMILES {target_smiles}生成SHAP力图: {str(e)}")



if len(smiles_test) > 0:
    generate_shap_force_plot_for_smiles(
        target_smiles='Oc1ccccc1/C=N/CC(O)c1ccccc1',
        smiles_list=smiles_test,
        X_data=X_test,
        shap_values=shap_values,
        feature_names=X.columns,
        output_dir='D:/Code/geomgcl/summary'
    )

# 示例: 为特定SMILES生成力图 (取消注释并替换为你的SMILES)
# generate_shap_force_plot_for_smiles(
#     target_smiles="CCO",  # 替换为你感兴趣的SMILES
#     smiles_list=smiles_test,  # 可以改为smiles_train或smiles_val
#     X_data=X_test,
#     shap_values=shap_values,
#     feature_names=X.columns,
#     output_dir='D:/Code/geomgcl/summary'
# )

# ========== 原有代码保持不变 ==========
# 3. SHAP Dependence Plot (选择一个特征，例如第一个特征)
force_plot = shap.force_plot(
    shap_values[0].base_values,
    shap_values[0].values,
    X_scaled[0],
    feature_names=X.columns,
    matplotlib=False
)
shap.save_html('D:\pKa-GCL\data\pka_30000\pka_30000.csv/shap_force_plot.html', force_plot)

# 4. SHAP Force Plot (对一个样本的解释)
shap.initjs()
force_plot = shap.force_plot(
    shap_values[0].base_values,
    shap_values[0].values,
    X_test[0],
    feature_names=X.columns,
    matplotlib=False
)
shap.save_html('D:\pKa-GCL\data\pka_30000\pka_30000.csv/pka_shap_XGB_1_force_plot.html', force_plot)