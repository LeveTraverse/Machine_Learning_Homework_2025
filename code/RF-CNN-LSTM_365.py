import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
from sklearn.ensemble import RandomForestRegressor

# 初始设置和数据加载函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_process(filepath):
    df = pd.read_csv(filepath, sep=",", low_memory=False)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df = df.dropna(subset=["DateTime"])
    df.set_index("DateTime", inplace=True)

    float_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "RR",
        "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    df_daily = pd.DataFrame()
    df_daily['Global_active_power'] = df['Global_active_power'].resample('D').sum()
    df_daily['Global_reactive_power'] = df['Global_reactive_power'].resample('D').sum()
    df_daily['Voltage'] = df['Voltage'].resample('D').mean()
    df_daily['Global_intensity'] = df['Global_intensity'].resample('D').mean()
    df_daily['Sub_metering_1'] = df['Sub_metering_1'].resample('D').sum()
    df_daily['Sub_metering_2'] = df['Sub_metering_2'].resample('D').sum()
    df_daily['Sub_metering_3'] = df['Sub_metering_3'].resample('D').sum()
    df_daily['RR'] = df['RR'].resample('D').first()
    df_daily['NBJRR1'] = df['NBJRR1'].resample('D').first()
    df_daily['NBJRR5'] = df['NBJRR5'].resample('D').first()
    df_daily['NBJRR10'] = df['NBJRR10'].resample('D').first()
    df_daily['NBJBROU'] = df['NBJBROU'].resample('D').first()
    df_daily["sub_metering_remainder"] = (
        df_daily["Global_active_power"] * 1000 / 60 -
        (df_daily["Sub_metering_1"] + df_daily["Sub_metering_2"] + df_daily["Sub_metering_3"])
    )
    df_daily = df_daily.dropna()
    return df_daily

# 构造样本函数
def create_sequences(data, input_len=90, output_len=90, target_idx=0):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, target_idx])
    return np.array(X), np.array(y)

# 加载原始训练和测试数据
train_df_original = load_and_process("train.csv")
test_df_original = load_and_process("test.csv")

# 定义输入和输出序列长度
input_steps = 90
output_steps = 365

# 原始数据的标准化
scaler_for_rf = MinMaxScaler()
train_scaled_for_rf = scaler_for_rf.fit_transform(train_df_original)

target_feature_index_original = 0

# 为随机森林创建序列数据
X_train_for_rf, y_train_for_rf = create_sequences(
    train_scaled_for_rf,
    input_len=input_steps,
    output_len=output_steps,
    target_idx=target_feature_index_original
)

print(f"X_train shape for RF: {X_train_for_rf.shape}")
print(f"y_train shape for RF: {y_train_for_rf.shape}")

# 随机森林特征选择

print("\n--- 开始随机森林特征重要性评估 ---")

# 展平 X_train_for_rf，使其适合随机森林的输入形状 (样本数, 特征总数)
# 特征总数 = input_steps * 原始特征数量
num_original_features = train_df_original.shape[1]
X_train_flattened = X_train_for_rf.reshape(X_train_for_rf.shape[0], -1)
# 对于随机森林的目标，我们使用每个序列的第一个预测值
y_train_rf_target = y_train_for_rf[:, 0]
# 训练随机森林回归器
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_flattened, y_train_rf_target)
# 获取特征重要性
importances = rf_model.feature_importances_

# 构建可读的特征名称列表
original_feature_names = train_df_original.columns.tolist()
flattened_feature_names = []
for step_idx in range(input_steps):
    for feature_name in original_feature_names:
        flattened_feature_names.append(f"{feature_name}_t-{input_steps - 1 - step_idx}")

# 将重要性与特征名关联并排序
feature_importances_df = pd.DataFrame({
    'Feature': flattened_feature_names,
    'Importance': importances
})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

print("\n--- 随机森林特征重要性排名 (前20) ---")
print(feature_importances_df.head(20))

# 计算每个原始特征的总重要性
original_feature_total_importances = {name: 0.0 for name in original_feature_names}
for _, row in feature_importances_df.iterrows():
    original_feature_name = row['Feature'].split('_t-')[0]
    original_feature_total_importances[original_feature_name] += row['Importance']

original_feature_importance_df = pd.DataFrame({
    'Original_Feature': list(original_feature_total_importances.keys()),
    'Total_Importance': list(original_feature_total_importances.values())
})
original_feature_importance_df = original_feature_importance_df.sort_values(by='Total_Importance', ascending=False)

print("\n--- 原始特征的总重要性排名 ---")
print(original_feature_importance_df)

# 选择最重要的特征
top_k_features = 8
selected_features = original_feature_importance_df['Original_Feature'].head(top_k_features).tolist()

# 确保目标特征 'Global_active_power' 始终包含在 selected_features 中
if 'Global_active_power' not in selected_features:
    selected_features.insert(0, 'Global_active_power') # 把它放在第一个位置
    selected_features = list(dict.fromkeys(selected_features)) # 去重并保持顺序

print(f"\n--- 最终选定的 {len(selected_features)} 个特征 ---")
print(selected_features)

# 基于选择的特征重新准备数据

print("\n--- 重新准备数据 (仅包含选定特征) ---")

train_df_filtered = train_df_original[selected_features]
test_df_filtered = test_df_original[selected_features]

# 对筛选后的数据重新进行标准化
scaler_filtered = MinMaxScaler()
train_scaled_filtered = scaler_filtered.fit_transform(train_df_filtered)
test_scaled_filtered = scaler_filtered.transform(test_df_filtered)

# 更新特征数量
n_features_filtered = len(selected_features)

# 找到目标特征在筛选后的特征列表中的新索引
target_feature_index_filtered = selected_features.index('Global_active_power')

# 使用筛选后的数据创建序列
X_train, y_train = create_sequences(
    train_scaled_filtered,
    input_len=input_steps,
    output_len=output_steps,
    target_idx=target_feature_index_filtered
)
X_test, y_test = create_sequences(
    test_scaled_filtered,
    input_len=input_steps,
    output_len=output_steps,
    target_idx=target_feature_index_filtered
)

print(f"重新准备后 X_train shape: {X_train.shape}")
print(f"重新准备后 y_train shape: {y_train.shape}")
print(f"重新准备后 X_test shape: {X_test.shape}")
print(f"重新准备后 y_test shape: {y_test.shape}")

# 创建新的 DataLoader
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PowerDataset(X_train, y_train)
test_ds = PowerDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)


# RF+CNN+LSTM 混合模型定义

class RF_CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len,
                 cnn_out_channels=64, kernel_size=3):
        super().__init__()

        # 输入特征嵌入层
        self.embedding = nn.Linear(input_size, 100)

        # 1D CNN 层，用于局部特征提取
        self.cnn = nn.Conv1d(
            in_channels=100,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu_cnn = nn.ReLU()
        self.bn_cnn = nn.BatchNorm1d(cnn_out_channels)

        # LSTM 层
        self.lstm = nn.LSTM(cnn_out_channels, hidden_size, num_layers, batch_first=True)

        # 全连接层（输出层）
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_len)

    def forward(self, x):
        # 特征嵌入
        x = self.embedding(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.1, train=self.training)

        # CNN 处理
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.bn_cnn(x)
        x = self.relu_cnn(x)

        # 为 LSTM 准备输入
        x = x.permute(0, 2, 1)

        # LSTM 处理
        out, _ = self.lstm(x)

        # 全连接层
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)
        out = torch.dropout(out, 0.4, train=self.training)

        # 残差连接
        residual = self.fc2(out)
        out = self.fc3(out + residual)

        out = torch.dropout(out, 0.1, train=self.training)
        out = torch.relu(out)
        out = torch.dropout(out, 0.1, train=self.training)
        out = self.fc4(out)

        return out

# 实例化 RF+CNN+LSTM 模型
model = RF_CNN_LSTM_Model(
    input_size=n_features_filtered,
    hidden_size=128,
    num_layers=4,
    output_len=output_steps,  # 预测未来365天
    cnn_out_channels=64,
    kernel_size=3
).to(device)

print(f"\n--- RF+CNN+LSTM 模型结构 ---")
print(model)

# 模型训练
criterion = nn.MSELoss()
MAEcriterion = nn.L1Loss()
loss_list = []

# === 第一轮训练：lr = 0.001 ===
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs_phase1 = 20
print("\n--- 开始第一阶段训练 (lr=0.001) ---")
for epoch in range(epochs_phase1):
    model.train()
    total_loss = 0
    total_mae = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        mae = MAEcriterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae.item()
    avg_loss = total_loss / len(train_loader)
    avg_mae = total_mae / len(train_loader)
    loss_list.append(avg_loss)
    print(f"[Phase 1 - Epoch {epoch+1}] Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}")

# === 第二轮训练：lr = 0.0001 ===
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
epochs_phase2 = 35
print("\n--- 开始第二阶段训练 (lr=0.0001) ---")
for epoch in range(epochs_phase2):
    model.train()
    total_loss = 0
    total_mae = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        mae = MAEcriterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae.item()
    avg_loss = total_loss / len(train_loader)
    avg_mae = total_mae / len(train_loader)
    loss_list.append(avg_loss)
    print(f"[Phase 2 - Epoch {epoch+1}] Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}")

# === 可视化 Loss 变化 ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# 模型评估和可视化
print("\n--- 开始模型评估 (滑动窗口预测) ---")
model.eval()
all_preds_sliding_window = []
all_trues_sliding_window = []

start_index_for_prediction = input_steps

with torch.no_grad():
    # 循环遍历测试集中每个可以进行预测的日期
    for i in range(start_index_for_prediction, len(test_scaled_filtered)):
        # 提取当前窗口的输入数据：前 input_steps 天的数据
        current_input_window = test_scaled_filtered[i - input_steps: i]
        xb = torch.tensor(current_input_window, dtype=torch.float32).unsqueeze(0).to(device)
        pred_sequence = model(xb).cpu().numpy()

        single_day_pred_scaled = pred_sequence[0, 0]
        all_preds_sliding_window.append(single_day_pred_scaled)

        # 获取当前天的真实值
        single_day_true_scaled = test_scaled_filtered[i, target_feature_index_filtered]
        all_trues_sliding_window.append(single_day_true_scaled)

all_preds_sliding_window = np.array(all_preds_sliding_window)
all_trues_sliding_window = np.array(all_trues_sliding_window)

print(f"Collected {len(all_preds_sliding_window)} single-day predictions.")

# 反归一化函数
def inverse_transform_single_feature_series(scaled_values, scaler_obj, feature_index, n_features_in_scaler):
    dummy_array = np.zeros((len(scaled_values), n_features_in_scaler))
    dummy_array[:, feature_index] = scaled_values
    original_values = scaler_obj.inverse_transform(dummy_array)[:, feature_index]
    return original_values

# 执行反归一化
preds_inv = inverse_transform_single_feature_series(
    all_preds_sliding_window,
    scaler_filtered,
    target_feature_index_filtered,
    n_features_filtered
)
trues_inv = inverse_transform_single_feature_series(
    all_trues_sliding_window,
    scaler_filtered,
    target_feature_index_filtered,
    n_features_filtered
)

# 计算最终的MSE和MAE
mse = mean_squared_error(trues_inv, preds_inv)
mae = mean_absolute_error(trues_inv, preds_inv)

print(f"最终评估 (RF+CNN+LSTM 混合模型) — MSE: {mse:.2f}, MAE: {mae:.2f}")

# 绘制总体的预测结果
plt.figure(figsize=(18, 7))
plt.plot(trues_inv, label="True Global Active Power", color='blue')
plt.plot(preds_inv, label="Predicted Global Active Power", color='red', linestyle='--')
plt.title("Global Active Power Forecast (Sliding Window Prediction on Test Set - RF+CNN+LSTM)")
plt.xlabel(f"Days in Test Set (Starting from Day {input_steps + 1})")
plt.ylabel("Global Active Power (Wh)")
plt.legend()
plt.grid(True)
plt.show()

# 绘制局部放大图
num_days_to_plot = min(200, len(preds_inv)) # 绘制前200天

plt.figure(figsize=(18, 7))
plt.plot(trues_inv[:num_days_to_plot], label="True Global Active Power", color='blue')
plt.plot(preds_inv[:num_days_to_plot], label="Predicted Global Active Power", color='red', linestyle='--')
plt.title(f"Global Active Power Forecast (First {num_days_to_plot} Days of Sliding Window Prediction - RF+CNN+LSTM)")
plt.xlabel("Days (Relative to Prediction Start)")
plt.ylabel("Global Active Power (Wh)")
plt.legend()
plt.grid(True)
plt.show()