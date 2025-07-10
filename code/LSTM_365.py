import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os

# 查看设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_process(filepath):
    df = pd.read_csv(filepath, sep=",", low_memory=False)

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df = df.dropna(subset=["DateTime"])  # 去除无效时间
    df.set_index("DateTime", inplace=True)

    # 强制数值转换，避免 object 类型
    float_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "RR",
        "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # 删除所有含 NaN 行

    # 按天聚合
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

# 文件路径
train_file = 'train.csv'
test_file = 'test.csv'

# 加载训练和测试数据
train_df = load_and_process("train.csv")
test_df = load_and_process("test.csv")

# 统一标准化器，对所有特征进行fit_transform
scaler = MinMaxScaler()

# 确保 train_df 和 test_df 的列顺序一致
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# 获取目标变量 'Global_active_power' 在标准化数组中的索引
target_feature_index = 0
n_features = train_scaled.shape[1] # 特征数量

# 定义输入和输出序列长度
input_steps = 90  # 使用过去90天的数据
output_steps = 365 # 预测未来365天的数据

# 构造样本（输入90天，输出365天）
def create_sequences(data, input_len, output_len, target_idx):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, target_idx])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, input_len=input_steps, output_len=output_steps, target_idx=target_feature_index)
X_test, y_test = create_sequences(test_scaled, input_len=input_steps, output_len=output_steps, target_idx=target_feature_index)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super().__init__()
        self.embedding = nn.Linear(input_size, 100)
        self.lstm1 = nn.LSTM(100, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_len)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.dropout(x, 0.4, train=True)
        out, _ = self.lstm1(x)
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)
        out = torch.dropout(out, 0.2, train=True)
        out = self.fc3(out + self.fc2(out))
        out = torch.dropout(out, 0.1, train=True)
        out = torch.relu(out)
        out = torch.dropout(out, 0.1, train=True)
        out = self.fc4(out)

        return out

model = LSTMModel(input_size=X_train.shape[2], hidden_size=128, num_layers=4, output_len=output_steps).to(device)

criterion = nn.MSELoss()
MAEcriterion = nn.L1Loss()
loss_list = []

# === 第一轮训练：lr = 0.001 ===
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs_phase1 = 20
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
import matplotlib.pyplot as plt
plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

model.eval()
all_preds_sliding_window = []
all_trues_sliding_window = []

start_index_for_prediction = input_steps

with torch.no_grad():
    # 循环遍历测试集中每个可以进行预测的日期
    for i in range(start_index_for_prediction, len(test_scaled)):
        # 提取当前窗口的输入数据：前 input_len 天的数据
        current_input_window = test_scaled[i - input_steps: i]
        xb = torch.tensor(current_input_window, dtype=torch.float32).unsqueeze(0).to(device)
        # 进行预测，模型会输出未来 output_len 天的预测
        pred_sequence = model(xb).cpu().numpy()

        single_day_pred_scaled = pred_sequence[0, 0]
        all_preds_sliding_window.append(single_day_pred_scaled)

        # 获取当前天的真实值
        single_day_true_scaled = test_scaled[i, target_feature_index]
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
preds_inv = inverse_transform_single_feature_series(all_preds_sliding_window, scaler, target_feature_index, n_features)
trues_inv = inverse_transform_single_feature_series(all_trues_sliding_window, scaler, target_feature_index, n_features)

# 计算最终的MSE和MAE
mse = mean_squared_error(trues_inv, preds_inv)
mae = mean_absolute_error(trues_inv, preds_inv)

print(f"Final Evaluation — MSE: {mse:.2f}, MAE: {mae:.2f}")

plt.figure(figsize=(18, 7))
plt.plot(trues_inv, label="True Global Active Power", color='blue')
plt.plot(preds_inv, label="Predicted Global Active Power", color='red', linestyle='--')
plt.title("Global Active Power Forecast (Sliding Window Prediction on Test Set)")
plt.xlabel(f"Days in Test Set (Starting from Day {input_steps + 1})")
plt.ylabel("Global Active Power (Wh)")
plt.legend()
plt.grid(True)
plt.show()

# 绘制局部放大图
num_days_to_plot = min(200, len(preds_inv))  # 绘制前200天

plt.figure(figsize=(18, 7))
plt.plot(trues_inv[:num_days_to_plot], label="True Global Active Power", color='blue')
plt.plot(preds_inv[:num_days_to_plot], label="Predicted Global Active Power", color='red', linestyle='--')
plt.title(f"Global Active Power Forecast (First {num_days_to_plot} Days of Sliding Window Prediction)")
plt.xlabel("Days (Relative to Prediction Start)")
plt.ylabel("Global Active Power (Wh)")
plt.legend()
plt.grid(True)
plt.show()