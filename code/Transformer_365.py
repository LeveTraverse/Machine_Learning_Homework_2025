import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import math

# 查看设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载和预处理函数
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

# 加载训练和测试数据
train_df = load_and_process("train.csv")
test_df = load_and_process("test.csv")

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

target_feature_index = 0
n_features = train_scaled.shape[1]

# 定义输入和输出序列长度
input_steps = 90  # 过去90天的数据
output_steps = 365 # 预测未来365天的数据

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

# Transformer 模型定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, output_len, dropout=0.1, input_steps=90):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.input_steps = input_steps

        # 线性层将输入特征投影到d_model维度
        self.encoder_linear = nn.Linear(input_size, d_model)
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_steps)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 前馈网络的维度
            dropout=dropout,
            batch_first=True  # 保持批次维度在前
        )

        # Transformer编码器堆栈
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 输出层：取最后一个时间步的输出
        self.decoder_linear1 = nn.Linear(d_model, 64)
        self.decoder_linear2 = nn.Linear(64, 64)
        self.bn = nn.BatchNorm1d(64)
        self.decoder_linear3 = nn.Linear(64, output_len)

    def forward(self, src):
        # 将输入特征投影到d_model维度
        src = self.encoder_linear(src)
        # 添加位置编码
        src = self.pos_encoder(src)
        # 通过Transformer编码器
        output = self.transformer_encoder(src)
        # 取最后一个时间步的输出作为整个序列的表示
        output = output[:, -1, :]
        # 通过解码器的线性层进行预测
        output = self.decoder_linear1(output)
        output = torch.relu(output)
        output = output + self.decoder_linear2(output)
        output = self.bn(output)
        output = self.decoder_linear3(output)

        return output


# 实例化 Transformer 模型
d_model = 64  # 模型内部维度，需能被 nhead 整除
nhead = 16  # 注意力头的数量
num_encoder_layers = 1  # 编码器层数
dropout_rate = 0.0  # dropout率

model = TransformerModel(
    input_size=n_features,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    output_len=output_steps,
    dropout=dropout_rate,
    input_steps=input_steps
).to(device)

criterion = nn.MSELoss()
MAEcriterion = nn.L1Loss()
loss_list = []

# === 第一轮训练：lr = 0.001 ===
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs_phase1 = 12
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs_phase2 = 20
for epoch in range(epochs_phase2):
    # model.train()
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

# 评估模型
model.eval()
all_preds_sliding_window = []
all_trues_sliding_window = []

start_index_for_prediction = input_steps

with torch.no_grad():
    for i in range(start_index_for_prediction, len(test_scaled)):
        current_input_window = test_scaled[i - input_steps: i]
        xb = torch.tensor(current_input_window, dtype=torch.float32).unsqueeze(0).to(device)

        pred_sequence = model(xb).cpu().numpy()
        single_day_pred_scaled = pred_sequence[0, 0]
        all_preds_sliding_window.append(single_day_pred_scaled)

        single_day_true_scaled = test_scaled[i, target_feature_index]
        all_trues_sliding_window.append(single_day_true_scaled)

all_preds_sliding_window = np.array(all_preds_sliding_window)
all_trues_sliding_window = np.array(all_trues_sliding_window)

print(f"Collected {len(all_preds_sliding_window)} single-day predictions.")
print(f"Shape of collected predictions: {all_preds_sliding_window.shape}")


# 反归一化函数
def inverse_transform_single_feature_series(scaled_values, scaler_obj, feature_index, n_features_in_scaler):
    dummy_array = np.zeros((len(scaled_values), n_features_in_scaler))
    dummy_array[:, feature_index] = scaled_values
    original_values = scaler_obj.inverse_transform(dummy_array)[:, feature_index]
    return original_values


preds_inv = inverse_transform_single_feature_series(all_preds_sliding_window, scaler, target_feature_index, n_features)
trues_inv = inverse_transform_single_feature_series(all_trues_sliding_window, scaler, target_feature_index, n_features)

mse = mean_squared_error(trues_inv, preds_inv)
mae = mean_absolute_error(trues_inv, preds_inv)

print(f"Final Evaluation — MSE: {mse:.2f}, MAE: {mae:.2f}")

# 绘制预测与真实值曲线
plt.figure(figsize=(18, 7))
plt.plot(trues_inv, label="True Global Active Power", color='blue')
plt.plot(preds_inv, label="Predicted Global Active Power", color='red', linestyle='--')
plt.title("Global Active Power Forecast (Sliding Window Prediction on Test Set - Transformer Model)")
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