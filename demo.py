import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense

# 假设 df 是你的数据框
df = pd.read_csv('5_data_0101.csv')

# 选择特征和目标变量
features = df[['channel', 'band_width', 'self_chan_use', 'tx_self_chan_use', 'channel_use',
               'vap_num', 'nss', 'mul_chan_use', 'beacon_chan_use', 'probe_chan_use', 'tx_power']]
targets = df[['up_flow', 'down_flow']]

# 归一化特征
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 创建时间序列数据
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10
X, y = create_sequences(features_scaled, targets.values, SEQ_LENGTH)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, X.shape[2])))
model.add(Dense(2))  # 预测 up_flow 和 down_flow
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 进行预测
predictions = model.predict(X_test)
