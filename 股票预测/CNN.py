import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('E:\毕业论文\数据\GOOG.csv')

close = df['Adj Close'].values.reshape(-1,1)
scaler = MinMaxScaler()
close_scaled = scaler.fit_transform(close)

train_size = int(len(close_scaled) * 0.8)
test_size = len(close_scaled) - train_size
train_data = close_scaled[0:train_size,:]
test_data = close_scaled[train_size:len(close_scaled),:]

def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:i+time_steps])
        y.append(dataset[i+time_steps])
    return np.array(X), np.array(y)

# 用30天的数据去预测下一天的数据
time_steps = 30
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 模型搭建
model = Sequential()
model.add(Conv1D(64, 4, padding='same', activation='relu', input_shape=(time_steps, 1)))  # 64个filter卷积核 大小4
model.add(MaxPooling1D(2))  # 每两个取一个大的   数据会减少一半
model.add(Flatten())  # 把二维数据变成一维的
model.add(Dense(32))  # 全连接层,神经元个数
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))  # 输出层 是一个一维的全连接神经网络
model.compile(loss='mse', optimizer=Nadam(learning_rate=0.001))
model.summary()
# 模型训练
model.fit(X_train, y_train, epochs=50, batch_size=64)

# 预测测试集
y_predict = model.predict(X_test)

# 反归一化处理
y_test = scaler.inverse_transform(y_test)
y_predict = scaler.inverse_transform(y_predict)

# 评估预测结果
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
corr_coef = np.corrcoef(y_test.T, y_predict.T)[0][1]
print('RMSE:', rmse)
print('MAE:', mae)
print('R2:', r2)
print('Correlation coefficient:', corr_coef)

# 数据可视化
plt.figure(figsize=(16,8))
plt.plot(y_test, label='Actual Price')
plt.plot(y_predict, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('RMSE='+str(rmse), y=-0.12)
plt.legend()
plt.show()