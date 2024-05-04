import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Attention, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 使用GPU计算
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 导入训练数据和测试数据
train_data_AAPL = pd.read_csv('E:\毕业论文\数据\AAPL.csv')
test_data_AMZN = pd.read_csv('E:\毕业论文\数据\AMZN.csv')
test_data_GOOG = pd.read_csv('E:\毕业论文\数据\GOOG.csv')
test_data_TSLA = pd.read_csv('E:\毕业论文\数据\TSLA.csv')

# 提取收盘价
AAPL_close = train_data_AAPL['Adj Close'].values.reshape(-1,1)
AMZN_close = test_data_AMZN['Adj Close'].values.reshape(-1,1)
GOOG_close = test_data_GOOG['Adj Close'].values.reshape(-1,1)
TSLA_close = test_data_TSLA['Adj Close'].values.reshape(-1,1)

# 归一化
scaler = MinMaxScaler()
AAPL_close_scaled = scaler.fit_transform(AAPL_close)
AMZN_close_scaled = scaler.fit_transform(AMZN_close)
GOOG_close_scaled = scaler.fit_transform(GOOG_close)
TSLA_close_scaled = scaler.fit_transform(TSLA_close)

# 创建时间序列数据集
def create_dataset(data,time_step):
    x, y = [], []
    for i in range(len(data)-time_step):
        x.append([a for a in data[i:i+time_step]])
        y.append(data[i+time_step])
    x = np.array(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    y = np.array(y)
    return x, y

# 时间步长为30
time_steps = 30

# 训练数据集
AAPL_x_train, AAPL_y_train = create_dataset(AAPL_close_scaled, time_steps)
AAPL_x_train = np.reshape(AAPL_x_train, (AAPL_x_train.shape[0], AAPL_x_train.shape[1], 1))

#测试数据集
AAPL_x_test, AAPL_y_test = create_dataset(AMZN_close_scaled, time_steps)
AMZN_x_test, AMZN_y_test = create_dataset(AMZN_close_scaled, time_steps)
GOOG_x_test, GOOG_y_test = create_dataset(GOOG_close_scaled, time_steps)
TSLA_x_test, TSLA_y_test = create_dataset(TSLA_close_scaled, time_steps)

# 模型搭建与训练
def create_CNN():
    """
    构建卷积神经网络（CNN）模型。

    Returns:
        model: 编译好的CNN模型。
    """

    # 创建一个顺序模型
    model = Sequential()

    # 添加一个卷积层
    # 参数说明：
    #   - 64: 卷积核的数量
    #   - 4: 卷积核的大小
    #   - padding='same': 使用padding='same'保持输入和输出的大小一致
    #   - activation='relu': 使用ReLU激活函数
    #   - input_shape=(time_steps, 1): 输入数据的形状（时间步长，特征数）
    model.add(Conv1D(64, 4, padding='same', activation='relu', input_shape=(time_steps, 1)))

    # 添加一个最大池化层
    # 参数说明：
    #   - 2: 池化窗口的大小
    model.add(MaxPooling1D(2))

    # 添加一个展平层，将二维数据变成一维的
    model.add(Flatten())

    # 添加一个全连接层
    # 参数说明：
    #   - 32: 全连接层的神经元数量
    model.add(Dense(32))

    # 添加一个Dropout层，用于防止过拟合
    # 参数说明：
    #   - 0.2: Dropout比例，即丢弃输入单元的比例
    model.add(Dropout(0.2))

    # 添加一个激活函数层，继续使用ReLU激活函数
    model.add(Activation('relu'))

    # 添加输出层，输出一个一维的全连接神经网络
    model.add(Dense(1))

    # 编译模型，选择优化器和损失函数
    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.01))

    # 训练模型
    # 参数说明：
    #   - AAPL_x_train: 输入数据
    #   - AAPL_y_train: 标签数据
    #   - epochs: 训练轮数
    #   - batch_size: 批次大小
    model.fit(AAPL_x_train, AAPL_y_train, epochs=50, batch_size=64)

    # 返回训练好的模型
    return model

def create_RNN():
    """
    构建简单循环神经网络（RNN）模型。

    Returns:
        model: 编译好的RNN模型。
    """

    # RNN模型搭建
    model = Sequential()

    # 添加一个SimpleRNN层
    # 参数说明：
    #   - units=50: SimpleRNN层中的神经元数量
    #   - activation='relu': 使用ReLU激活函数
    #   - input_shape=(time_steps, 1): 输入数据的形状（时间步长，特征数）
    model.add(SimpleRNN(units=50, activation='relu', input_shape=(time_steps, 1)))

    # 添加一个全连接层
    # 参数说明：
    #   - units=1: 全连接层中的神经元数量
    #   - activation='linear': 线性激活函数，保持输出值的范围
    model.add(Dense(units=1, activation='linear'))

    # 编译模型，选择优化器和损失函数
    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.01))

    # 训练模型
    # 参数说明：
    #   - AAPL_x_train: 输入数据
    #   - AAPL_y_train: 标签数据
    #   - epochs: 训练轮数
    #   - batch_size: 批次大小
    model.fit(AAPL_x_train, AAPL_y_train, epochs=50, batch_size=64)

    # 返回训练好的模型
    return model

def create_LSTM():
    """
    构建长短期记忆网络（LSTM）模型。

    Returns:
        model: 编译好的LSTM模型。
    """

    # LSTM模型搭建
    model = Sequential()

    # 添加第一个LSTM层
    # 参数说明：
    #   - units=50: LSTM层中的神经元数量
    #   - return_sequences=True: 返回完整的输出序列
    #   - input_shape=(time_steps, 1): 输入数据的形状（时间步长，特征数）
    model.add(LSTM(units=50, input_shape=(time_steps, 1)))

    # 添加一个全连接层
    # 参数说明：
    #   - units=1: 全连接层中的神经元数量
    model.add(Dense(units=1))

    # 编译模型，选择优化器和损失函数
    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.01))

    # 训练模型
    # 参数说明：
    #   - AAPL_x_train: 输入数据
    #   - AAPL_y_train: 标签数据
    #   - epochs: 训练轮数
    #   - batch_size: 批次大小
    model.fit(AAPL_x_train, AAPL_y_train, epochs=50, batch_size=64)

    # 返回训练好的模型
    return model

def create_GRU():
    """
    构建一个具有多个GRU层的模型。

    参数:
        time_steps (int): 输入数据中的时间步长。
        features (int): 输入数据中的特征数。

    返回:
        model: 编译好的GRU模型。
    """

    # 创建Sequential模型
    model = Sequential()

    # 添加第一个GRU层
    # 参数说明：
    #   - units=100: GRU层中的神经元数量
    #   - return_sequences=True: 返回完整的输出序列
    #   - input_shape=(time_steps, features): 输入数据的形状（时间步长，特征数）
    model.add(GRU(units=64, return_sequences=True, input_shape=(30, 1)))

    # 添加Dropout层，减少过拟合的可能性
    model.add(Dropout(0.2))

    # 添加第二个GRU层
    # 这里return_sequences设为True，以便将输出序列传递给下一层GRU
    model.add(GRU(units=32, return_sequences=True))

    # 再次添加Dropout层
    model.add(Dropout(0.2))

    # 添加第三个GRU层
    # 不再需要return_sequences=True，因为这是最后一个GRU层，不需要将输出序列传递给下一层
    model.add(GRU(units=16))

    # 添加全连接层
    # 输出层的单元数量为1，因为我们的任务是回归问题，预测一个数值
    model.add(Dense(units=1))

    # 编译模型，选择优化器和损失函数
    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.01))

    # 训练模型
    # 参数说明：
    #   - AAPL_x_train: 输入数据
    #   - AAPL_y_train: 标签数据
    #   - epochs: 训练轮数
    #   - batch_size: 批次大小
    model.fit(AAPL_x_train, AAPL_y_train, epochs=50, batch_size=64)

    # 返回训练好的模型
    return model

def create_LSTM_self():
    inputs = Input(shape=(30, 1))
    lstm_output = LSTM(units=50, return_sequences=True)(inputs)
    attention_output = Attention()([lstm_output, lstm_output])
    output = Dense(units=1)(attention_output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.01))
    model.fit(AAPL_x_train, AAPL_y_train, epochs=50, batch_size=64)
    return model

# 测试AMZN股票数据
def Predict(name, x_test, y_test, model):
    # 预测
    y_predict = model.predict(x_test)
    y_predict = y_predict.reshape(-1, 30)  # 将三维数组转换为二维数组
    y_predict = y_predict[:, -1].reshape(-1, 1)

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
    plt.rcParams['font.family'] = 'Microsoft YaHei'# Microsoft YaHei, SimHei
    plt.figure(figsize=(16, 8))
    plt.suptitle(name+' Predict')
    plt.plot(y_test, label='真实价格')
    plt.plot(y_predict, label='预测价格')
    plt.xlabel('时间')
    plt.ylabel('收盘价')
    plt.title('RMSE=' + str(rmse), y=-0.12)
    plt.legend()
    plt.show()
    return


import matplotlib.pyplot as plt
from keras.callbacks import History


def create_CNN_and_plot_losses():
    model = Sequential()

    # 添加一个SimpleRNN层
    # 参数说明：
    #   - units=50: SimpleRNN层中的神经元数量
    #   - activation='relu': 使用ReLU激活函数
    #   - input_shape=(time_steps, 1): 输入数据的形状（时间步长，特征数）
    model.add(SimpleRNN(units=100, activation='relu', input_shape=(time_steps, 1)))

    # 添加一个全连接层
    # 参数说明：
    #   - units=1: 全连接层中的神经元数量
    #   - activation='linear': 线性激活函数，保持输出值的范围
    model.add(Dense(units=1, activation='linear'))

    # 编译模型，选择优化器和损失函数
    model.compile(loss='mse', optimizer=Nadam(learning_rate=0.001))

    # 定义一个History对象来记录训练过程中的指标
    history = History()

    # 训练模型，并将训练过程记录在history对象中
    model.fit(AAPL_x_train, AAPL_y_train, epochs=100, batch_size=32, validation_data=(AAPL_x_test, AAPL_y_test),
              callbacks=[history])

    # 绘制训练和验证误差曲线
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return model


# 调用函数来创建模型并绘制误差曲线
model = create_CNN_and_plot_losses()
