##使用DNN搭建神经网络
import  cv2
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import category_encoders as ce
import numpy as np
import keras as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.special as ssp
from keras.layers import Dense, Dropout

def pro_process():
    data_path = r"H:/python_project/7103/data_24fall/path_to_file.csv"
    df = pd.read_csv(data_path)
    mapping = {
        0: 'A',
        1: 'B',
        2: 'C',
        3:"D"
    }

    df['Target'] = df['Target'].replace(mapping)
    return df


# 读取CSV数据集，并拆分为训练集和测试集
# 该函数的传入参数为CSV_FILE_PATH: csv文件路径
def load_data(IRIS):
    target_var = 'Target'  # 目标变量
    # 数据集的特征
    features = list(IRIS.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = IRIS[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量进行编码
    IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(IRIS['target'])
    y_bin_labels = []  # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        IRIS['y' + str(i)] = transformed_labels[:, i]
    # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(IRIS[features], IRIS[y_bin_labels],
                                                        train_size=0.75, test_size=0.25, random_state=0)
    return train_x, test_x, train_y, test_y, Class_dict



if __name__ == "__main__":
    df=pro_process()
    train_x, test_x, train_y, test_y, Class_dict=load_data(df)

    # 2. 定义模型
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    adam_optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.add(K.layers.Dense(units=18, input_dim=9, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=36, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=18, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=9, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.20))
    model.add(K.layers.Dense(units=4, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    # 3. 训练模型
    b_size = 8
    max_epochs =500
    print("Starting training ")
    h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)

    print("Training finished \n")

    # 4. 评估模型
    eval = model.evaluate(test_x, test_y, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
          % (eval[0], eval[1] * 100))




######z








