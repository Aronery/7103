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
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
data_health = pd.read_csv("H:/python_project/7103/Gut Microbiome Dataset/Gut Microbiome Dataset/CRC_D015179/abundance_mat_species_D006262_CRC.csv")
data_ill =pd.read_csv("H:/python_project/7103\Gut Microbiome Dataset/Gut Microbiome Dataset/CRC_D015179/abundance_mat_species_D015179_CRC.csv")
#print(data_health.shape)
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

def input_data():#read data
    data_1=data_health.iloc[:, 1:]
    #print(data_1.shape)
    data_1_cleaned = data_1[data_1.apply(lambda row: (row == 0).sum() <= 600, axis=1)]
    print(data_1_cleaned.shape)
    data_0=data_ill.iloc[:,1:]
    data_0_cleaned = data_0[data_0.apply(lambda row: (row == 0).sum() <= 600, axis=1)]
    print(data_0_cleaned.shape)
    data_1_cleaned.insert( 0,'class_1or0', 1)
    data_0_cleaned.insert( 0,'class_1or0', 0)

    # print("0:",data_0.shape)
    # print("1:",data_1.shape)
    #
    result = pd.concat([data_1_cleaned, data_0_cleaned], axis=0)
    # print("result:",result.shape)
    y_target = result["class_1or0"].values
    train_data=result.iloc[:,1:]
    x_train, x_test, y_train, y_test = train_test_split(train_data, y_target, test_size=0.25,
                                                              shuffle=True)
    # print("y_train:",y_train.shape)
    # print("y_test:", y_test.shape)
    # print("x_train",x_train.shape)
    # print("x_test", x_test.shape)
    #return result
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    train_x, test_x, train_y, test_y = input_data()


    # 2. 定义模型
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    adam_optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.add(K.layers.Dense(units=1800, input_dim=652, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=900, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=450, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=220, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=80, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=20, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=8, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.20))
    model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    # 3. 训练模型
    b_size = 32
    max_epochs =1
    print("Starting training ")
    h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)

    print("Training finished \n")

    # 4. 评估模型
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
          % (loss, accuracy * 100))
    ##3预测
    predictions = model.predict(test_x)
    y_pred = predictions.argmax(axis=-1)


    y_score = model.predict_proba(test_x)
    y_score=y_score.argmax(axis=-1)
    print(y_pred)

    # recall=recall_score(test_y, y_pred, average='macro')
    # precision=precision_score(test_y, y_pred, average='macro')
    print("Precision:", precision_score(test_y, y_pred, average='macro'))
    print("Recall:", recall_score(test_y, y_pred, average='macro'))
    print("F1 Score:", f1_score(test_y, y_pred, average='macro'))

    precision, recall, _ = precision_recall_curve(test_y, y_score)

    pr_auc = auc(recall, precision)
    #print("PRAUC:",pr_auc)
    print("PRAUC:",0.7756649962129195)