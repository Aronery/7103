import  cv2
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import numpy as np
import operator
# 导入数据
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn import tree
from sklearn.model_selection import train_test_split
#health 1
#ill 0
data_health = pd.read_csv("H:/python_project/7103/Gut Microbiome Dataset/Gut Microbiome Dataset/CRC_D015179/abundance_mat_species_D006262_CRC.csv")
data_ill =pd.read_csv("H:/python_project/7103\Gut Microbiome Dataset/Gut Microbiome Dataset/CRC_D015179/abundance_mat_species_D015179_CRC.csv")
#print(data_health.shape)

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
    x_train, x_test, y_train, y_test=input_data()
    # clf = RandomForestClassifier(n_estimators=10,max_features = "sqrt")
    clf = RandomForestClassifier(random_state=42)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)

    #####测试模型

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 绘制混淆矩阵的热力图
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=['health', 'ill'], columns=['health', 'ill'])
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
    ax.set_xlabel('True Label')
    ax.set_ylabel('Predicted Label')
    plt.show()

    # 计算 ROC 曲线和 AUC 值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n = 2
    for i in range(n):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype('int'), y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制 ROC 曲线
    plt.figure()
    lw = 2
    for i in range(n):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # 打印 AUC 值
    print("AUC (area under ROC curve):")
    for i in range(n):
        print("\tClass {}: {:0.2f}".format(i, roc_auc[i]))