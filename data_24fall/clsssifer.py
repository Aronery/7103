import  cv2
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
data_train=pd.read_csv("Train.csv")
data_test=pd.read_csv("Test.csv")
data_validata=pd.read_csv("Validate.csv")


#print(data_train.columns)
#print(data_train.shape)



def Nancy(array,a):
    """

    :param array: 需要填充的数组
    :param n_neighbors: K—NN的参数
    :return: 修改后的参数
    """
    imputer = KNNImputer(n_neighbors=a)
    df_filled = pd.DataFrame(imputer.fit_transform(array), columns=array.columns)
    return df_filled

# print(data_train["Profession"])
# a=Nancy(data_train,3)
# print(a.shape)



def data_split():
    #data_train['target'] = pd.factorize(data_train['Class(Target)'])[0]
    data_train["Gender"]=data_train["Gender"].fillna(np.random.choice([0, 1]))#随机填充性别
    #data_train["Graduate"] = Nancy(data_train["Graduate"],2)
    #data_train["Profession"] = Nancy(data_train["Profession"],3)
    data_train["Graduate"] = data_train["Graduate"].fillna(np.random.choice([0, 1]))
    data_train["Profession"] = data_train["Profession"].fillna("Artist")
    #按比例等分
    data_train["Years_of_Working "] = data_train["Years_of_Working "].fillna(data_train["Years_of_Working "].median())
    data_train["Family_Members"] = data_train["Family_Members"].fillna(2)
    data_train["Category"] = data_train["Category"].fillna("Cat_6")
    #独热码
    df_Profession = pd.get_dummies(data_train, columns=['Profession'])
    df_Spending_Score = pd.get_dummies(df_Profession, columns=['Spending_Score'])
    df_Category = pd.get_dummies(df_Spending_Score, columns=['Category'])

    y_target = df_Category["Class(Target)"].values
    x_features_one = df_Category.drop(['ID', 'Class(Target)'], axis=1)
    # print(y_target)
    # print(x_features_one)
    x_train, x_test, y_train, y_test = train_test_split(x_features_one, y_target, test_size=0.25,
                                                                    random_state=66)
    # print(x_train.shape)
    # print(y_train.shape)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    ###随机森林
    # x_train, x_test, y_train, y_test=data_split()
    # clf = RandomForestClassifier(random_state=8,n_estimators=100)
    # clf = clf.fit(x_train, y_train)
    #
    # y_pred = clf.predict(x_test)
    # y_score = clf.predict_proba(x_test)
    #
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Precision:", precision_score(y_test, y_pred, average='macro'))
    # print("Recall:", recall_score(y_test, y_pred, average='macro'))
    # print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))


    ###########决策树
    # #x_train, x_test, y_train, y_test = data_split()
    # x_train, x_test, y_train, y_test=data_split()
    # clf = DecisionTreeClassifier()
    # clf = clf.fit(x_train, y_train)
    #
    # y_pred = clf.predict(x_test)
    # y_score = clf.predict_proba(x_test)
    #
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Precision:", precision_score(y_test, y_pred, average='macro'))
    # print("Recall:", recall_score(y_test, y_pred, average='macro'))
    # print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    #### SVM 0.4305555555555556     liner:0.44604          #################################
    x_train, x_test, y_train, y_test=data_split()
    #svm = SVC(kernel='rbf', gamma='scale',probability=True)
    svm = SVC(kernel="linear", gamma='scale', probability=True)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    y_score = svm.predict_proba(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))







