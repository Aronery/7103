import  cv2
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import category_encoders as ce
import numpy as np

import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# using target encoding
# Tutorial: https://www.kaggle.com/ryanholbrook/target-encoding
def target_encoding(name, df, m=1):
    df[name] = df[name].str.split(";")
    df = df.explode(name)
    overall = df["label"].mean()
    df = df.groupby(name).agg(
        freq=("label", "count"),
        in_category=("label", np.mean)
    ).reset_index()
    df["weight"] = df["freq"] / (df["freq"] + m)
    df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
    return df


def data_process(data_train):
    #data_train = pd.read_csv("Train.csv")
    encoder = LabelEncoder()
    #data_train['target'] = pd.factorize(data_train['Class(Target)'])[0]


    data_train["Gender"]=data_train["Gender"].fillna(np.random.choice([0, 1]))#随机填充性别
    data_train["Graduate"] = data_train["Graduate"].fillna(np.random.choice([0, 1]))
    data_train["Profession"] = data_train["Profession"].fillna("Artist")
    #按比例等分
    data_train["Years_of_Working "] = data_train["Years_of_Working "].fillna(data_train["Years_of_Working "].median())
    data_train["Family_Members"] = data_train["Family_Members"].fillna(2)
    data_train["Category"] = data_train["Category"].fillna("Cat_6")

    # encoder_columns = ['Profession', 'Spending_Score',"Category"]
    # encoder = ce.TargetEncoder(cols=encoder_columns)
    # df_encoded = encoder.fit_transform(data_train[encoder_columns], data_train['target'])
    # df_encoded = pd.DataFrame(df_encoded, columns=encoder_columns + ['target'])
    #
    # data_train = data_train.join(df_encoded,rsuffix='_other')
    #x_features_one = data_train.drop(
        #['ID', 'Class(Target)', "Profession", "Spending_Score", "Category", "Class(Target)", "target_other"], axis=1)

    ##使用pd进行目标编码
    # data_train['Profession'] = pd.factorize(data_train['Profession'])[0]
    # data_train['Spending_Score'] = pd.factorize(data_train['Spending_Score'])[0]
    # data_train['Category'] = pd.factorize(data_train['Category'])[0]

    label_encoder = LabelEncoder()
    data_train['Profession'] =  label_encoder.fit_transform(data_train['Profession'])
    data_train['Spending_Score'] = label_encoder.fit_transform(data_train['Spending_Score'])
    data_train['Category'] = label_encoder.fit_transform(data_train['Category'])
    data_train['Target'] = label_encoder.fit_transform(data_train['Class(Target)'])

    y_target = data_train["Target"].values
    x_features_one = data_train.drop(['ID', 'Class(Target)'], axis = 1)
    # print(y_target)
    # print(x_features_one)
    # x_train, x_test, y_train, y_test = train_test_split(x_features_one, y_target, test_size=0.25,
    #                                                                 random_state=50)
    # print(x_train.shape)
    # print(y_train.shape)
    return x_features_one,y_target

def data_split(x_features_one,y_target):
    x_train, x_test, y_train, y_test = train_test_split(x_features_one, y_target, test_size=0.25,
                                                                    shuffle=True, random_state=66)

    return x_train, x_test, y_train, y_test



if __name__ == "__main__":
    data_train = pd.read_csv("path_to_file.csv")
    y_target=data_train["Target"]
    x_features_one = data_train.drop("Target", axis=1)

    ###随机森林
    x_train, x_test, y_train, y_test=data_split(x_features_one,y_target)
    clf = RandomForestClassifier(random_state=8,n_estimators=100)
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    #####使用在验证集上
    #
    # x_features_one_val, y_target_val=data_process(data_validata)
    # y_pred_val = clf.predict(x_features_one_val)
    # y_score_val = clf.predict_proba(x_features_one_val)
    # print("Accuracy_val:", accuracy_score(y_target_val, y_pred_val))