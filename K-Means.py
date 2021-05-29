import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # create dataset
    train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
    train = pd.read_csv(train_url)
    test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
    test = pd.read_csv(test_url)


    # analize dataset
    print("***** Train_Set *****")
    print(train.head())
    print("\n")
    print("***** Test_Set *****")
    print(test.head())

    print("***** Train_Set *****")
    print(train.describe())
    print("\n")
    print("***** Test_Set *****")
    print(test.describe())

    print(train.columns.values)

    # We need to deal with missing values due to K-Means not supporting it

    print("*****In the train set*****")
    print(train.isna().sum())
    print("\n")
    print("*****In the test set*****")
    print(test.isna().sum())

    # replace empty values with Mean value
    train.fillna(train.mean(), inplace=True)
    test.fillna(test.mean(), inplace=True)

    print("*****In the train set*****")
    print(train.isna().sum())
    print("\n")
    print("*****In the test set*****")
    print(test.isna().sum())

    train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();
    plt.show()

    train.info()

    # remove useless data
    train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # transform non-numeric data into numeric
    labelEncoder = LabelEncoder()
    labelEncoder.fit(train['Sex'])
    labelEncoder.fit(test['Sex'])
    train['Sex'] = labelEncoder.transform(train['Sex'])
    test['Sex'] = labelEncoder.transform(test['Sex'])

    train.info()
    test.info()

    # creating K-Means model

    # create dataset without "survived"
    X = np.array(train.drop(['Survived'], 1).astype(float))
    y = np.array(train['Survived'])
    train.info()

    kmeans = KMeans(n_clusters=2)  # You want cluster the passenger records into 2: Survived or Not survived
    kmeans.fit(X)

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct / len(X))

    # optimization of model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans.fit(X_scaled)

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct / len(X))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
