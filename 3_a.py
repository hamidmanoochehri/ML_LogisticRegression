import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def SVM_SGD(df, epoch, intial_w, gamma_0, d, variance):
    objectives = []
    w = intial_w
    for t in range(0, epoch):
        gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)
        df_shuffle = df.sample(frac=1)
        df_shuffle = df_shuffle.reset_index(drop=True)
        N = df_shuffle.shape[0]
        for row in range(0, df_shuffle.shape[0]):
            x = np.reshape(df_shuffle.iloc[row, :-1].to_numpy(), (df.shape[1] - 1, 1))
            y = df_shuffle.iloc[row, -1]
            s = y * np.dot(w, x)
            derive_w = w / variance - N * y * (1 - sigmoid(s)) * (x.transpose())
            w = w - gamma_t * derive_w
        objective = loss(df_shuffle, w, variance)
        objectives.append(objective)
    return w, objectives


def loss(df, w, variance):
    loss_sum = 0
    for row in range(0, df.shape[0]):
        x = np.reshape(df.iloc[row, :-1].to_numpy(), (df.shape[1] - 1, 1))
        y = df.iloc[row, -1]
        s = -y * np.dot(w, x)
        loss = np.log(1 + np.exp(s))
        loss_sum = loss_sum + loss
    reg = np.dot(w, w.transpose()) / (2 * variance)
    obj = reg + loss_sum
    return obj[0][0]


def predict(df, w):
    error = 0
    for row in range(0, df.shape[0]):
        predict = 0
        x = df.iloc[row, :-1]
        y = df.iloc[row, -1]
        p1 = w.dot(x)
        predict = np.sign(p1)
        if predict == 0:
            predict = 1
        if predict != y:
            error = error + 1
    return error / df.shape[0]


columns = ["variance", "skewness", "curtosis", "entropy", "label"]

# Reading training and testing data
train = pd.read_csv(r'nn\train.csv', names=columns, dtype=np.float64())
train = train.replace({"label": 0}, -1)
test = pd.read_csv(r'nn\test.csv', names=columns, dtype=np.float64())
test = test.replace({"label": 0}, -1)

train.insert(0, "b", 1)
test.insert(0, "b", 1)

# Setting hyperparameter
print("Epoch value")
epoch = int(input())
print("Gamma_0 value")
gamma_0 = float(input())
print("D value")
d = float(input())

initial_w = np.zeros((1, train.shape[1] - 1))
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

for variance in variances:
    final_w, obj = SVM_SGD(train, epoch, initial_w, gamma_0, d, variance)
    plt.plot(obj)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    title = "gamma_0=" + str(gamma_0) + ", d=" + str(d) + ", variance=" + str(variance)
    plt.title(title)
    plt.show()
    train_error = predict(train, final_w)
    test_error = predict(test, final_w)
    print(title)
    print("train_error", train_error)
    print("test_error", test_error, "\n")
