import torch
from kan import KAN
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

df1 = pd.read_csv("./ELECTROENCEPHALOGRAM_OF_SENSITIVITY_BODIES/specified_finger.csv",
                  delimiter=";", decimal=".", header=0)
df2 = pd.read_csv("./ELECTROENCEPHALOGRAM_OF_SENSITIVITY_BODIES/thumb_finger.csv",
                  delimiter=";", decimal=".", header=0)

df1["target"] = "specified_finger"
df2["target"] = "thumb_finger"


first_10k = df1.iloc[:10000]
second_10k = df2.iloc[-10000:]


df = pd.concat([first_10k, second_10k])
#Removing duplicate rows
df.drop_duplicates(subset=None, keep='first', inplace=True)
#Reset index of dataframe
df.reset_index(drop=True, inplace=True)
#Return column names of dataframe
column_names = list(df.columns)
#Checking data balance
target_var = df[column_names[-1]]
balance = Counter(target_var)
#Initialization of encoder
labelencoder = LabelEncoder()
#Encoding target variable
df["target"] = labelencoder.fit_transform(df["target"])

"""Splitting independent 'x' and dependent 'y'  variables of dataframe 'df'"""
x = df[column_names[0:len(column_names)-1]]
#Standardization or mean removal and variance scaling
x = (x-x.mean())/x.std()
#Converting to numpy array
x = x.values
#Create target variable
y = df[column_names[len(column_names)-1]].values

"""Splitting on training sample and test sample"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train = torch.from_numpy(x_train).clone().detach()
y_train = torch.from_numpy(y_train).clone().detach()

x_test = torch.from_numpy(x_test).clone().detach()
y_test = torch.from_numpy(y_test).clone().detach()

custom_dataset = {
    'train_input': x_train,
    'test_input': x_test,
    'train_label': y_train.view(-1, 1),
    'test_label': y_test.view(-1, 1)
}

model = KAN(width=[16,5,1], grid=5, k=3)

def mean_squared_error(y_true, y_guess):
    return np.mean((y_true - y_guess) ** 2)

def train_acc():
    mse = mean_squared_error(custom_dataset['train_label'][:,0].detach().numpy(), model(custom_dataset['train_input'])[:,0].detach().numpy())
    return 1 / (1 + mse)

def test_acc():
    mse = mean_squared_error(custom_dataset['test_label'][:,0].detach().numpy(), model(custom_dataset['test_input'])[:,0].detach().numpy())
    return 1 / (1 + mse)


results = model.train(custom_dataset, opt="LBFGS", steps=15,
                      metrics=(train_acc, test_acc), beta=7)

model.prune()
model.plot(beta=2, scale=1, title='Trained KAN Model')

y_pred = model(x_test)
y_pred = y_pred.detach().numpy()
y_pred[y_pred < 0] = 0
y_pred = np.round(y_pred) 

a_score = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
roc_auc_score = roc_auc_score(y_test, y_pred)

# lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
# model.auto_symbolic(lib=lib)

# formula1, formula2 = model.symbolic_formula()[0]

# def acc(formula1, formula2, X, y):
#     batch = X.shape[0]
#     correct = 0
#     for i in range(batch):
#         logit1 = np.array(formula1.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
#         logit2 = np.array(formula2.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
#         correct += (logit2 > logit1) == y[i]
#     return correct/batch

# print('train acc of the formula:', acc(formula1, formula2, custom_dataset['train_input'], custom_dataset['train_label']))
# print('test acc of the formula:', acc(formula1, formula2, custom_dataset['test_input'], custom_dataset['test_label']))





