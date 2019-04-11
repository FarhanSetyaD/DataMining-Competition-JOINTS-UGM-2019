import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.elu(self.fc1(X))
        X = self.fc2(X)
        X = F.leaky_relu(X)
        X = self.fc3(X)
        X = F.sigmoid(X)
        X = self.fc4(X)
        X = self.softmax(X)

        return X
    
# load IRIS dataset
dataset = pandas.read_csv('data/train_num2.csv')
dataset = dataset.drop(columns=['id', 'fac_7'])
dataset = dataset.fillna(dataset.mean())

# transform species to numerics
x_data = dataset.drop(columns='gender')
y_data = dataset['gender']
target_names = ['putri', 'campur', 'putra']

#normalize data
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

train_X, test_X, train_y, test_y = train_test_split(x_data, y_data, test_size=0.8)

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y.values).long())
test_y = Variable(torch.Tensor(test_y.values).long())


net = Net()

criterion = nn.CrossEntropyLoss(
    weight = torch.Tensor([1, 2.789, 1.827])
    )# cross entropy loss

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(10000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss)

predict_out = net(test_X)

_, predict_y = torch.max(predict_out, 1)
print(predict_y)
print('accuracy: ', accuracy_score(test_y, predict_y))



