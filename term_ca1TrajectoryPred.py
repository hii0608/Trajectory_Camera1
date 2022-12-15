## 카메라 한 대, 차량 전체 차량에 대해 9초 후 10초때의 위치 예측
'''
    TrajectoryPred.py와 다른 점 : TrajectoryPred는 하나의 파일로 모두 합쳐서 50프레임 간격으로 학습, 예측하는데
    이 파일은 50프레임 마다 나뉜 데이터프레임의 한 로우씩 가져다 학습
    -> 1, 2, 3, ... , 20일 뒤를 예측하도록 만들려면 변경해야함

'''
import pandas as pd
import numpy as np

import matplotlib
import glob, os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
import random

from pylab import mpl, plt

from datetime import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

def test(testX, testY, model) :
    # 데이터 하나 당 epoch 씩 학습
    for i in range(1):
        test_X = torch.Tensor(testX)
        test_y = torch.Tensor(testY)

        #todo - loss 가 이 위치 또는 더 상위에 있어야 하나?
        test_X = torch.Tensor(test_X).to(device)
        test_y = torch.Tensor(test_y).to(device)
        y_test_pred = model(test_X)

        loss = loss_fn(y_test_pred[-1], test_y[-1])
        print("test loss : ", loss.item())
        return loss.item()
        # test_predict = model(test_X)


# 일반적인 sequential data로 변환 - 한 개 df에 대해 
# 각각 normalize 하면 denormalize 가 힘들어서 전체 값에서 normalize 함
def split_seq(seq,window,horizon,scaler_):

    df = pd.DataFrame({"x" : seq[0], "y" : seq[1]})
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_.fit_transform(df[['x','y']].values)
    df = scaled_data

    X=[]; Y=[]
    for i in range(len(seq[0])-(window+horizon)+1):
        x=df[i:(i+window)]
        y=df[i+window+horizon-1]

        # x=df.iloc[i:(i+window)]
        # y=df.iloc[i+window+horizon-1]
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1])
        return out

def train(trainX, trainY, model) : 
  for t in range(1) : 
  # for t in range(num_epochs): #궤적 데이터 하나에 대한 epoch 
    trainX = torch.Tensor(trainX)
    trainY = torch.Tensor(trainY)

    y_train_pred = model(trainX)

    loss = loss_fn(y_train_pred, trainY)

    hist[t] = loss.item()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

  return model, loss



if __name__ == '__main__':

    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    with open('./data/xyposList1120.pickle', 'rb') as f:    
      data = pickle.load(f)

    print(data[:400])

    totalX = []
    totalY = []
    #for i in range(epoch) -> 데이터를 원하는 horizontal coefficient 크기로 하고, 해당 데이터 넘어가면, 다음 데이터
    #그 후 에폭.. 전체 데이터에 대해서.. 하고 그 다음에 다음 데이터로 넘어가기...
    for carIdx in range(len(data)) : 
      for frameIdx in range(len(data[carIdx][0])) : 
          totalX.append(data[carIdx][0][frameIdx])
          totalY.append(data[carIdx][1][frameIdx])
    df = pd.DataFrame({"x" : totalX,  "y": totalY})
  
    with open("data/merged_1121.pickle", "wb") as fw:
        pickle.dump(df, fw)
   
    window = 49 #며칠 전의 값 참고? # 마지막 프레임
    horizon = 1 #얼마나 먼 미래? #마지막 프레임의 위치 예측

    num_epochs = 200
    hist = np.zeros(num_epochs)

    flag = int(len(data) * 0.7) # 

    trainData = data[:flag] # 2744
    testData = data[flag:] # 1173

    input_dim = 2
    hidden_dim = 128
    num_layers = 4
    output_dim = 2
    
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()


    scaler_ = MinMaxScaler(feature_range=(0, 1))
    #train - 2735
    trainData = trainData[:300]
    for ep in range(num_epochs) : 
        print("epoch : ", ep)
        print("epoch : ", ep)
        print("epoch : ", ep)
        totalLoss = 0
        for idx, row in enumerate(trainData) :
            trainX, trainY = split_seq(row, window, horizon,scaler_)
            model, loss = train(trainX, trainY, model)
            totalLoss+=loss.item()
        print("total Loss mean : ",totalLoss/len(trainData[0]))

    # todo 모델 저장
    torch.save(model,  './model/term_car1_model_e200_d30.pt')

    model = torch.load('./model/term_car1_model_e200_d30.pt').to(device)
    totalLoss = 0
    for idx, row in enumerate(testData) :
      testX, testY = split_seq(row, window, horizon,scaler_)  
      totalLoss += (test(testX, testY, model))
    
    print("totalLoss mean : ", totalLoss/len(testData))
    print("totalLoss mean : ", totalLoss/len(testData))
    print("total(testData): ", len(testData))
      
    

    
    # start = time.time()
    # print(start)

    # test(testData)
    # end = time.time()

    # print(end)
    # print(end-start)