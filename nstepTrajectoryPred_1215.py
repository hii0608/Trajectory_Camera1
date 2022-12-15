## 카메라 한 대, 차량 전체 차량에 대해 9초 후 10초때의 위치 예측
'''
  총 50frame이 카메라 1 내에서 얻어낸 차량 한 대의 궤적 데이터
  목표 : 한 대 당 1~20 frame  뒤의 위치를 예측함
  -> 데이터 셋 변경 필요 ;

  데이터셋 목표 형태 : 50 frame 단위로 데이터 셋을 생성함

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
import gc
import torch


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False



# 일반적인 sequential data로 변환 - 한 개 df에 대해 
# 각각 normalize 하면 denormalize 가 힘들어서 전체 값에서 normalize 함
def split_seq(seq,window,horizon):

  scaler_ = MinMaxScaler(feature_range=(0, 1))
  # scaler 제외함 - 2022 12 15
  scaled_data = scaler_.fit_transform(seq[['x','y']].values)
  # seq = seq[['x','y']].values
  X = []; Y = []
  for carIdx in range(0, len(seq), 50) : 
    # dt = seq[carIdx:carIdx+50]
    dt = scaled_data[carIdx:carIdx+50]


    # scaled_data : 예측할 step : 1~20
    # window+horizon : 참고할 값. 동일 조건에서 step에 따라 예측 성능을 비교하기 위해 10frame으로 설정함
    # print(len(dt)-(window+horizon)+1)# 40
    # print(window) # 10
    # print(horizon) # 1
    for idx in range(len(dt)-(window+horizon)+1):
      # print("idx :")
      x=dt[idx:(idx+window)]
      y=dt[idx+window+horizon-1]
      # print("y : ", y)
      # print("x : ",x)
      # print("idx+window+horizon-1 : ",idx+window+horizon-1)
      X.append(x)
      Y.append(y)

  return np.array(X), np.array(Y), scaler_
    # return X, Y, scaler_ 

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



def train(trainX, trainY, model, num_epochs) : 
  loss_fn = torch.nn.MSELoss()
  optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
   

  for t in range(num_epochs):
    # print(np.array(trainX))
    # print(np.array(trainX))
    # sys.exit()
    train_X = torch.Tensor(trainX).to(device)
    train_y = torch.Tensor(trainY).to(device)

    y_train_pred = model(train_X).to(device)
    
    loss = loss_fn(y_train_pred, train_y)

    x_loss = loss_fn(y_train_pred[:, 0], train_y[:, 0])
    y_loss = loss_fn(y_train_pred[:, 1], train_y[:, 1])

    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
        # print("x_loss : ", x_loss.item())
        # print("y_loss : ", y_loss.item())

    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()
    train_predict = model(train_X).to(device)

  return model, loss

def test(testData,  horizon) :
  # 데이터 하나 당 epoch 씩 학습
  testX, testY, scaler = split_seq(testData, window, horizon)

  model = torch.load('./camera1/model/win_25_step_{}_model_e2000_d1800.pt'.format(horizon)).to(device)
 
  loss_fn = torch.nn.MSELoss()
  test_X = torch.Tensor(testX).to(device)
  test_y = torch.Tensor(testY).to(device)

  y_test_pred = model(test_X)

  loss = loss_fn(y_test_pred, test_y)
  # print("test loss : ", loss.item())

  pred_x = y_test_pred[:, 0]
  pred_y = y_test_pred[:, 1]
  
  print(pred_x)
  print(test_y[:,0])

  sys.exit()

  # print(pred_x)
  # print(pred_y)
  # sys.exit()

  x_loss = loss_fn(y_test_pred[:, 0], test_y[:, 0])
  y_loss = loss_fn(y_test_pred[:, 1], test_y[:, 1])

  # print("x_loss : ", x_loss)
  # print("y_loss : ", y_loss)
  
  x_loss_r = torch.sqrt(x_loss)
  y_loss_r = torch.sqrt(y_loss)
  # print("x_loss_r : ", x_loss_r)
  # print("y_loss_r : ", y_loss_r)
  
  print("testMSE - MSE x - y - RMSE x - y")
  print(loss.item(), " ",x_loss.item(), " ", y_loss.item(), " ",x_loss_r.item(), " ",y_loss_r.item()  )

  gc.collect()
  torch.cuda.empty_cache()

  return loss.cpu().item(), x_loss.cpu().item(), y_loss.cpu().item(), x_loss_r.cpu().item(), y_loss_r.cpu().item()

  # test_predict = model(test_X)


if __name__ == '__main__':
  gc.collect()
  torch.cuda.empty_cache()


  print(torch.cuda.is_available())
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # device = torch.device("cpu")
  # map_location=torch.device('cpu')
  print(device)

  with open('./camera1/data/merged_1121.pickle', 'rb') as f:    
    data = pickle.load(f)

  # data = data[:50*1800]
  data = data[:50*1300]
  print(data[:20])


  num_epochs = 2000
  hist = np.zeros(num_epochs)
  
  
  window = 25 #며칠 전의 값 참고? # 마지막 프레임
  horizon = 1 #얼마나 먼 미래? #마지막 프레임의 위치 예측

  flag = int(len(data) * 0.7) # 

  trainData = data.iloc[:flag] # 2744
  testData = data.iloc[flag:] # 1173

  lossList = []
  horizonList = []
  xlossList = []  
  ylossList = []
  xlossRList = [] 
  ylossRList = []
  xTrueList = []
  yTrueList = []

  input_dim = 2
  hidden_dim = 128
  num_layers = 4
  output_dim = 2










  # testLoop(window, horizon,testData)
  for horizonNum in range(1,21) : 
    gc.collect()
    torch.cuda.empty_cache()

    loss, x_loss, y_loss, x_loss_r, y_loss_r = test(testData, horizonNum)
    lossList.append(loss)
    horizonList.append(horizonNum)
    xlossList.append(x_loss)
    ylossList.append(y_loss)
    xlossRList.append(x_loss_r)
    ylossRList.append(y_loss_r)
  
  pd.set_option('display.max_colwidth', None)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  df1 = pd.DataFrame({"horizon" : horizonList, "loss(MSE)" : lossList, "xloss" : xlossList, "yloss" : ylossList, "xlossRMSE" : xlossRList, "ylossRMSE" : ylossRList})
  print(df1)
  df1.to_csv('./camera1/result/xyMSE_RMSE-testNum1173-01-.csv')

  