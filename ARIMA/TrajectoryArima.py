'''
## 카메라 한 대, 차량 전체 차량에 대해 9초 후 10초때의 위치 예측
  총 50frame이 카메라 1 내에서 얻어낸 차량 한 대의 궤적 데이터
  목표 : 한 대 당 1~20 frame  뒤의 위치를 예측함

  데이터셋 목표 형태 : 50 frame 단위로 데이터 셋을 생성

  x, y 좌표 각각에 대해 학습, 정규화 X


-> ARIMA  모델을 이용해 예측
X좌표와, Y 좌표에 대해 각각 예측


50개 데이터 기준으로 20스텝 뒤를 예측.
궤적 하나 끝나면 다음거... 

일단 50개 기준으로 step 에 맞춰서 예측하도록.. 

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

import math, time
import itertools
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
import argparse
import statsmodels.api as sm
from config import arima_parse
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import font_manager, rc


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

'''
  기존 값이 시간순으로 value 만 있는걸 확인했음
  그러면 50개마다 돌아가면서 넣어주면 될 것 같고, 
  그 때 step을 조절해주면 될 것 같음.-> split_seq도 미니배치를 만드는거 말곤 굳이..? 필요 없나본데..
'''

def Visualize(pred_arima_y, test_y, path) :
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 10))
    # 모델이 예측한 가격 그래프
    plt.plot(pred_arima_y, color='gold')
    # 실제 가격 그래프
    plt.plot(test_y, color='green')

    plt.legend(['예측값', '실제값'])
    plt.title("값 비교")
    plt.show()
    plt.savefig(path)

    return

def calRMSE(data, columnN, args) :
    data = data[:50*10]
    data = data[columnN]
    
    start_date='20211120'
    date_list=pd.date_range(start=start_date, periods=len(data), freq='1T').astype(int) #일단 50개

    df = pd.DataFrame({"dTime" : date_list, columnN : data})

    
    orderList = []
    trueList = []
    predList = []
    meanRMSE = []
    for j in range(1,21 ) : 
      order = (j, 0 ,j )
      RMSEList = []
      for i in range(0, len(df)//50) : 
        
        s = i*50
        e = (i+1)*50
        # print(s, " : ", e )
        nowDf = df[s:e]
        print(nowDf[columnN][-1:].values)

        model = ARIMA(nowDf[columnN], order = order)
        model_fit = model.fit()
        model_fit.save('model/ARIMA_fit_x_{}_{}_{}.pt'.f(order[0],order[1],order[2]))

        # model = ARIMA(nowDf[columnN], order=args.order)
        # model_fit = model.fit()
        # model_fit.save(args.model_save)

        # print("여기 : ", model_fit.summary())
        
        # preds = model_fit.predict(1, 50, typ='levels')1F
        # preds = model_fit.forecast(24*7, typ='levels')
        
        preds = model_fit.forecast(50, typ='levels')
        print("preds[-1:].values : ", preds[-1:].values)
        # sys.exit()

        RMSE = mean_squared_error(nowDf[columnN][-1:].values, preds[-1:].values)**0.5
        # print("nowDf[columnN][-1:].values: ", nowDf[columnN][-1:].values)
        # print("preds[-1:].values : ", preds[-1:].values)
        print(RMSE)
        RMSEList.append(RMSE)
      orderList.append(j)
      meanRMSE.append(sum(RMSEList)/len(RMSEList))
    
    return orderList, meanRMSE

    
    
    return 

# 스텝 수 조절
# RMSE - test set 


def main():
    parser = argparse.ArgumentParser(description='Embedding arguments')
    arima_parse(parser)
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)
    with open('./data/merged_1121.pickle', 'rb') as f:    
        data = pickle.load(f)

    # print(data['x'][49])

    # print(data['x'][50])

    # print(data['x'][51])

    # sys.exit()
    
    orderList, XRMSE = calRMSE(data, 'x', args)
    _, YRMSE = calRMSE(data, 'y', args)

    print("XRMSE : ", XRMSE)
    print("YRMSE : ", YRMSE)


    df2.pd.DataFrame({"order":orderList, "xRMSE" : XRMSE, "yRMSE" : YRMSE})
    print(df2)
    df2.to_csv("./output/RMSExy.csv")





if __name__ == '__main__':
    main()
