# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:27:04 2019

@author: ZuoYS
"""

import scipy.io as scio
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# 交叉验证，10次，每次取一列作为验证集
def cross_validate(train_data):                  
    y_rbf_list = []
    for i in range(10):
        index = np.random.randint(0, 19)         # 随机数，整数，范围0至19
        validate_set = train_data[:,index]
        validate_set = np.expand_dims(validate_set, 1)
        train_set_new = np.delete(train_data, index, axis=1)   # numpy delete方法
        y_rbf = process_per_area(train_set_new, validate_set)
        y_rbf_list.append(y_rbf)
    y_rbf_list = np.array(y_rbf_list)
    
    return y_rbf_list.T
        

def process_per_area(train_data, test_data):
    k=5                                                       # k控制对异常值的容忍程度，越大越能容忍
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.5)             # C与k相反，越大越不能容忍异常值；gamma设大容易过拟合
    X = np.mat(range(1, 144  + 1)).T
    area_1_train = []
    for i in range(train_data.shape[0]):
        area_1_time = train_data[i,:]
        area_1_time_new = []
        Q1 = np.percentile(area_1_time,25)
        Q3 = np.percentile(area_1_time,75)
        for j in range(len(area_1_time)):
            if Q1-k*(Q3-Q1) <= area_1_time[j] <= Q3+k*(Q3-Q1):  #清除异常值
                area_1_time_new.append(area_1_time[j])
        area_1_time_new = np.array(area_1_time_new)
        area_1_time_avg = np.median(area_1_time_new)
        area_1_train.append(area_1_time_avg)
    area_1_train = np.array(area_1_train)
    y_rbf = svr_rbf.fit(X, area_1_train).predict(X)             # SVR拟合
        
    return y_rbf


dataFile = 'demand_10.mat'
data_read = scio.loadmat(dataFile)                              # read .mat file, return a dict
data = data_read['demand']                                      # Find data term
mse = 0                                                         # 各项指标，MSE,MAE,R2_score
mae = 0
r2 = 0
mse_area = 0
mae_area = 0
path_out = './result_area/'
for i in range(data.shape[1]):
    area = data[:,i]
    area_train = area[0:2880]
    area_train = area_train.reshape(144, 20, order='F')
    area_test = area[2880:3024]
    #area_test = np.expand_dims(area_test, 1)
    y_rbf_list = cross_validate(area_train)
    y_predict = np.mean(y_rbf_list, 1)
    for j in range(y_predict.shape[0]):
        y_predict[j] = math.ceil(y_predict[j])
        mse_area += (y_predict[j] - area_test[j])**2
        mae_area += np.abs(y_predict[j] - area_test[j])
    mse_area = mse_area/area_train.shape[0]
    mae_area = mae_area/area_train.shape[0]
    #y_rbf = process_per_area(area_train, area_test)
    mse += mse_area
    mae += mae_area
    r2 += r2_score(area_test, y_predict)
    
    '''plt.plot(area_test, label='Test')                      # 输出预测的图和真实的测试图
    plt.plot(y_predict, label='Predict')
    plt.title('Area '+ str(i+1) + ' predict curve')
    plt.xlabel('Time')
    plt.ylabel('Number of cars')
    plt.legend(fontsize=12)
    plt.savefig(path_out + str(i+1) + '.png')
    plt.clf()'''
    
mse_avg = mse/data.shape[1]
mae_avg = mae/data.shape[1]
r2_avg = r2/data.shape[1]
print('Predict MSE: %6.2f, MAE: %2.2F, R2_score: %.2f' % (mse_avg, mae_avg, r2_avg))