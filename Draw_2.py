# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:06:05 2019

@author: ZuoYS
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Avgpooling_Bilinear_PSNR_list = pd.read_csv('single_image_PSNR_list_Gaussian_Avgpooling_Bilinear.csv', header=None).to_numpy()
#Avgpooling_Bicubic_PSNR_list = pd.read_csv('single_image_PSNR_list_Gaussian_Avgpooling_Bicubic.csv', header=None).to_numpy()
#Maxpooling_Bilinear_PSNR_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bilinear.csv', header=None).to_numpy()
#Maxpooling_Bicubic_PSNR_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic.csv', header=None).to_numpy()
x = np.arange(0, 5000, 1)
#x_1 = np.arange(0, 4999, 1)
x_2 = np.arange(0, 1899, 1)
DIP_list = pd.read_csv('DIP_single_image_PSNR_list_Gaussian_lena_5000iters.csv', header=None).to_numpy()  #读取csv文件，注意"header=None"，当csv中没有标题信息时要加。把第一行也读进来
DD_list = pd.read_csv('DD_single_image_PSNR_list_Gaussian_lena.csv', header=None).to_numpy()
DED_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic_lena_5000iter.csv', header=None).to_numpy()
DED_3X3_Conv_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic_lena_5000iter_3x3conv.csv', header=None).to_numpy()
DED_imgcross_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic_Cross_lena_3000iter.csv', header=None).to_numpy()
DED_setcross_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic_cross_set_lena_1900iter.csv', header=None).to_numpy()
DED_L1Loss_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic_lena_1900iter_L1.csv', header=None).to_numpy()
DED_L1ToL2_list = pd.read_csv('single_image_PSNR_list_Gaussian_L1_to_L2_lena_1900iter.csv', header=None).to_numpy()
DED_L2ToL1_list = pd.read_csv('single_image_PSNR_list_Gaussian_L2_to_L1_lena_1900iter.csv', header=None).to_numpy()
DED_down2_list = pd.read_csv('single_image_PSNR_list_Gaussian_Maxpooling_Bicubic_First_lena_down2_1900iter.csv', header=None).to_numpy()
#Diff_DED_DD = DED_list - DED_imgcross_list

'''DIP_fit = np.polyfit(x,DIP_list,13)   #polyfit，函数拟合，后面的参数表示多少次的多项式
DD_fit = np.polyfit(x,DD_list,13)
DED_fit = np.polyfit(x,DED_list,13)
Diff_fit = np.polyfit(x,Diff_DED_DD,13)
DED_3X3_Conv_fit = np.polyfit(x,DED_3X3_Conv_list,13)
DED_imgcross_fit = np.polyfit(x,DED_imgcross_list,13)
DED_L1loss_fit = np.polyfit(x,DED_L1Loss_list,13)

DIP_vals = np.polyval(DIP_fit,x)    #polyval，做完拟合以后还要和x对应
DD_vals = np.polyval(DD_fit,x)
DED_vals = np.polyval(DED_fit,x)
Diff_vals = np.polyval(Diff_fit,x)
DED_3X3_Conv_vals = np.polyval(DED_3X3_Conv_fit,x)
DED_imgcross_vals = np.polyval(DED_imgcross_fit,x)
DED_L1loss_vals = np.polyval(DED_L1loss_fit,x)'''
plt.figure(figsize=(8, 6))

'''plt.plot(Avgpooling_Bilinear_PSNR_list, color='pink', label='Avgpooling_Bilinear')
plt.plot(Avgpooling_Bicubic_PSNR_list,  color='lightblue', label='Avgpooling_Bicubic')
plt.plot(Maxpooling_Bilinear_PSNR_list, color='yellow', label='Maxpooling_Bilinear')
plt.plot(Maxpooling_Bicubic_PSNR_list, color='green', label='Maxpooling_Bicubic')
plt.plot(DIP_list, color='red', label='DIP')
plt.plot(DD_list, color='coral', label='Deep decoder')
plt.plot(DED_list, color='green', label='DED with 1x1 Conv')
plt.plot(DED_3X3_Conv_list,color='red',label='DED with 3x3 Conv')'''
plt.plot(DED_imgcross_list[0:1900],color='green',label='DED with down2 img_cross')
plt.plot(DED_down2_list,label='DED with down2 single image')
#plt.plot(DED_setcross_list, color='blue', label='DED with set_cross')
#plt.plot(DED_L1Loss_list, color='blue', label='DED with L1 loss')
#plt.plot(DED_L1ToL2_list, label='DED with L1-L2 loss')
#plt.plot(DED_L2ToL1_list, label='DED with L2-L1 loss')
plt.plot(DED_list[0:1900], color='red', label='DED with single image')

'''plt.plot(x,DIP_vals,color='red',label='DIP')
plt.plot(x,DD_vals,color='blue',label='Deep decoder')
plt.plot(x_1,DED_3X3_Conv_vals,color='red',label='DED with 3x3 Conv')
plt.plot(x_1,DED_imgcross_vals,color='red',label='DED with img_cross')
plt.plot(x_1,DED_L1loss_vals,color='blue',label='DED with L1 loss')
plt.plot(x_1,DED_vals,color='red',label='DED with L2 loss')'''
#plt.plot(x_1,Diff_vals,label='Single image vs. Cross image')
plt.xlabel('iteration',fontsize=15)
plt.ylabel('PSNR(dB) (Gaussian denoise)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 设置坐标标签字体大小
#ax.set_xlabel(..., fontsize=20)
#ax.set_ylabel(..., fontsize=20)
# 设置图例字体大小
#ax.legend(..., fontsize=20)

plt.legend(fontsize=15)
#plt.grid(axis='y')
