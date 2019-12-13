# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:06:36 2019

@author: ZuoYS
"""

import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(19680801)
DD_list = np.array([29.03,28.97,29.09,29.04,28.37,28.52,29.25,28.98,29.17,29.62])
DIP_list = np.array([29.21,29.22,29.13,29.37,29.16,29.41,29.31,29.40,29.20,29.09])
DED_list = np.array([29.70,29.60,29.51,29.40,29.83,29.66,29.39,29.67,29.38,29.58])

Maxpooing_Bicubic_list = np.array([29.70,29.60,29.51,29.40,29.83,29.66,29.39,29.67,29.38,29.58])
Maxpooing_Bilinear_list = np.array([29.35,29.44,29.52,29.65,29.38,29.58,29.51,29.65,29.84,29.60])
Avgpooling_Bicubic_list = np.array([29.61,29.50,29.51,29.45,29.43,29.78,29.38,29.82,29.21,29.57])
Avgpooling_Bilinear_list = np.array([29.15,29.34,28.92,29.04,29.00,29.28,29.09,29.25,29.13,29.10])

Maxpooling_Bicubic_mean = np.mean(Maxpooing_Bicubic_list)  #求均值，方差
Maxpooling_Bicubic_var = np.var(Maxpooing_Bicubic_list)
Maxpooling_Bilinear_mean = np.mean(Maxpooing_Bilinear_list)
Maxpooling_Bilinear_var = np.var(Maxpooing_Bilinear_list)
Avgpooling_Bicubic_mean = np.mean(Avgpooling_Bicubic_list)
Avgpooling_Bicubic_var = np.var(Avgpooling_Bicubic_list)
Avgpooling_Bilinear_mean = np.mean(Avgpooling_Bilinear_list)
Avgpooling_Bilinear_var = np.var(Avgpooling_Bilinear_list)

DD_list_mean = np.mean(DD_list)
DIP_list_mean = np.mean(DIP_list)
DED_list_maen = np.mean(DED_list)


all_data = [Avgpooling_Bilinear_list, Avgpooling_Bicubic_list, Maxpooing_Bilinear_list, Maxpooing_Bicubic_list]
#all_data = [DD_list, DIP_list, DED_list]

labels = ['Avg_Bil', 'Avg_Bic', 'Max_Bil', 'Max_Bic']
#labels = ['Deep decoder', 'DIP', 'DED']
boxprops = dict(linewidth=1.2)  #设置线宽

bplot = plt.boxplot(all_data, patch_artist=True, labels=labels, whis=[0, 100], boxprops=boxprops)  # 设置箱型图可填充
plt.title('Different up/down-sampling methods in single image')
#plt.title('Differenr methods in single image')

colors = ['pink', 'lightblue', 'yellow', 'lightgreen']
#colors = ['pink', 'lightblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)  # 为不同的箱型图填充不同的颜色

plt.grid(axis="y")
plt.xlabel('Method')
plt.ylabel('PSNR(dB) (Gaussian denoise)')
plt.show()