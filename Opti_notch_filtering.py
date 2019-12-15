import cv2
import numpy as np

img = cv2.imread('./Fig0526(a)(original_DIP).tif') #输入图像
h = img.shape[0]                          #得到图像尺寸
w = img.shape[1]
c = img.shape[2]
M_1 = 20
M_2 = 30
D0 = 5                                  #滤波器参数
n = 4                                   #滤波器阶数
img_double = np.asarray(img,dtype=float)
img_P = np.asarray(img,dtype=float)
img_noise = np.zeros([h,w,c],dtype=float)
img_noise_show = np.zeros([h,w,c],dtype=float)
noise_G = np.random.normal(0,30,[h,w])     #生成高斯噪声
for i in range(h):
    for j in range(w):
        img_P[i,j,:] = img_P[i,j,:] + 40*np.sin(M_1*i) + 40*np.sin(M_2*j)    #添加周期正弦噪声

for t in range(c):
    img_noise[:,:,t] = img_P[:,:,t] + noise_G                       #添加高斯噪声

img_noise_show = np.where(img_noise < 0, 0, np.where(img_noise > 255, 255, img_noise))  #为了显示，将数据范围clip为[0,255]

Freq = np.zeros([h,w,c],dtype=complex)
Freq_show = np.zeros([h,w,c],dtype=complex)
Freq_noise = np.zeros([h,w,c],dtype=complex)
Freq_noise_show = np.zeros([h,w,c],dtype=complex)

for i in range(c):
    Freq[:,:,i] = np.fft.fft2(img_double[:,:,i])
    Freq[:,:,i] = np.fft.fftshift(Freq[:,:,i])                #计算原图的频谱
    Freq_noise[:,:,i] = np.fft.fft2(img_noise[:,:,i])
    Freq_noise[:,:,i] = np.fft.fftshift(Freq_noise[:,:,i])    #计算受干扰图像的频谱

Freq_show = np.log(1 + abs(Freq))
Freq_noise_show = np.log(1 + abs(Freq_noise))

for i in range(img_noise.shape[2]):
    Freq_noise_show[:,:,i] = 255*Freq_noise_show[:,:,i]/(Freq_noise_show[:,:,i].max())   #进行归一化
    Freq_show[:, :, i] = 255*Freq_show[:, :, i]/(Freq_show[:, :, i].max())

Freq_distance = abs(Freq-Freq_noise)/(h*w)                 #根据频谱之间的差异寻找干扰模式的频率中心点

H_pr = np.zeros([h,w,c],dtype=float)                       #构造陷波滤波
D_1_pair = np.zeros([h,w],dtype=float)
D_2_pair = np.zeros([h,w],dtype=float)

F_d_x = 126                                                #寻找到的频率中心点1
F_d_y = 155                                                #寻找到的频率中心点2

for i in range(h):                                         #为了避免除0错误，分母上加上一个很小的值1e-6
    for j in range(w):
        D_1_pair[i,j] = (1 / (1 + (D0 / (1e-6 + np.sqrt((i-h/2-F_d_x)**2 + (j-w/2)**2)))**(2*n))) * (1 / (1 + (D0 / (1e-6 + np.sqrt((i-h/2+F_d_x)**2 + (j-w/2)**2)))**(2*n)))
        D_2_pair[i,j] = (1 / (1 + (D0 / (1e-6 + np.sqrt((i-h/2)**2 + (j-w/2-F_d_y)**2)))**(2*n))) * (1 / (1 + (D0 / (1e-6 + np.sqrt((i-h/2)**2 + (j-w/2+F_d_y)**2)))**(2*n)))

H_pr = 1 - D_1_pair * D_2_pair                             #得到陷波滤波器

N = np.zeros([h,w,c],dtype=complex)
n = np.zeros([h,w,c],dtype=complex)
n_show = np.zeros([h,w,c],dtype=float)

for i in range(c):
    N[:,:,i] = H_pr * Freq_noise[:,:,i]                    #进行频域点乘（滤波）
    n[:,:,i] = np.fft.ifft2(N[:,:,i])

n = np.real(n)
for i in range(h):
    for j in range(w):
        n[i,j,:] = (-1)**(i+j)*n[i,j,:]                    #得到空域上的干扰模式

for i in range(c):
    n_show[:,:,i] = 255*n[:,:,i]/(n[:,:,i].max())          #进行显示

weight = np.zeros([h,w,c],dtype=float)                     #计算权重矩阵

for i in range(0,h,16):                                    #以16x16为邻域大小计算w(x,y)
    for j in range(0,w,16):
        A = np.mean(img_noise[i:i+16,j:j+16,:]*n[i:i+16,j:j+16,:])
        B = np.mean(img_noise[i:i+16,j:j+16,:])*np.mean(n[i:i+16,j:j+16,:])
        C = np.mean(n[i:i+16,j:j+16,:]*n[i:i+16,j:j+16,:])
        D = np.mean(n[i:i+16,j:j+16,:])*np.mean(n[i:i+16,j:j+16,:])
        weight[i:i+16,j:j+16,:] = (A-B)/(C-D)

f_restore = img_noise - n*weight                           #得到干净图像的估计值

cv2.imwrite('./img_restore.png',f_restore)                #输出干净图像的估计值
cv2.imwrite('./noise_mode.png',n_show)                    #输出干扰模式
cv2.imwrite('./noise_P.png',img_noise)                    #输出受干扰的图像
cv2.imwrite('./Freq_distance.png', 255*Freq_distance/(Freq_distance.max()))  #输出频谱差异图
cv2.imwrite('./H_np.png', 255*H_pr/(H_pr.max()))          #输出滤波器图
