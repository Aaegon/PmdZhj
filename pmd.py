from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase
from Unwrapped_phase import Unwrappedphase
import numpy as np
import torch
import os
import math
import statistics
import cv2

'''
功能：调用PMD，有效区域提取，绝对相位效果评价
'''


class PMD():
    '''
    调用PMD，返回PMD之后的绝对相位图像
    '''
    def __init__(self, datapath, th, width = 2448, height = 2048):
        self.datapath = datapath
        self.width = width
        self.height = height
        self.th = th

    def compute_phase_cuda(self):
        datapath = self.datapath
        th = self.th

        W = WrappedPhase(datapath, width=self.width, height=self.height)
        B = Binariization(datapath, th=th, width= self.width, height=self.height)
        U = Unwrappedphase(datapath, row=self.height, col= self.width)

        # 计算折叠相位
        I = W.getImageData()
        wph = W.computeWrappedphase(I)

        # 格雷码二值化
        gc = B.get_Binary_wph(10)

        # 计算绝对相位
        series, series1 = U.gray_to_series(gc)
        absphase = U.get_absphase(series, series1, wph)
        absphase_scale = ((absphase * 255) / (2 ** U.n * np.pi)).to(torch.uint8)  # 映射到灰度值
        return absphase_scale.cpu().numpy()
class Region_Extraction():
    '''
    有效区域提取
    area_cal()返回有效区域和无效区域
    '''
    def __init__(self, datapath, width = 2448, height = 2048):
        self.datapath = datapath
        self.width = width
        self.height = height

    def getImageData(self, datapath):
        I = np.empty((4, self.height, self.width), dtype=np.uint8)  # 预分配数组
        for i in range(4):
            filename = os.path.join(datapath, f"sin{i}.png")
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            I[i] = img
        return I
    #深度为8位的图像之间计算灰度差值，直接减会溢出
    def np8_cal(self, i0, i1):
        i01 = np.empty(i0.shape, dtype=np.uint8)
        idx_compare0 = i0 < i1
        idx_compare1 = i0 >= i1
        i01[idx_compare0] = i1[idx_compare0] - i0[idx_compare0]
        i01[idx_compare1] = i0[idx_compare1] - i1[idx_compare1]
        return i01

    def area_cal(self):
        I = self.getImageData(self.datapath)
        i01 = self.np8_cal(I[0], I[1])
        i12 = self.np8_cal(I[1], I[2])
        i23 = self.np8_cal(I[2], I[3])
        i30 = self.np8_cal(I[3], I[0])

        idx_compare = i01 < i12
        i01[idx_compare] = i12[idx_compare]
        idx_compare = i01 < i23
        i01[idx_compare] = i23[idx_compare]
        idx_compare = i01 < i30
        i01[idx_compare] = i30[idx_compare]

        # cv.imshow('mat', i01)
        # k = cv2.waitKey()
        # 查看直方图，可以看到两波峰之间的位置大约为30-50，如果采用全局阈值可以选用30-50
        # hist, bins = np.histogram(i01.flatten(), bins=256)
        # plt.bar(range(len(hist)), hist)
        # plt.show()

        # 全局阈值，采用分位数确定阈值，腐蚀膨胀再腐蚀
        threshold_value_0 = np.percentile(i01, 40)
        threshold_value_255 = np.percentile(i01, 40)

        idx_compare = i01 < threshold_value_0
        i01[idx_compare] = 0
        idx_compare = i01 >= threshold_value_255
        i01[idx_compare] = 255
        kernel = np.ones((6, 6), np.uint8)
        i01 = cv2.erode(i01, kernel, iterations=1)
        kernel = np.ones((6, 6), np.uint8)
        i01 = cv2.dilate(i01, kernel, iterations=10)
        kernel = np.ones((10, 10), np.uint8)
        i01 = cv2.erode(i01, kernel, iterations=10)

        idx_phase_val = (i01 == 255)
        idx_phase_unval = (i01 == 0)

        return idx_phase_val,idx_phase_unval
class Image_Evaluation():
    '''
    绝对相位图像评价
    这里评价直方图方差和平均梯度
    效果比较好的直方图方差通常在10**8-10**9之间，这边用log函数对它进行一个粗略的归一化处理
    forward()输出两个结果
    '''
    def __init__(self, datapath, th, width = 2448, height = 2048):
        self.datapath = datapath
        self.th = th
        self.width = width
        self.height = height

    def PMD_Init(self):
        pmd = PMD(datapath= self.datapath,th = self.th, width= self.width, height= self.height)
        reg = Region_Extraction(datapath=self.datapath, width= self.width, height= self.height)
        I = pmd.compute_phase_cuda()
        idx,idx0 = reg.area_cal()
        return I,idx

    def cal_variance(self, I):
        hist, bins = np.histogram(I.flatten(), bins=32)
        variance = statistics.variance(hist)
        result = math.log(variance, 10)/8
        return result**4
    def cal_gradient(self, I):
        # 计算 Sobel 梯度
        sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
        sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 计算平均梯度
        average_gradient = np.mean(gradient_magnitude)
        return average_gradient

    def forward(self):
        I,idx = self.PMD_Init()
        return self.cal_gradient(I[idx]),self.cal_variance(I[idx]),I
if __name__ == "__main__":
    aa = PMD(datapath= '/home/zhj/zhj/pmd/datapath/data_in_dtdp/103', th = [0.2, 4.1, 2.0, 1.3, 1.8])
    result = aa.compute_phase_cuda()
    cv2.imwrite('output/abs_phase.png',result)