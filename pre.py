from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase
from Unwrapped_phase import Unwrappedphase
import numpy as np
import torch
import os
import math
import statistics
import cv2
import torchvision.transforms as transforms

'''
功能：调用PMD，有效区域提取，绝对相位效果评价
'''


class PMD():
    '''
    调用PMD，返回PMD之后的绝对相位图像
    '''
    def __init__(self, datapath, th, width = 2432, height = 2048):
        self.datapath = datapath
        self.width = width
        self.height = height
        self.th = th
    
    def get_graycodes(self):
        datapath = self.datapath
        th = self.th
        B = Binariization(datapath, th=th, width= self.width, height=self.height)
        W = WrappedPhase(datapath, width=self.width, height=self.height)

        gc = B.get_GC_images()     
        I = W.getImageData()

        # 计算调制度  
        _, env_brightness = W.computeModulation(I)
        I_off = env_brightness/4
        




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
    
    def compute_modulation(self):
        datapath = self.datapath
        W = WrappedPhase(datapath, width=self.width, height=self.height)
        I = W.getImageData()

        # 计算调制度  
        modulation, env_brightness = W.computeModulation(I)
        # 去掉灰度极高和极低的噪点
        modulation = torch.clip(modulation, min = torch.quantile(modulation, 0.1), max = torch.quantile(modulation, 0.9))
        env_brightness = torch.clip(env_brightness, min = torch.quantile(env_brightness, 0.1), max = torch.quantile(env_brightness, 0.9))
        # 归一化
        modulation = (modulation-modulation.min())/(modulation.max() - modulation.min())*255
        modulation = modulation.to(torch.uint8)
        env_brightness = (env_brightness-env_brightness.min())/(env_brightness.max() - env_brightness.min())*255
        env_brightness = env_brightness.to(torch.uint8)

        
        return modulation.cpu().numpy(), env_brightness.cpu().numpy()    

def extra_region_through_binary(img):
    # 全局阈值，采用分位数确定阈值，腐蚀膨胀腐蚀，适用于modulation，比env_bri(总亮度)要好调参并且形态学操作更少
    threshold_value_0 = np.percentile(img, 30)
    threshold_value_255 = np.percentile(img, 30)
    idx_0 = img <= threshold_value_0
    idx_255 = img >= threshold_value_255
    img[idx_255] = 255
    img[idx_0] = 0
    kernel_15 = np.ones((15, 15), np.uint8)
    kernel_20 = np.ones((20, 20), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)

    img = cv2.erode(img, kernel_15, iterations=2)
    img = cv2.dilate(img, kernel_20, iterations=2)
    # kernel = np.ones((15, 15), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    return img

def extra_region_through_graycode(gc):
    pass


if __name__ == "__main__":
    aa = PMD(datapath= r'D:\zhj\code\datapath\data_final\13', th = [0.6, 1.9, 1.4, 1.2, 1.2])
    # result = aa.compute_phase_cuda()
    # modulation, env_brightness = aa.compute_modulation()
    # # cv2.imwrite('output/abs_phase.png',result)

    # modulation = cv2.GaussianBlur(modulation, (5, 5), 1.5)
    # cv2.imwrite('output/modulation.png',modulation)
    # cv2.imwrite('output/env_brightness.png',env_brightness)

    # mask = extra_region_through_binary(modulation)
    # cv2.imwrite('output/mask.png',mask)

    gc = aa.get_graycodes()
    gc[0] = gc[0]|gc[1]|gc[2]|gc[3]|gc[4]
    cv2.imwrite('output/gc_mask.png',gc[0])







    