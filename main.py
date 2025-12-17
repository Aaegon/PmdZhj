from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase
from Unwrapped_phase import Unwrappedphase
import numpy as np
import torch
import os
import math
import statistics
import cv2
import matplotlib.pyplot as plt

'''
功能: 生成绝对相位 绝对相位梯度 调制图
'''


class PMD():
    '''
    功能: 生成绝对相位 绝对相位梯度 调制图
    '''
    def __init__(self, datapath, th, width = 2432, height = 2048):
        self.datapath = datapath
        self.width = width
        self.height = height
        self.th = th

    def get_th(self, I):
        width, height = I.shape[1:]
        result = np.zeros((width, height),dtype=np.float32)
        for i in range(I.shape[0]):
            result += I[i]
        result = result/4
        return result.astype(np.uint8)
    
    def get_gradient(self, I):
        # 计算 Sobel 梯度
        sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
        sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

        # 计算梯度的幅值和方向
        magnitude = cv2.magnitude(sobelx, sobely)
        angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

        # 计算平均梯度
        return magnitude.astype(np.uint8), angle.astype(np.uint8)
    def normalize_img(self, I):
        img_normalized = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX)
        return img_normalized
    
    def getRawWph(self):
        datapath = self.datapath
        W = WrappedPhase(datapath, width = self.width, height = self.height)
        I = W.getImageData()
        wph = W.computeWrappedphase(I)
        return wph
    
    def getWph(self):
        datapath = self.datapath
        W = WrappedPhase(datapath, width = self.width, height = self.height)
        I = W.getImageData()
        wph = W.computeWrappedphase(I)
        phaScaled = wph * 255 / (2 * math.pi)  # 将pha转换到为图像灰度尺度
        phaScaled1 = phaScaled.cpu().numpy().astype(np.uint8)
        return phaScaled1
    def getWph_MD(self):
        datapath = self.datapath
        W = WrappedPhase(datapath, width = self.width, height = self.height)
        I = W.getImageData()
        wph_M, wph_D = W.computeWrappedphase_M_D(I)
        return wph_M.cpu().numpy(), wph_D.cpu().numpy()

    
    def getSavePath(self, outPath, saveType, save_middle_type=""):
        datapath = self.datapath
        normalizedPath = os.path.normpath(datapath)
        lastName = os.path.basename(normalizedPath)
        savePath = os.path.join(outPath,lastName)+save_middle_type+saveType
        return savePath
    def saveSingleMD(self, outPath):
        savePathM = self.getSavePath(outPath, '.npy','M')
        savePathD = self.getSavePath(outPath, '.npy','D')
        M,D = self.getWph_MD()
        np.save(savePathM, M)
        np.save(savePathD, D)
    
    def saveSingleWph(self, outPath):
        savePath = self.getSavePath(outPath, '.png')
        wph = self.getWph()
        cv2.imwrite(savePath, wph)
    
    def compute_phase_cuda(self):
        datapath = self.datapath
        # th = self.th

        W = WrappedPhase(datapath, width=self.width, height=self.height)
        # B = Binariization(datapath, th=th, width= self.width, height=self.height)
        # U = Unwrappedphase(datapath, row=self.height, col= self.width)

        # 计算折叠相位
        I = W.getImageData()
        wph = W.computeWrappedphase(I)

        # # 格雷码二值化
        # gc = B.get_Binary_wph(10)

        # # 计算绝对相位
        # series, series1 = U.gray_to_series(gc)
        # absphase = U.get_absphase(series, series1, wph)
        # absphase_scale = ((absphase * 255) / (2 ** U.n * np.pi)).to(torch.uint8)  # 映射到灰度值
        # absphase_scale = absphase_scale.cpu().numpy()
        # th = self.get_th(I)
        # gratitude, angle = self.get_gradient(absphase_scale)
        return wph
    
def saveMultipleWph(sourceFolder, outputFolder, th):
    for fileName in os.listdir(sourceFolder):
        absPath = os.path.join(sourceFolder, fileName)
        aa = PMD(absPath, th)
        aa.saveSingleWph(outputFolder)

def saveMultipleMD(sourceFolder, outputFolder, th):
    for fileName in os.listdir(sourceFolder):
        absPath = os.path.join(sourceFolder, fileName)
        aa = PMD(absPath, th)
        aa.saveSingleMD(outputFolder)

def compute_and_save_gradient(image_path, save_prefix='output'):
    # 1. 读取灰度图像（0~255）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # 2. 计算梯度（使用 np.diff，差分结果会减小一列/一行）
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
    dy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 


    mag = cv2.magnitude(dx, dy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    mag = 255-mag

    cv2.imwrite(f"{save_prefix}_gradient.png", mag)

if __name__ == "__main__":
    # aa = PMD(datapath= '/home/zhj/zhj/pmd/datapath/data_in_dtdp/103', th = [0.2, 4.1, 2.0, 1.3, 1.8])
    # phase, th, gratitude, angle = aa.compute_phase_cuda()
    # cv2.imwrite('output/abs_phase.png',phase)
    # cv2.imwrite('output/th.png',th)
    # cv2.imwrite('output/gratitude.png',gratitude)
    # cv2.imwrite('output/gratitude_angle.png',angle)

    # img_stack = np.stack([phase, th, gratitude], axis=0)
    # # 创建一个空的 RGB 图像数组 (2448, 2048, 3)
    # rgb_img = np.zeros((phase.shape[0], phase.shape[1], 3), dtype=np.uint8)
    # # 为每个图层指定颜色通道
    # rgb_img[..., 0] = img_stack[0]  # 将第一个灰度图像放到红色通道
    # rgb_img[..., 1] = img_stack[2]  # 将第二个灰度图像放到绿色通道
    # rgb_img[..., 2] = img_stack[1]  # 将第三个灰度图像放到蓝色通道
    # cv2.imwrite('output/result.png', rgb_img)

    # coeffs2 = pywt.dwt2(phase, 'haar')
    # LL, (LH, HL, HH) = coeffs2
    # print(HH.shape)
    # th_nolmal = aa.normalize_img(th)
    # cv2.imwrite('output/normalized_th.png', th_nolmal)

    # saveMultipleWph(sourceFolder= '/home/zhj/zhj/pmd/datapath/data_final/', outputFolder = 'output/wph', th = [0.2, 4.1, 2.0, 1.3, 1.8])
    # saveMultipleMD(sourceFolder= '/home/zhj/zhj/pmd/datapath/data_final/', outputFolder = 'output/wph', th = [0.2, 4.1, 2.0, 1.3, 1.8])
    compute_and_save_gradient('output/test_unwrapping/use_least_squares_unwrapped17.png')