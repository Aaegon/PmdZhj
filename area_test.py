from pmd import Region_Extraction
import numpy as np
import cv2
'''
功能：测试有效区域是否合理
有效区域提取用的传统的形态学操作，四步相移之后通过膨胀腐蚀提取有效区域，鲁棒性很差，新的环境需要重新调

'''

if __name__ == "__main__":
    aa = Region_Extraction(r'D:\datapath\20240418\data_final\00', 2432, 2048)
    I = np.empty((1,2048,2432), dtype=np.uint8)
    mm,nn = aa.area_cal()
    I[0][mm] = 255
    I[0][nn] = 0
    cv2.imshow('mat', I[0])
    k = cv2.waitKey(20000)