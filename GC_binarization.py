import glob
import os
import time
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from wrapped_phase_filter import WrappedPhase 


'''
功能：二值化格雷码
重点在于阈值的计算，目前采用四幅图像求各个点阈值的方法

'''

class Binariization():
    '''对格雷码图像进行二值化
    
    get_GC_images:读取格雷码图像,J_array.shape(4,2048,2448)
    get_Binary_adaptive:自适应阈值法二值化
    
    Attributes:
        datapath:存储格雷码图像的路径
        edge_length:相机采集照片的大致边长，一般不用动
        n:n副格雷码图像
        
    '''
    def __init__(self, datapath, th, width = 2432, height = 2048,edge_length:int=2000):
        self.th1, self.th2, self.th3, self.th4, self.th5 = th
        self.datapath = datapath
        self.width = width
        self.height = height
        self.edge_length = edge_length
        self.n = self.get_GC_nums()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cuda")
        #self.device = torch.device("cpu")
        
    def get_GC_nums(self):
        GC_files = glob.glob(os.path.join(self.datapath, 'gc*'))
        # 获取匹配文件的数量
        number_of_GC = len(GC_files)
        return number_of_GC
        
    def get_GC_images(self):
        '''读取相机采集的格雷码图片
            将相机采集的格雷码照片命名为GC0,GC1,GC2,GC3,GC4
        
        Args
            filename:相机采集格雷码图像所在的文件夹，filename = r"D:\Project\PMD\abs_phase\gray_binarization"
        
        return:
            相机采集格雷图像的N维数组,J_array.shape(4,2048,2448)
        
        '''
        J_array = np.empty((self.n, self.height, self.width), dtype=np.uint8)  # 预分配数组

        for i in range(self.n):
            filename = os.path.join(self.datapath, f"gc{i}.png")
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            #img = cv.GaussianBlur(img, (3,3), 1)
            J_array[i] = img

        return J_array  


    def get_threshold(self, m: int = 4):
        '''利用四幅相移图计算阈值'''
        wp = WrappedPhase(self.datapath)
        I = wp.getImageData(m)
        # for i in range(4):
        #     I[i] = cv.GaussianBlur(I[i], (3,3), 1)
            
        I_th = torch.from_numpy(I.astype(np.float32)).to(self.device)
        
        I_th = torch.mean(I_th, dim=0).round().to(torch.uint8)
        
        #cv.imwrite('./output/TH_wph.png', I_th.cpu().numpy())  
        return I_th     #tensor,cuda:0

    def get_Binary_wph(self, offset:int=20):
        '''利用四幅相移图求阈值，将格雷码图像二值化处理
        Args:增加了一个补偿offset，该值越小，阈值越小，图像越白
        return:四幅二值化后格雷码图像，tensor,cuda:0
        '''
        
        threshold = self.get_threshold()        #threshold.device：cuda:0
        
        graycodes = self.get_GC_images()        #ndarray,(5,2048,2448)
        graycodes = torch.from_numpy(graycodes).to(self.device)

        k = threshold / (threshold.max() + 1e-6)
        offset *= k
        
        graycodes[0][graycodes[0] <= (threshold + self.th1*offset)] = 0
        graycodes[0][graycodes[0] > (threshold + self.th1*offset)] = 255

        graycodes[1][graycodes[1] <= (threshold + self.th2*offset)] = 0
        graycodes[1][graycodes[1] > (threshold + self.th2*offset)] = 255

        graycodes[2][graycodes[2] <= (threshold + self.th3*offset)] = 0
        graycodes[2][graycodes[2] > (threshold + self.th3*offset)] = 255

        graycodes[3][graycodes[3] <= (threshold + self.th4*offset)] = 0
        graycodes[3][graycodes[3] > (threshold + self.th4*offset)] = 255

        graycodes[4][graycodes[4] <= (threshold + self.th5*offset)] = 0
        graycodes[4][graycodes[4] > (threshold + self.th5*offset)] = 255
        


        return graycodes

    
if __name__ == "__main__":
    t0 = time.time()
    
    datapath = r"D:\Project\PMD\demo\2023_final\PMD3.2\pos22-26\pos25"
    bgc = Binariization(datapath)
    #gc = bgc.get_Binary_gc(0)        #gc->cuda:0
    gc = bgc.get_Binary_wph(12)
    
    gc=gc.cpu().numpy()
    

    
    for u in range(bgc.n):
        #gc[u].astype(np.uint8)
        cv.imwrite('.\output' + '\Binarized_GC-' + str(u) + ".png",gc[u])


