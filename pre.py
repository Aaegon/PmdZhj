from GC_binarization import Binariization
from wrapped_phase_filter import WrappedPhase
from Unwrapped_phase import Unwrappedphase
import numpy as np
import torch
import ast
import os
import math
import statistics
import cv2
import torchvision.transforms as transforms

'''
功能：调用PMD，有效区域提取
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_graycodes(self, T):
        datapath = self.datapath
        th = self.th
        B = Binariization(datapath, th=th, width= self.width, height=self.height)
        W = WrappedPhase(datapath, width=self.width, height=self.height)
        # 获取格雷码
        gc = B.get_GC_images()
        gc = torch.from_numpy(gc.astype(np.float32)).to(self.device)
        # 获取四步正弦计算光源的直流分量+环境光   
        I = W.getImageData()
        _, env_brightness = W.computeModulation(I)
        env_brightness_ = env_brightness/4*255
        # 用直流分量平替Ioff
        I_off = env_brightness_.unsqueeze(0).repeat(5, 1, 1)
        #计算置信度
        N = torch.abs(gc-I_off)/(gc+1e-6)
        confidence = torch.min(N, dim=0, keepdim=False)[0]
        mask = confidence > T
        result = torch.where(mask, torch.tensor(255, dtype=torch.uint8, device = self.device), 
                                   torch.tensor(0, dtype=torch.uint8, device = self.device))
        return result.cpu().numpy()
    
    def getRawWph(self):
        datapath = self.datapath
        W = WrappedPhase(datapath, width = self.width, height = self.height)
        I = W.getImageData()
        wph = W.computeWrappedphase(I)
        return wph
    
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
    def compute_phase_series_modulations_cuda(self):
        #计算折叠相位，调制度，绝对级数
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

        # 计算绝对级数
        series, series1 = U.gray_to_series(gc)
        series1 = series1.int().to(torch.uint8)
        abs_series = torch.zeros_like(series, dtype=torch.uint8).to(self.device)
        idx1 = wph <= (math.pi / 2)
        idx2 = (wph > (math.pi / 2)) & (wph < (3 * math.pi / 2))
        idx3 = wph >= (3 * math.pi / 2)
        abs_series[idx1] = series1[idx1]
        abs_series[idx2] = series[idx2]
        abs_series[idx3] = series1[idx3] - 1

        # 计算调制度  
        modulation, _ = W.computeModulation(I)
        return wph.cpu().numpy(), abs_series.cpu().numpy(), modulation.cpu().numpy()
    
    def get_raw_modulation(self):
        datapath = self.datapath
        W = WrappedPhase(datapath, width=self.width, height=self.height)
        I = W.getImageData()
        # 计算调制度  
        modulation, env_brightness = W.computeModulation(I)
        return modulation

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
        # 返回调制度和总光强
        return modulation.cpu().numpy(), env_brightness.cpu().numpy()    
## 调制度提取有效区域
def extra_region_through_binary(img):
    # 全局阈值，采用分位数确定阈值，腐蚀膨胀腐蚀，适用于modulation，比env_bri(总亮度)要好调参并且形态学操作更少
    threshold_value_0 = np.percentile(img, 30)
    threshold_value_255 = np.percentile(img, 30)
    idx_0 = img <= threshold_value_0
    idx_255 = img >= threshold_value_255
    img[idx_255] = 255
    img[idx_0] = 0
    # # 矩形核
    # kernel_15 = np.ones((15, 15), np.uint8)
    # kernel_20 = np.ones((20, 20), np.uint8)
    # kernel_5 = np.ones((5, 5), np.uint8)
    # 椭圆核
    kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_20 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    img = cv2.erode(img, kernel_15, iterations=2)
    img = cv2.dilate(img, kernel_20, iterations=2)
    return img

def extra_region_through_graycode(gc):
    pass

def get_modulations_from_datapath(datafolder, savefolder):
    # 从原始图像中提取modulation用于labelme标定
    all_items = os.listdir(datafolder)
    for item in all_items:
        full_path = os.path.join(datafolder, item)
        save_path = os.path.join(savefolder, item) + '.png'
        aa = PMD(datapath = full_path, th = [0.6, 1.9, 1.4, 1.2, 1.2])
        modulation, env_brightness = aa.compute_modulation()
        cv2.imwrite(save_path,modulation)
    print("deal down!")

def get_masks_from_datapath(datafolder, savefolder):
    # 从原始图像中提取mask
    all_items = os.listdir(datafolder)
    for item in all_items:
        full_path = os.path.join(datafolder, item)
        save_path = os.path.join(savefolder, item) + '.png'
        aa = PMD(datapath = full_path, th = [0.6, 1.9, 1.4, 1.2, 1.2])
        modulation, env_brightness = aa.compute_modulation()
        modulation = cv2.GaussianBlur(modulation, (5, 5), 1.5)

        mask = extra_region_through_binary(modulation)
        cv2.imwrite(save_path, mask)
        
    print("deal down!")
def visualize_demo(img, savepath):
    #简单的保存可视化验证

    result = (img-img.min())/(img.max()-img.min() + 1e-8) * 255
    result = result.astype(np.uint8)
    print(np.percentile(result, 50), np.percentile(result, 85))
    cv2.imwrite(savepath, img)

def cal_data_for_training(datafolder = None, savefolder = None, th_txt = 'th_2026v0.txt'):
    '''
    保存数据：
    折叠相位 wrapped_phase float32
    调制度   modulation    float32
    相位级数 series        int
    '''
    ths = {}
    with open(th_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            idx, list_str = line.split(maxsplit=1)  # 只分一次
            values = ast.literal_eval(list_str)      # 字符串转 Python list

            ths[idx] = values
    # aa = PMD(datapath= r'D:\zhj\code\datapath\data_final\17', th =  [5.0, 2.4, 1.9000000000000001, 3.7, 2.2])
    # wph, series, modulation = aa.compute_phase_series_modulations_cuda()
    files = os.listdir(datafolder)
    for file in files:
        th = ths[file]
        path = os.path.join(datafolder, file)
        savefile = os.path.join(savefolder, file) + '.npy'
        aa = PMD(path, th)
        wph, series, modulation = aa.compute_phase_series_modulations_cuda()
        wph = (wph + np.pi) % (2 * np.pi) - np.pi

        modulation = cv2.GaussianBlur(modulation, (5, 5), sigmaX=1.0)
        modulation = (modulation - modulation.min())/(modulation.max() - modulation.min() + 1e-8)
        modulation = np.clip(modulation, 0.1, 0.9)
        mod = (modulation - modulation.min())/(modulation.max() - modulation.min() + 1e-8)
        mod = np.clip(mod, 0.0, 1.0)
        data = {
            "wph": wph.astype(np.float32),
            "modulation": mod.astype(np.float32),
            "series": series.astype(np.int16)
        }

        np.save(savefile, data)

def augment_and_save_dict_patches(source_dir, save_dir, patch_size=256, stride=128, m_threshold=20):
    """
    数据裁剪、增强
    source_dir: 原始73个大字典.npy文件的路径
    save_dir: 增强后切片字典的保存路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取所有npy文件
    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    patch_count = 0

    for fname in npy_files:
        # 1. 读取原始大图字典
        data_dict = np.load(os.path.join(source_dir, fname), allow_pickle=True).item()
        
        # 提取各个组件 (请根据你实际的 Key 名匹配，如 'wph', 'series', 'modulation', 'fringes')
        wph_full = data_dict['wph']          # [H, W]
        series_full = data_dict['series']    # [H, W]
        mod_full = data_dict['modulation']  # [H, W]
        fringe_full = data_dict['fringes']    # [4, H, W]
        
        h, w = wph_full.shape

        # 2. 滑动窗口切片
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                
                # 判定调制度有效性
                mod_patch = mod_full[y:y+patch_size, x:x+patch_size]
                if np.mean(mod_patch) < m_threshold:
                    continue # 剔除掉调制度过低的背景区域块

                # 3. 构建新的切片字典
                patch_dict = {
                    'wph': wph_full[y:y+patch_size, x:x+patch_size].astype(np.float32),
                    'series': series_full[y:y+patch_size, x:x+patch_size].astype(np.int64),
                    'modulation': mod_patch.astype(np.float32),
                    'fringes': fringe_full[:, y:y+patch_size, x:x+patch_size].astype(np.float32)
                }

                # 4. 以字典格式保存为新的 .npy
                save_name = f"patch_{patch_count:05d}.npy"
                np.save(os.path.join(save_dir, save_name), patch_dict)
                patch_count += 1

    print(f"数据增强完成！共生成有效 Patch 字典: {patch_count} 个")

if __name__ == "__main__":
    # aa = PMD(datapath= r'D:\zhj\code\datapath\data_final\17', th =  [0.6, 1.9, 1.4, 1.2, 1.2])
    # result = aa.compute_phase_cuda()
    # modulation, env_brightness = aa.compute_modulation()
    # cv2.imwrite('output/abs_phase.png',result)
    # final = np.stack([result, modulation, env_brightness], axis = 2)
    # modulation = cv2.GaussianBlur(modulation, (5, 5), 1.5)
    # cv2.imwrite('output/modulation.png',modulation)
    # cv2.imwrite('output/env_brightness.png',env_brightness)

    # mask = extra_region_through_binary(modulation)
    # cv2.imwrite('output/mask.png',mask)

    # gc = aa.get_graycodes(0.12)
    # cv2.imwrite('output/gc_mask.png',gc)

    # final[mask==0] = 255
    # cv2.imwrite('output/result.png',final)
    # # get_modulations_from_datapath('D:\zhj\code\datapath\data_final', 'D:\zhj\code\datapath\modulations')
    # # get_masks_from_datapath('D:\zhj\code\datapath\data_final', 'D:\zhj\code\datapath\masks')
    cal_data_for_training(datafolder=r'D:\zhj\pmd\0418\data_final', savefolder= 'data', th_txt= 'th_2026v0.txt')


                         




    