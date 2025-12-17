from skimage.restoration import unwrap_phase
import cv2
import numpy as np
import math
from main import PMD
import pyfftw
from scipy.fft import dctn, idctn
from scipy.fft import fft2, ifft2
from scipy.ndimage import sobel

def use_Skimage(wph):
    # 使用质量图（如相位导数方差）引导展开
    unwrapped = unwrap_phase(wph)+math.pi
    result = unwrapped * 255 / (2 * math.pi)
    cv2.imwrite('output/test_unwrapping/use_skimage.png', result.astype(np.uint8))

def wrap_2pi_to_pi(phase_0_2pi):
    """将 [0, 2π] 映射到 [-π, π]"""
    return (phase_0_2pi + np.pi) % (2 * np.pi) - np.pi

# def unwrap_phase_lse(wrapped_phase_pi):
#     """最小二乘法相位展开（频域泊松解）"""
#     h, w = wrapped_phase_pi.shape

#     # 计算梯度并 wrap 回 [-π, π]
#     dx = wrap_2pi_to_pi(np.diff(wrapped_phase_pi, axis=1))
#     dy = wrap_2pi_to_pi(np.diff(wrapped_phase_pi, axis=0))

    # dx_padded = np.zeros_like(wrapped_phase_pi)
    # dy_padded = np.zeros_like(wrapped_phase_pi)
    # dx_padded[:, :-1] = dx
    # dy_padded[:-1, :] = dy

    # # 计算散度
    # fx = np.zeros_like(wrapped_phase_pi)
    # fy = np.zeros_like(wrapped_phase_pi)
    # fx[:, 0] = dx_padded[:, 0]
    # fx[:, 1:] = dx_padded[:, 1:] - dx_padded[:, :-1]
    # fy[0, :] = dy_padded[0, :]
    # fy[1:, :] = dy_padded[1:, :] - dy_padded[:-1, :]
    # f = fx + fy

    # # 频域解泊松方程 Δφ = f
    # yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # denom = 2 * (np.cos(2 * np.pi * xx / w) + np.cos(2 * np.pi * yy / h) - 2)
    # denom[0, 0] = 1  # 避免除以 0
    # f_fft = fft2(f)
    # unwrapped = np.real(ifft2(f_fft / denom))

    # return unwrapped

# def phase_unwrap_image(input_path, output_path):
#     # Step 1: 读取灰度图像
#     gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     gray_img = gray_img.astype(np.float32)

#     # Step 2: 映射到 [0, 2π]
#     phase_0_2pi = gray_img / 255.0 * 2 * np.pi

#     # Step 3: 转换为 [-π, π]
#     wrapped_phase = wrap_2pi_to_pi(phase_0_2pi)

#     # Step 4: 相位展开
#     unwrapped_phase = unwrap_phase_lse(wrapped_phase)

#     # Step 5: 归一化为 [0, 255]
#     unwrapped_norm = unwrapped_phase - unwrapped_phase.min()
#     unwrapped_norm = 255.0 * (unwrapped_norm / unwrapped_norm.max())
#     unwrapped_uint8 = unwrapped_norm.astype(np.uint8)

#     # Step 6: 保存图像
#     cv2.imwrite(output_path, unwrapped_uint8)
#     print(f"Unwrapped phase image saved to: {output_path}")

def unwrap_phase_lse_(wrapped_phase_pi):
    """最小二乘法相位展开（频域泊松解）"""
    h, w = wrapped_phase_pi.shape
    
    dx = wrap_2pi_to_pi(np.diff(wrapped_phase_pi, axis=1))
    dy = wrap_2pi_to_pi(np.diff(wrapped_phase_pi, axis=0))

    dx_padded = np.zeros_like(wrapped_phase_pi)
    dy_padded = np.zeros_like(wrapped_phase_pi)
    dx_padded[:, :-1] = dx
    dy_padded[:-1, :] = dy

    # 计算散度
    fx = np.zeros_like(wrapped_phase_pi)
    fy = np.zeros_like(wrapped_phase_pi)
    fx[:, 0] = dx_padded[:, 0]
    fx[:, 1:] = dx_padded[:, 1:] - dx_padded[:, :-1]
    fy[0, :] = dy_padded[0, :]
    fy[1:, :] = dy_padded[1:, :] - dy_padded[:-1, :]
    f = fx + fy

    # 频域解泊松方程 Δφ = f
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    denom = 2 * (np.cos(2 * np.pi * xx / w) + np.cos(2 * np.pi * yy / h) - 2)
    denom[0, 0] = 1  # 避免除以 0
    f_fft = fft2(f)
    unwrapped = np.real(ifft2(f_fft / denom))

    # Step 5: 归一化为 [0, 255]
    unwrapped_norm = unwrapped - unwrapped.min()
    unwrapped_norm = 255.0 * (unwrapped_norm / unwrapped_norm.max())
    unwrapped_uint8 = unwrapped_norm.astype(np.uint8)

    # Step 6: 保存图像
    cv2.imwrite('output/test_unwrapping/use_least_squares_unwrapped.png', unwrapped_uint8)

# === 使用示例 ===
if __name__ == "__main__":
    img = cv2.imread('output/wph/00.png')
    aa = PMD(datapath= '/home/zhj/zhj/pmd/datapath/data_final/17', th = [0.2, 4.1, 2.0, 1.3, 1.8])
    wph = aa.getRawWph().cpu().numpy()
    print(np.min(wph), np.max(wph))
    unwrap_phase_lse_(wph)



