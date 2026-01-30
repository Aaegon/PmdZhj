import numpy as np
import cv2
import os
from multiprocessing import Pool

def generate_and_save_single(args):
    """
    单个样本生成函数，供进程池调用
    args: (idx, save_dir, patch_size, period, burr_strength)
    """
    idx, save_dir, size, period, burr_strength = args
    
    # --- 1. 物理模型构建 ---
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    X, Y = np.meshgrid(x, y)

    # 单调背景 (上到下递增) + 随机低频波动
    y_grad = np.random.uniform(0.7, 0.9)
    low_freq = cv2.resize(np.random.normal(0, 1, (4, 4)), (size, size), interpolation=cv2.INTER_CUBIC)
    abs_phase_ideal = (y_grad * Y + low_freq * 3) * (2 * np.pi / period)
    abs_phase_ideal += np.random.uniform(0, 50) 

    # 局部缺陷模拟 (凹陷/划痕)
    defect = np.zeros_like(X)
    if np.random.rand() > 0.2: # 80% 的样本包含缺陷
        cx, cy = np.random.uniform(size//4, 3*size//4, 2)
        sigma = np.random.uniform(1.5, 5)
        amp = np.random.uniform(0.3, 1.5)
        defect = amp * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))
    
    abs_phase_physical = abs_phase_ideal + (defect * (2 * np.pi / period))

    # --- 2. 互相关毛刺噪声生成 ---
    # 模拟传感器不确定性导致的共模噪声
    raw_noise = np.random.normal(0, 0.1 * burr_strength, (size, size))
    smooth_noise = cv2.GaussianBlur(raw_noise, (3, 3), 0)
    abs_phase_observed = abs_phase_physical + smooth_noise

    # 生成带毛刺的级数 k 和 折叠相位 wph
    k_series = np.floor(abs_phase_observed / (2 * np.pi)).astype(np.int64)
    wph = np.mod(abs_phase_observed, 2 * np.pi)

    # --- 3. 条纹图与调制度 ---
    A_light = np.random.uniform(100, 140)
    B_mod = np.random.uniform(70, 100)
    # 局部调制度衰减 (模拟漆面反光不均或缺陷处的对比度下降)
    mod_map = B_mod * (1 - 0.2 * defect / np.max(defect + 1e-6)) 
    
    fringes = []
    for i in range(4):
        p_noise = np.random.normal(0, 1.5, (size, size))
        I = A_light + mod_map * np.cos(wph - i * np.pi/2) + p_noise
        fringes.append(np.clip(I, 0, 255).astype(np.uint8))
    
    # --- 4. 封装字典并保存 ---
    # 归一化处理：wph映射到 [0, 1], modulation映射到 [0, 1]
    sample_dict = {
        'fringes': np.stack(fringes, axis=0).astype(np.float32) / 255.0,
        'wph': (wph / (2 * np.pi)).astype(np.float32),
        'series': k_series, # 注意：级数保持原始整数
        'modulation': (mod_map / 255.0).astype(np.float32),
        'abs_phase_gt': abs_phase_physical.astype(np.float32)
    }
    save_path = os.path.join(save_dir, f"sim_{idx:06d}.npy")
    np.save(save_path, sample_dict)

def batch_generate_multiprocess(total_count, save_dir, num_workers=8):
    """
    多进程批量生成
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"开始生成 {total_count} 个仿真样本，使用 {num_workers} 个核心...")
    
    # 准备任务参数列表
    tasks = [(i, save_dir, 256, 32, 2.0) for i in range(total_count)]
    
    with Pool(num_workers) as p:
        p.map(generate_and_save_single, tasks)
        
    print("生成任务全部完成！")

if __name__ == "__main__":
    # 建议先生成 10000 组用于阶段 A 预训练
    batch_generate_multiprocess(total_count=100, save_dir='./sim_dataset_A', num_workers=os.cpu_count())