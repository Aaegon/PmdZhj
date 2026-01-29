import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ==========================================
# 1. 单个样本生成函数
# ==========================================
def generate_sample(img_size=256, period=None, global_k_shift=0):
    """
    生成单组高质量相位数据
    :param global_k_shift: 全局级数偏移量，确保 k 从 0 开始
    """
    if period is None:
        period = random.uniform(28, 42) # 随机周期增加泛化性
    
    x_idx = np.arange(img_size)
    y_idx = np.arange(img_size)
    X, Y = np.meshgrid(x_idx, y_idx)
    
    # --- A. 随机贴边不规则掩模 (Mask) ---
    num_stuck = random.randint(2, 4)
    stuck_edges = random.sample(range(4), num_stuck) # 0:左, 1:右, 2:上, 3:下
    
    l = 0 if 0 in stuck_edges else random.randint(20, 50)
    r = img_size if 1 in stuck_edges else random.randint(img_size-50, img_size-20)
    t = 0 if 2 in stuck_edges else random.randint(20, 50)
    b = img_size if 3 in stuck_edges else random.randint(img_size-50, img_size-20)
    
    mask_bool = np.ones((img_size, img_size), dtype=bool)
    roughness, smooth = 4.0, 3.5
    if 0 not in stuck_edges:
        mask_bool &= (X > (l + gaussian_filter(np.random.randn(img_size)*roughness, sigma=smooth))[:, None])
    if 1 not in stuck_edges:
        mask_bool &= (X < (r + gaussian_filter(np.random.randn(img_size)*roughness, sigma=smooth))[:, None])
    if 2 not in stuck_edges:
        mask_bool &= (Y > (t + gaussian_filter(np.random.randn(img_size)*roughness, sigma=smooth))[None, :])
    if 3 not in stuck_edges:
        mask_bool &= (Y < (b + gaussian_filter(np.random.randn(img_size)*roughness, sigma=smooth))[None, :])
    
    mask = mask_bool.astype(np.float32)

    # --- B. 生成绝对相位 Phi ---
    # 使用较宽的 X_offset 确保 Phi 始终为正
    X_offset = 150 
    surf_rand = np.random.randn(img_size, img_size) * 160
    distort = gaussian_filter(surf_rand, sigma=random.uniform(15, 25))
    Phi = (2 * np.pi / period) * (X + distort + X_offset)

    # --- C. 真实感噪声 (底噪 + 跳变处轻微抖动) ---
    base_noise = np.random.normal(0, 0.02, (img_size, img_size))
    jump_weight = np.exp(-((np.cos(Phi) + 1)**2) / 0.005) 
    jitter_noise = np.random.normal(0, 0.06, (img_size, img_size)) * jump_weight
    
    # 生成带噪声的包裹相位
    wrapped_phase = np.arctan2(np.sin(Phi + base_noise + jitter_noise), 
                               np.cos(Phi + base_noise + jitter_noise))
    
    # --- D. 计算级数 k 并对齐 ---
    k_order_raw = np.round((Phi - wrapped_phase) / (2 * np.pi)).astype(np.int32)
    
    # 执行全局对齐 (k = k_raw - global_min)
    k_final = k_order_raw - global_k_shift
    
    # 背景处理：mask外设为 -1 (ignore_index)
    k_final[mask == 0] = -1
    wrapped_phase *= mask # 背景设为 0
    
    # 调制度 (Modulation) 模拟有效区域
    modulation = gaussian_filter(mask, sigma=1.0) * 0.9 + 0.05

    return wrapped_phase, modulation, k_final, Phi, mask

# ==========================================
# 2. 单个样本可视化函数
# ==========================================
def visualize_sample():
    # 临时生成一个不带偏移的样本查看范围
    w, m, k, phi, mask = generate_sample(global_k_shift=0)
    
    # 找出该样本有效区域的最小 k 作为演示偏移
    k_min = k[mask > 0].min()
    k -= k_min
    k[mask == 0] = -1

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 包裹相位
    axes[0].imshow(((w + np.pi) / (2 * np.pi) * 255).astype(np.uint8), cmap='gray')
    axes[0].set_title("Input: Wrapped Phase (Gray)")
    
    # 2. 调制度/掩模
    im1 = axes[1].imshow(m, cmap='magma')
    axes[1].set_title("Input: Modulation / Mask")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. 级数标签 (k 从 0 开始)
    k_show = np.ma.masked_where(k == -1, k)
    im2 = axes[2].imshow(k_show, cmap='jet')
    axes[2].set_title(f"Label: Fringe Order k\n(Starts from 0)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. 绝对相位 (Ground Truth)
    phi_show = np.ma.masked_where(mask == 0, phi)
    im3 = axes[3].imshow(phi_show, cmap='viridis')
    axes[3].set_title("Info: Absolute Phase ($\Phi$)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. 批量生成函数
# ==========================================
def batch_generate(save_dir, num_samples=1000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # --- 第一步：确定全局级数偏移量 ---
    # 随机生成 50 组样本，找出一个能让所有样本 k 都 >= 0 的全局最小值
    print("正在计算全局级数偏移量...")
    mins = []
    for _ in range(50):
        _, _, k, _, mask = generate_sample(img_size = 256, global_k_shift=0)
        if np.any(mask > 0):
            mins.append(k[mask > 0].min())
    global_min_k = min(mins)
    print(f"确定的全局最小级数为: {global_min_k}，所有数据将以此对齐。")

    # --- 第二步：批量生成并保存 ---
    print(f"开始生成数据集至: {save_dir}")
    for i in tqdm(range(num_samples)):
        w, m, k, _, _ = generate_sample(img_size = 256, global_k_shift=global_min_k)
        
        sample_dict = {
            'wph': w.astype(np.float32),
            'modulation': m.astype(np.float32),
            'series': k.astype(np.int16)
        }
        
        np.save(os.path.join(save_dir, f"sample_{i:05d}.npy"), sample_dict)
    
    print("任务完成！")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 选项 1: 可视化检查效果
    # visualize_sample()
    
    # 选项 2: 正式批量生成 (示例生成 100 组)
    batch_generate("./simulation_data", num_samples=1000)