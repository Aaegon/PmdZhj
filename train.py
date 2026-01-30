from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import FringeDataset
from deflect_unet import DeflectoNet, PhasePatchDataset

def visualize_prediction(model_path, test_data_path, device='cuda', num_classes=15):
    '''
    用训练完的模型进行可视化的函数
    '''
    # 1. 初始化模型并加载权重
    model = DeflectoNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 加载单组测试数据
    data = np.load(test_data_path, allow_pickle=True).item()
    
    # 预处理输入 (与 Dataset 逻辑一致)
    fringes = data['fringes'].astype(np.float32) 
    wph_norm = (data['wph'] / (2 * np.pi)).astype(np.float32)
    mod_norm = data['modulation'].astype(np.float32)
    
    # 组装输入 [1, 6, H, W]
    x_input = np.concatenate([fringes, wph_norm[np.newaxis, ...], mod_norm[np.newaxis, ...]], axis=0)
    x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)

    # 3. 模型推理
    with torch.no_grad():
        output = model(x_tensor)
        # 获取预测级数 (取概率最大的索引)
        pred_series = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # 4. 绝对相位重构与偏差计算
    # 注意：模型预测的是相对级次，需要加上原始 Patch 的偏移量或只对比相对值
    # 这里我们对比相对偏差，因为绝对相位的整体平移不影响缺陷检测
    phi_wrapped = data['wph'] *2*np.pi
    # abs_phase_gt = data['abs_phase_gt']
    
    # 重构绝对相位: Phi = phi + 2 * pi * (k_pred + offset)
    # 为了对比方便，我们将预测的级次序列平移到与 GT 相同的量级
    gt_series = data['series'].astype(np.uint64)
    k_offset = np.min(gt_series)
    abs_phase_gt = phi_wrapped + 2 * np.pi * gt_series
    recon_abs = phi_wrapped + 2 * np.pi * (pred_series + k_offset)
    
    # 计算偏差 (预测 - 真值)
    # 理想情况下，这个图中不应有 2pi 的阶跃，只有细小的毛刺噪声
    residual = recon_abs - abs_phase_gt

    # 5. 绘图
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(phi_wrapped, cmap='gray')
    plt.title('Wrapped Phase (Input)')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(pred_series, cmap='jet')
    plt.title('Predicted K Series')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(recon_abs, cmap='gray')
    plt.title('Reconstructed Absolute Phase')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(abs_phase_gt, cmap='gray')
    plt.title('Ground Truth Absolute Phase')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    # 偏差值可视化，使用 RdBu 色标观察正负偏差
    plt.imshow(residual, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Radians')
    plt.title('Prediction Residual (Error Map)')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    # 绘制一行剖面线对比，最直观看到跳变处是否平滑
    mid_row = pred_series.shape[0] // 2
    plt.plot(abs_phase_gt[mid_row, :], label='GT', color='black', alpha=0.5)
    plt.plot(recon_abs[mid_row, :], label='Pred', linestyle='--', color='red')
    plt.title(f'Cross-section at Row {mid_row}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 使用示例
# visualize_prediction('deflecto_net_pretrained.pth', './sim_dataset_A/sim_000001.npy')

if __name__ == "__main__":
    visualize_prediction('weights/deflecto_net_pretrained200.pth', './sim_dataset_A/sim_000018.npy')

