import cv2
import numpy as np
import torch
from pre import PMD

def wrap_to_pi(x):
    """将 [0, 2π] 映射到 [-π, π]"""
    return (x + torch.pi) % (2 * torch.pi) - torch.pi
def unwrap_phase_lse(wrapped_phase_pi):
    h, w = wrapped_phase_pi.shape
    device = wrapped_phase_pi.device

    dx = wrap_to_pi(wrapped_phase_pi[:, 1:] - wrapped_phase_pi[:, :-1])
    dy = wrap_to_pi(wrapped_phase_pi[1:, :] - wrapped_phase_pi[:-1, :])

    dx_p = torch.zeros_like(wrapped_phase_pi)
    dy_p = torch.zeros_like(wrapped_phase_pi)
    dx_p[:, :-1] = dx
    dy_p[:-1, :] = dy

    fx = torch.zeros_like(wrapped_phase_pi)
    fy = torch.zeros_like(wrapped_phase_pi)
    fx[:, 0] = dx_p[:, 0]
    fx[:, 1:] = dx_p[:, 1:] - dx_p[:, :-1]
    fy[0, :] = dy_p[0, :]
    fy[1:, :] = dy_p[1:, :] - dy_p[:-1, :]

    f = fx + fy

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )

    denom = 2 * (
        torch.cos(2 * torch.pi * xx / w) +
        torch.cos(2 * torch.pi * yy / h) - 2
    )
    denom[0, 0] = 1

    phi = torch.real(torch.fft.ifft2(torch.fft.fft2(f) / denom))
    return phi

def unwrap_phase_wls(wrapped_phase_pi, weight):
    """
    加权最小二乘相位展开（WLS, 频域近似解）
    
    wrapped_phase_pi: [H, W], ∈ [-π, π]
    weight:           [H, W], ≥ 0（调制度/置信度）
    """
    h, w = wrapped_phase_pi.shape
    device = wrapped_phase_pi.device

    # 梯度（wrapped）
    dx = wrap_to_pi(wrapped_phase_pi[:, 1:] - wrapped_phase_pi[:, :-1])
    dy = wrap_to_pi(wrapped_phase_pi[1:, :] - wrapped_phase_pi[:-1, :])

    # 权重（梯度位置）
    wx = weight[:, :-1]
    wy = weight[:-1, :]

    dx_w = wx * dx
    dy_w = wy * dy

    # padding
    dx_p = torch.zeros_like(wrapped_phase_pi)
    dy_p = torch.zeros_like(wrapped_phase_pi)
    dx_p[:, :-1] = dx_w
    dy_p[:-1, :] = dy_w

    # 散度 div( w^2 * grad )
    fx = torch.zeros_like(wrapped_phase_pi)
    fy = torch.zeros_like(wrapped_phase_pi)

    fx[:, 0] = dx_p[:, 0]
    fx[:, 1:] = dx_p[:, 1:] - dx_p[:, :-1]

    fy[0, :] = dy_p[0, :]
    fy[1:, :] = dy_p[1:, :] - dy_p[:-1, :]

    f = fx + fy

    # 频域泊松解
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )

    denom = (
        wx.mean() * (2 * torch.cos(2 * torch.pi * xx / w) - 2) +
        wy.mean() * (2 * torch.cos(2 * torch.pi * yy / h) - 2)
    )
    denom[0, 0] = 1.0  # gauge fixing

    phi = torch.real(torch.fft.ifft2(torch.fft.fft2(f) / denom))
    return phi

def visualize_phase(phi, clip_percentile=(5, 95)):
    """
    物理安全的相位可视化
    """
    phi_cpu = phi.detach().cpu().numpy()

    p1, p99 = np.percentile(phi_cpu, clip_percentile)
    phi_clip = np.clip(phi_cpu, p1, p99)

    vis = (phi_clip - p1) / (p99 - p1 + 1e-8)
    vis = (vis * 255).astype(np.uint8)
    return vis

def get_gradient(I):
    # 计算 Sobel 梯度
    sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
    sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

    # 计算梯度的幅值和方向
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

    # 计算平均梯度
    return (255-magnitude).astype(np.uint8), angle.astype(np.uint8)



# === 使用示例 ===
if __name__ == "__main__":
    aa = PMD(datapath= r'D:\zhj\code\datapath\data_final\17', th = [0.2, 4.1, 2.0, 1.3, 1.8])
    wph = aa.getRawWph()
    modulation = aa.get_raw_modulation()
    weight = modulation/(modulation.max() + 1e-8)
    # unwrapped = unwrap_phase_lse(wph)
    unwrapped = unwrap_phase_wls(wph, weight)
    img = visualize_phase(unwrapped)
    cv2.imwrite('output/use_least_squares_unwrapped.png', img)
    img_gra, _ = get_gradient(img)
    cv2.imwrite('output/wls_gra.png', img_gra)



