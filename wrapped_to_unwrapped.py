import cv2
import numpy as np
import torch
from pre import PMD, extra_region_through_binary
import math

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

def unwrap_phase_wls_defect_friendly(
    wrapped_phase_pi,
    weight,
    sigma_g=1.0,
    min_w=0.3
):
    """
    缺陷友好 WLS：
    - wd: 调制度（data term）
    - ws: 梯度抑制平滑（smooth term）
    """
    h, w = wrapped_phase_pi.shape
    device = wrapped_phase_pi.device

    # ---- data term 权重（clip，避免过度忽略） ----
    wd = torch.clamp(weight, min=min_w)

    # ---- wrapped 梯度 ----
    dx = wrap_to_pi(wrapped_phase_pi[:, 1:] - wrapped_phase_pi[:, :-1])
    dy = wrap_to_pi(wrapped_phase_pi[1:, :] - wrapped_phase_pi[:-1, :])

    # ---- 梯度幅值（用于 smooth 权重）----
    gx = torch.zeros_like(wrapped_phase_pi)
    gy = torch.zeros_like(wrapped_phase_pi)
    gx[:, :-1] = dx
    gy[:-1, :] = dy
    grad_mag = torch.sqrt(gx**2 + gy**2)

    ws = torch.exp(-(grad_mag**2) / (sigma_g**2))

    # ---- 分别作用在梯度上 ----
    wx = wd[:, :-1] * ws[:, :-1]
    wy = wd[:-1, :] * ws[:-1, :]

    dx_w = wx * dx
    dy_w = wy * dy

    # padding
    dx_p = torch.zeros_like(wrapped_phase_pi)
    dy_p = torch.zeros_like(wrapped_phase_pi)
    dx_p[:, :-1] = dx_w
    dy_p[:-1, :] = dy_w

    # divergence
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

    denom = (
        wx.mean() * (2 * torch.cos(2 * math.pi * xx / w) - 2) +
        wy.mean() * (2 * torch.cos(2 * math.pi * yy / h) - 2)
    )
    denom[0, 0] = 1.0

    phi = torch.real(torch.fft.ifft2(torch.fft.fft2(f) / denom))
    return phi
def unwrap_phase_wls_robust(
    wrapped_phase_pi,
    weight,
    sigma_g=1.0,
    min_w=0.3,
    eps=1e-3
):
    """
    强缺陷保护 WLS：
    - 梯度越大，平滑越弱（近似鲁棒）
    """
    h, w = wrapped_phase_pi.shape
    device = wrapped_phase_pi.device

    wd = torch.clamp(weight, min=min_w)

    dx = wrap_to_pi(wrapped_phase_pi[:, 1:] - wrapped_phase_pi[:, :-1])
    dy = wrap_to_pi(wrapped_phase_pi[1:, :] - wrapped_phase_pi[:-1, :])

    gx = torch.zeros_like(wrapped_phase_pi)
    gy = torch.zeros_like(wrapped_phase_pi)
    gx[:, :-1] = dx
    gy[:-1, :] = dy
    grad_mag = torch.sqrt(gx**2 + gy**2)

    # 鲁棒 smooth 权重（Tukey/Charbonnier 风格）
    ws = 1.0 / torch.sqrt(grad_mag**2 + sigma_g**2 + eps)

    wx = wd[:, :-1] * ws[:, :-1]
    wy = wd[:-1, :] * ws[:-1, :]

    dx_w = wx * dx
    dy_w = wy * dy

    dx_p = torch.zeros_like(wrapped_phase_pi)
    dy_p = torch.zeros_like(wrapped_phase_pi)
    dx_p[:, :-1] = dx_w
    dy_p[:-1, :] = dy_w

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

    denom = (
        wx.mean() * (2 * torch.cos(2 * math.pi * xx / w) - 2) +
        wy.mean() * (2 * torch.cos(2 * math.pi * yy / h) - 2)
    )
    denom[0, 0] = 1.0

    phi = torch.real(torch.fft.ifft2(torch.fft.fft2(f) / denom))
    return phi

def visualize_phase(phi, clip_percentile=(5, 95), is_cpu = False):
    """
    物理安全的相位可视化
    """
    if is_cpu:
        phi_cpu = phi
    else:
        phi_cpu = phi.detach().cpu().numpy()

    p1, p99 = np.percentile(phi_cpu, clip_percentile)
    print("拉伸前：",p1,p99)
    phi_clip = np.clip(phi_cpu, p1, p99)

    vis = (phi_clip - p1) / (p99 - p1 + 1e-8)
    vis = (vis * 255).astype(np.uint8)
    return vis

def get_gradient(I):
    """
    获取展开后的平均梯度，可视化明显一点的版本
    """
    # 计算 Sobel 梯度
    sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
    sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

    # 计算梯度的幅值和方向
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

    # 计算平均梯度
    return (255-magnitude).astype(np.uint8), angle.astype(np.uint8)


def defect_enhance(img, alpha=0.5):
    """
    缺陷增强项（修复版本）
    """
    if isinstance(img, torch.Tensor):
        residual = img.cpu().numpy().astype(np.float64)
    else:
        residual = img.astype(np.float64)
    
    gx = cv2.Sobel(residual, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(residual, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2 + 1e-8)  # 避免除零

    # 使用filter2D代替Laplacian
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float64)
    lap = cv2.filter2D(residual, cv2.CV_64F, laplacian_kernel)

    enhance = np.abs(lap) - alpha * grad
    return visualize_phase(enhance, (0,100), True)

def test_imaging_0112(datapath, th):
    """
    0112 低通滤波提取残差，再通过拉氏变换-滤波增强残差， wls的残差视觉上缺陷太不明显了
    0115 发现是因为wls的时候把缺陷区域抹平了 用了缺陷友好型效果还是差
    """
    aa = PMD(datapath, th)
    wph = aa.getRawWph()
    modulation = aa.get_raw_modulation()
    # modulation = modulation.detach().cpu().numpy()
    # mask = extra_region_through_binary(modulation)/255
    # weight = torch.from_numpy(mask).to('cuda')
    weight = modulation/(modulation.max() + 1e-6)
    unwrapped_lse = unwrap_phase_lse(wph)
    unwrapped_wls = unwrap_phase_wls(wph, weight)

    print("lse:")
    img_lse = visualize_phase(unwrapped_lse)
    print("wls:")
    img_wls = visualize_phase(unwrapped_wls)
    cv2.imwrite('output/lse.png', img_lse)
    cv2.imwrite('output/wls.png', img_wls)

    # residual_lse_wls = visualize_phase(torch.abs(unwrapped_lse - unwrapped_wls))
    # cv2.imwrite('output/wls-lse.png', residual_lse_wls)

    lse = unwrapped_lse.detach().cpu().numpy()
    wls = unwrapped_wls.detach().cpu().numpy()

    lse_bg = cv2.GaussianBlur(lse, (0,0), sigmaX=10)
    wls_bg = cv2.GaussianBlur(wls, (0,0), sigmaX=10)

    lse_residual = lse - lse_bg
    wls_residual = wls - wls_bg
    print("lse:")
    defect_lse = defect_enhance(lse_residual, alpha = 0.5 )
    print("wls:")
    defect_wls = defect_enhance(wls_residual, alpha = 0.5 )

    cv2.imwrite('output/wls_defect.png', defect_wls)
    cv2.imwrite('output/lse_defect.png', defect_lse)


# === 使用示例 ===
if __name__ == "__main__":
    test_imaging_0112(datapath= r'D:\zhj\code\datapath\data_final\17', th = [0.2, 4.1, 2.0, 1.3, 1.8])




