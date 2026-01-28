from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from torch.utils.data import Dataset, DataLoader
class FringeDataset(Dataset):
    def __init__(self, root_dir, use_sincos=True):
        """
        root_dir: 数据目录，里面全是 .npy
        use_sincos: 是否把 wrapped phase 转成 sin/cos
        """
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.npy"))
        assert len(self.files) > 0, "数据目录下没有 .npy 文件"

        self.use_sincos = use_sincos

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()

        phi = data["wph"].astype(np.float32)
        mod = data["modulation"].astype(np.float32)
        order = data["series"].astype(np.float32)

        # -------- sanity check（很重要） --------
        if not np.isfinite(phi).all():
            raise ValueError(f"{self.files[idx]}: wrapped_phase 有 NaN/Inf")
        if not np.isfinite(mod).all():
            raise ValueError(f"{self.files[idx]}: modulation 有 NaN/Inf")

        mod = np.clip(mod, 0.0, 1.0)

        # -------- 输入构造 --------
        if self.use_sincos:
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            inputs = np.stack([sin_phi, cos_phi, mod], axis=0)  # (3,H,W)
        else:
            inputs = np.stack([phi, mod], axis=0)               # (2,H,W)

        inputs = torch.from_numpy(inputs)
        order = torch.from_numpy(order).unsqueeze(0)            # (1,H,W)
        mod = torch.from_numpy(mod).unsqueeze(0)                # (1,H,W)
        wph = torch.from_numpy(phi).unsqueeze(0)

        return inputs, order, mod, wph

def build_dataloader(
    root_dir,
    batch_size=1,
    shuffle=True,
    num_workers=0
):
    dataset = FringeDataset(root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

def compute_gt_absolute_phase_from_batch(inputs, gt_order, use_sin):
    """
    inputs:   (B, C, H, W), channel 0 = sinφ, channel 1 = cosφ
    gt_order: (B, 1, H, W) or (B, H, W)

    return:
        abs_phase_gt: (B, H, W)  numpy
    """

        # -------- sin / cos --------
    if use_sin:
        sin_phi = inputs[:, 0]          # (B, H, W)
        cos_phi = inputs[:, 1]          # (B, H, W)

        wrapped_phase = torch.atan2(sin_phi, cos_phi)
    else:
        wrapped_phase = inputs[:, 0]
    # -------- GT order --------
    gt = gt_order.squeeze(1) if gt_order.ndim == 4 else gt_order

    abs_phase_gt = wrapped_phase + 2 * torch.pi * gt

    return abs_phase_gt.cpu().numpy()

def test_gt_absolute_phase(
    dataloader,
    save_dir="gt_abs_phase_check",
    max_vis=5,
    use_sin = False
):
    os.makedirs(save_dir, exist_ok=True)

    for i, (inputs, gt_order, modulation) in enumerate(dataloader):
        if i >= max_vis:
            break

        # -------- 生成 GT 绝对相位 --------
        abs_phase_gt = compute_gt_absolute_phase_from_batch(inputs, gt_order, use_sin)

        abs_phase = abs_phase_gt[0]*255/(2**5*np.pi)  # (H, W)
        abs_phase = abs_phase.astype(np.uint8)
        
        # -------- 可视化 --------
        plt.figure(figsize=(6, 4))
        plt.imshow(abs_phase, cmap="gray")
        plt.colorbar()
        plt.title(f"GT Absolute Phase | Sample {i}")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"gt_abs_phase_{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved {save_path}")
