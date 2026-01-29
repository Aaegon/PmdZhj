from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import FringeDataset
from residual_unet import ResidualUNet, ResBlock, FringeOrderLoss

def visualize_prediction(
    inputs,        # (C, H, W)  0: sinφ, 1: cosφ
    gt_order,      # (1, H, W) or (H, W)
    pred_order,    # (1, H, W) or (H, W)
    wph,
    save_path,
    title=""
):
    # -------- 计算 wrapped phase --------
    # sin_phi = inputs[0].cpu().numpy()
    # cos_phi = inputs[1].cpu().numpy()
    # wrapped_phase = np.arctan2(sin_phi, cos_phi)
    wrapped_phase = wph.squeeze().cpu().numpy()
    # wrapped_phase = np.mod(wrapped_phase, 2 * np.pi)

    gt = gt_order.squeeze().cpu().numpy()
    pred = pred_order.squeeze().cpu().numpy()

    # -------- 绝对相位 --------
    abs_phase_gt = (wrapped_phase + 2 * np.pi * gt)*255/(2**5*np.pi)
    abs_phase_pred = (wrapped_phase + 2 * np.pi * pred)*255/(2**5*np.pi)

    abs_phase_gt = abs_phase_gt.astype(np.uint8)
    abs_phase_pred = abs_phase_pred.astype(np.uint8)

    abs_phase_err = abs_phase_pred - abs_phase_gt
    pm1_mask = (np.abs(pred - gt) >= 1).astype(float)

    # -------- 可视化 --------
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    im0 = axs[0].imshow((abs_phase_gt), cmap="gray")
    axs[0].set_title("GT Absolute Phase")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow((abs_phase_pred), cmap="gray")
    axs[1].set_title("Pred Absolute Phase")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(abs_phase_err, cmap="bwr")
    axs[2].set_title("Abs Phase Error")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(pm1_mask, cmap="gray")
    axs[3].set_title("|Order Error| > 1")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def predict_and_visualize(
    model_class,
    model_kwargs,
    checkpoint_path,
    dataloader,
    device="cuda",
    save_dir="pred_vis",
    max_vis=10
):
    """
    model_class: ResidualUNet
    model_kwargs: dict, e.g. {"in_channels": 3}
    checkpoint_path: .pth file
    dataloader: DataLoader
    """

    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ 构建 & 加载模型
    model = model_class(**model_kwargs).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")

    # 2️⃣ 推理 + 画图
    with torch.no_grad():
        for i, (inputs, gt_order, modulation, wph) in enumerate(dataloader):
            if i >= max_vis:
                break

            inputs = inputs.to(device)
            gt_order = gt_order.to(device)

            pred = model(inputs)
            pred_int = torch.round(pred)
            
            visualize_prediction(
                inputs[0],
                gt_order[0],
                pred_int[0],
                wph,
                save_path=os.path.join(save_dir, f"sample_{i}.png"),
                title=f"Sample {i}"
            )

    print(f"Saved visualizations to {save_dir}")

def test():
    dataset = FringeDataset("simulation_data", True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predict_and_visualize(
        model_class=ResidualUNet,
        model_kwargs={"in_channels": 3},
        checkpoint_path="weights/best_fold_4.pth",
        dataloader=loader,
        device="cuda",
        save_dir="vis_best_fold_4",
        max_vis=10
    )

def train():
    root_dir = Path("simulation_data")
    all_files = sorted(root_dir.glob("*.npy"))
    num_samples = len(all_files)

    print("Total samples:", num_samples)

    dataset = FringeDataset(root_dir, use_sincos = True)
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n========== Fold {fold+1}/{K} ==========")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        model = ResidualUNet(in_channels=3).to(device)
        criterion = FringeOrderLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_err = 1e9
        best_state = None

        # -------- 训练 --------
        for epoch in range(200):
            model.train()
            train_loss = 0.0

            for inputs, gt_order, modulation, wph in train_loader:
                inputs = inputs.to(device)
                gt_order = gt_order.to(device)
                modulation = modulation.to(device)
                wph = wph.to(device)

                pred = model(inputs)
                loss = criterion(pred, gt_order, modulation, wph, epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # -------- 验证 --------
            model.eval()
            abs_err = []
            pm1_err = []

            with torch.no_grad():
                for inputs, gt_order, modulation, wph in val_loader:
                    inputs = inputs.to(device)
                    gt_order = gt_order.to(device)

                    pred = model(inputs)
                    pred_int = torch.round(pred)

                    diff = torch.abs(pred_int - gt_order)

                    abs_err.append(diff.mean().item())
                    pm1_err.append((diff > 1).float().mean().item())

            mean_abs_err = np.mean(abs_err)
            mean_pm1 = np.mean(pm1_err)

            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.4f} | "
                f"Val MAE {mean_abs_err:.4f} | "
                f">±1 {mean_pm1:.4f}"
            )

            # -------- 选最稳模型（不是最小 loss） --------
            if mean_pm1 < best_val_err:
                best_val_err = mean_pm1
                best_state = model.state_dict()

        # 保存这一折的最佳模型
        torch.save(best_state, f"weights/best_fold_{fold}.pth")

        fold_results.append({
            "fold": fold,
            "pm1_error": best_val_err,
        })
    print("\n====== K-fold Summary ======")
    for r in fold_results:
        print(f"Fold {r['fold']} | >±1 error: {r['pm1_error']:.4f}")

    best_fold = min(fold_results, key=lambda x: x["pm1_error"])
    print("\nBest fold:", best_fold)


if __name__ == "__main__":
    test()
    # train()

