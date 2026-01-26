import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import build_dataloader

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)
    
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.res = ResBlock(in_ch, out_ch)

    def forward(self, x):
        return self.res(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

class ResidualUNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        self.in_conv = ResBlock(in_channels, 32)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.out_conv(x)

def gradient(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy

class FringeOrderLoss(nn.Module):
    """
    Least-Squares based loss for fringe order prediction,
    including:
    - data term (pred vs GT)
    - gradient LS (pred gradient vs GT gradient)
    - self-smoothness term (pred gradient magnitude)
    """
    def __init__(self, grad_weight=0.4, self_grad_weight=0.1):
        """
        grad_weight:  GT gradient loss 权重
        self_grad_weight: 预测自身平滑约束权重
        """
        super().__init__()
        self.grad_weight = grad_weight
        self.self_grad_weight = self_grad_weight

    def forward(self, pred, gt, modulation, wph):
        """
        pred, gt   : (B,1,H,W)   fringe order
        modulation : (B,1,H,W)   ∈ [0,1]
        wph        : (B,1,H,W)   wrapped phase
        """
        # ===== 1. 数据项（最小二乘 order） =====
        data_loss = (modulation * (pred - gt) ** 2).mean()

        # ===== 2. 梯度约束：预测 vs GT =====
        abs_pred = wph + 2 * torch.pi * pred
        abs_gt   = wph + 2 * torch.pi * gt

        dx_p, dy_p = gradient(abs_pred)
        dx_g, dy_g = gradient(abs_gt)

        mod_x = modulation[:, :, :, 1:]
        mod_y = modulation[:, :, 1:, :]

        grad_loss = (
            (mod_x * (dx_p - dx_g) ** 2).mean() +
            (mod_y * (dy_p - dy_g) ** 2).mean()
        )

        # ===== 3. 自身最小二乘平滑约束 =====
        self_grad_loss = (
            (dx_p ** 2 * mod_x).mean() +
            (dy_p ** 2 * mod_y).mean()
        )

        # ===== 总 loss =====
        total_loss = data_loss + self.grad_weight * grad_loss + self.self_grad_weight * self_grad_loss
        return total_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = build_dataloader(
        root_dir="data",
        batch_size=1,
        shuffle=True
    )

    model = ResidualUNet(in_channels=3).to(device)
    criterion = FringeOrderLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(300):
        model.train()
        epoch_loss = 0.0

        for inputs, gt_order, modulation in dataloader:
            inputs = inputs.to(device)        # (B,3,H,W)
            gt_order = gt_order.to(device)    # (B,1,H,W)
            modulation = modulation.to(device)

            pred = model(inputs)
            loss = criterion(pred, gt_order, modulation)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch:03d} | Loss {epoch_loss / len(dataloader):.6f}")


