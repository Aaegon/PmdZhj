import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

class CoordConv(nn.Module):
    """增加坐标通道，帮助网络理解空间位置与级次的单调关系"""
    def __init__(self, in_channels):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        # 生成归一化坐标 [0, 1]
        y_coord = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(batch_size, 1, h, w).to(x.device)
        x_coord = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(batch_size, 1, h, w).to(x.device)
        x = torch.cat([x, y_coord, x_coord], dim=1)
        return self.conv(x)

class DeflectoNet(nn.Module):
    def __init__(self, num_classes=20): # 20为最大相对级次数
        super(DeflectoNet, self).__init__()
        # 初始 6 通道输入：4 Fringes + 1 WPH + 1 MOD
        self.coord_conv = CoordConv(6)
        
        # 标准 U-Net 结构
        def encoder_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = encoder_block(6, 64)
        self.enc2 = encoder_block(64, 128)
        self.enc3 = encoder_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = encoder_block(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = encoder_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = encoder_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = encoder_block(128, 64)
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [B, 6, H, W]
        x = self.coord_conv(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)

class PhasePatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 加载字典数据
        data = np.load(self.file_list[idx], allow_pickle=True).item()
        
        # 1. 输入组装 [6, H, W]
        # 这里的归一化顺序要和仿真一致
        fringes = data['fringes'].astype(np.float32) # [4, H, W], 已归一化
        wph = data['wph'][np.newaxis, ...].astype(np.float32) / (2 * np.pi) # [1, H, W]
        mod = data['modulation'][np.newaxis, ...].astype(np.float32) # [1, H, W]
        
        x_input = np.concatenate([fringes, wph, mod], axis=0)
        
        # 2. 标签处理
        series = data['series'].astype(np.int64)
        series_min = np.min(series)
        target = series - series_min # 相对级次化，方便跨样本泛化
        
        return torch.from_numpy(x_input), torch.from_numpy(target)
    
class DeflectoLoss(nn.Module):
    def __init__(self, tv_weight=0.01):
        super(DeflectoLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # 设置为none以便应用掩模
        self.tv_weight = tv_weight

    def forward(self, pred, target, mod_mask):
        """
        pred: [B, num_classes, H, W]
        target: [B, H, W]
        mod_mask: 调制度掩模 [B, H, W], 来自输入的第6通道
        """
        # 1. 基础分类 Loss (仅在有效调制度区域计算)
        ce = self.ce_loss(pred, target) 
        # 使用调制度作为权重，mod低的地方loss贡献小
        weighted_ce = (ce * mod_mask).mean()

        # 2. TV Loss (抑制级次预测碎噪点)
        # 对预测概率图求空间梯度
        prob = F.softmax(pred, dim=1)
        tv_h = torch.pow(prob[:, :, 1:, :] - prob[:, :, :-1, :], 2).mean()
        tv_w = torch.pow(prob[:, :, :, 1:] - prob[:, :, :, :-1], 2).mean()
        
        return weighted_ce + self.tv_weight * (tv_h + tv_w)

def train_phase_unwrapping():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 15 # 根据仿真数据的最大级次动态范围设定
    
    model = DeflectoNet(num_classes=num_classes).to(device)
    dataset = PhasePatchDataset('./sim_dataset_A')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion = DeflectoLoss(tv_weight=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Phase A Pre-training...")
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # 提取调制度通道作为掩模 (第6通道，索引5)
            mod_mask = x[:, 5, :, :] 
            
            optimizer.zero_grad()
            outputs = model(x)
            
            loss = criterion(outputs, y, mod_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), "deflecto_net_pretrained.pth")

def visualize_prediction(model, datafolder, savefolder, device="cuda",):
    pass

if __name__ == "__main__":
    train_phase_unwrapping()