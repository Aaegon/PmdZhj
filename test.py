from torch.utils.data import DataLoader
from dataset import FringeDataset, test_gt_absolute_phase

use_sin = False
dataset = FringeDataset("data", use_sin)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

test_gt_absolute_phase(
    dataloader=loader,
    save_dir="gt_abs_phase_debug",
    max_vis=10,
    use_sin = use_sin
)
