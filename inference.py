import matplotlib.pyplot as plt
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

from src.data.panbench_datamodule import PanBenchDataModule
from src.models.components.pnn import PNN
from src.models.panbench_module import PanBenchLitModule

model = PNN()
checkpoint = torch.load(
    r"C:\Users\87897\Desktop\Pansharpening\logs\train\runs\2023-11-01_14-33-55/checkpoints\epoch=35-val_psnr=28.1618-val_ssim=0.7918.ckpt"
)
# the error is the following
# RuntimeError: Error(s) in loading state_dict for PNN:
#     Missing key(s) in state_dict: "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias".
#     Unexpected key(s) in state_dict: "net.net.0.weight", "net.net.0.bias", "net.net.2.weight", "net.net.2.bias", "net.net.4.weight", "net.net.4.bias".
# help me please
state_dict = {}
for key in checkpoint["state_dict"]:
    new_key = key.replace("net.net", "net")
    state_dict[new_key] = checkpoint["state_dict"][key]
model.load_state_dict(state_dict)
model.eval()


dm = PanBenchDataModule(batch_size=1)
dm.setup()
test_loader = dm.val_dataloader()

psnr = PSNR(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
ssim = SSIM(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])

avg_psnr = 0
avg_ssim = 0

for batch in test_loader:
    with torch.no_grad():
        pd = model(batch["lrms"], batch["lrpan"])
        gt = batch["ms"]
        avg_psnr += psnr(pd, gt).item() * pd.size(0)
        avg_ssim += ssim(pd, gt).item() * pd.size(0)
avg_psnr /= len(test_loader.dataset)
avg_ssim /= len(test_loader.dataset)
print(f"PSNR: {avg_psnr}")
print(f"SSIM: {avg_ssim}")
