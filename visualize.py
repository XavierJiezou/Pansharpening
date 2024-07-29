import torch
import glob
import os
import numpy as np
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.metrics import ref_evaluate, ergas
from src.utils.metrics import no_ref_evaluate
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from src.data.panbench_datamodule import PanBenchDataModule

from src.models.components.pnn import PNN
from src.models.components.pannet import PanNet
from src.models.components.msdcnn import MSDCNN
from src.models.components.tfnet import TFNet
from src.models.components.fusionnet import FusionNet
from src.models.components.gppnn import GPPNN
from src.models.components.srppnn import SRPPNN
from src.models.components.pgcu import PGCU
from src.models.components.cmfnet import UEDM


class Visualize:
    def __init__(self) -> None:
        self.model_sets = {
            "PNN": PNN(),  # ("net._orig_mod.", "")
            "PanNet": PanNet(),
            "MSDCNN": MSDCNN(),
            "TFNet": TFNet(),
            "FusionNet": FusionNet(),
            "GPPNN": GPPNN(),
            "SRPPNN": SRPPNN(),
            "PGCU": PGCU(),
            "CMFNet": UEDM(width=32),  # ("net.", "")
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data(self) -> VisionDataset:
        dm = PanBenchDataModule(batch_size=1, pin_memory=True)
        dm.setup()
        test_loader = dm.test_dataloader()
        return test_loader

    def load_model(self, model_name) -> torch.nn.Module:
        if model_name == "CMFNet":
            ckpt_path = glob.glob(f"logs/train/runs/cmfnet/checkpoints/*.ckpt")[0]
        else:
            ckpt_path = glob.glob(f"logs/train/runs/{model_name.lower()}/checkpoints/*.ckpt")[0]
        checkpoint = torch.load(ckpt_path,map_location=self.device)
        state_dict = {}
        for key in checkpoint["state_dict"]:
            if model_name == "CMFNet":
                new_key = key.replace("net.", "")
            else:
                new_key = key.replace("net._orig_mod.", "")
            state_dict[new_key] = checkpoint["state_dict"][key]
        model = self.model_sets[model_name]
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def visualize(self, sample_path: str = None, input_type: str = "reduced") -> None:
        if not os.path.exists("vis_pdf"):
            os.mkdir("vis_pdf")
        # 绘制一个长为10，宽为2的子图
        # 第一行显示各个模型生成的fake_lrms的RGB图像，共9个模型，所有有9个子图，然后还有一个真实的RGB图像，即为ms
        # 第二行显示各个模型生成的fake_lrms的图像和真实MS的差值图像，共9个模型，所有有9个子图，然后还有一个真实ms图像和它自己的差值图像
        data = self.load_data()
        index = 0
        for sample in tqdm(data):
            index += 1
            if not sample_path:
                # sample = next(iter(data))
                intput_ms = sample["lrms"] if input_type == "reduced" else sample["ms"]
                intput_pan = (
                    sample["lrpan"] if input_type == "reduced" else sample["pan"]
                )
            intput_ms, intput_pan = intput_ms.to(self.device), intput_pan.to(self.device)
            groud_truth = sample["ms"]
            plt.figure(figsize=(20, 4))
            # 调小间距
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            # 设置字体和字号
            for i, j in enumerate(self.model_sets):
                with torch.no_grad():
                    fake = self.load_model(j).forward(intput_ms, intput_pan)
                    fake = fake.cpu()
                # fake = (fake*255).type(torch.uint8)
                fake = torch.clip(fake, 0, 1)
                PSNR = psnr(fake, groud_truth, data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
                ERGAS = ergas(
                    *[
                        i.permute(1, 2, 0).numpy()
                        for i in [fake.squeeze(0), groud_truth.squeeze(0)]
                    ]
                )
                plt.subplot(2, 10, i + 1)
                plt.title(
                    f"{j}", fontdict={"family": "Times New Roman", "size": 12}
                )
                
                plt.imshow(fake.squeeze(0).permute(1, 2, 0)[..., :3])
                plt.axis("off")
                plt.subplot(2, 10, i + 11)
                # 设置标题在图片下方
                
                # plt.title(
                #     f"{PSNR:.4f}\n{ERGAS:.4f}", fontdict={"family": "Times New Roman", "size": 12}
                # )
                # 设置标题的位置在图片下方
                plt.text(0.5, -0.25, f"{PSNR:.4f}\n{ERGAS:.4f}", ha='center', va='bottom', transform=plt.gca().transAxes, fontdict={"family": "Times New Roman", "size": 12})
                diff = (fake.squeeze(0) - groud_truth.squeeze(0)).abs().permute(1, 2, 0)
                diff = torch.mean(diff, dim=-1, keepdim=True) / 255
                # diff = np.abs((fake.squeeze(0) - groud_truth).permute(1, 2, 0).numpy())
                # diff = np.mean(diff, axis=-1)/255
                # 2%线性拉伸diff
                diff = np.clip(diff, np.quantile(diff, 0.02), np.quantile(diff, 0.98))

                plt.imshow(diff, cmap="rainbow")
                plt.axis("off")
            plt.subplot(2, 10, 10)
            plt.title("GT")
            plt.imshow(groud_truth.squeeze(0).permute(1, 2, 0)[..., :3])
            plt.axis("off")
            plt.subplot(2, 10, 20)
            # 设置标题为：PSNR换行ERGAS
            plt.text(0.5, -0.25, f"PSNR\nERGAS", ha='center', va='bottom', transform=plt.gca().transAxes, fontdict={"family": "Times New Roman", "size": 12})
            plt.imshow(
                torch.zeros_like(groud_truth.squeeze(0).permute(1, 2, 0)[..., 0]), cmap="rainbow"
            )
            plt.axis("off")
            plt.savefig(
                    f"vis_pdf/{index}_{sample['sate']}_{sample['scene']}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )

    def save(self, sample_path: str = None, input_type: str = "reduced") -> None:
        # 绘制一个长为10，宽为2的子图
        # 第一行显示各个模型生成的fake_lrms的RGB图像，共9个模型，所有有9个子图，然后还有一个真实的RGB图像，即为ms
        # 第二行显示各个模型生成的fake_lrms的图像和真实MS的差值图像，共9个模型，所有有9个子图，然后还有一个真实ms图像和它自己的差值图像
        data = self.load_data()
        index = 0
        for sample in data:
            index += 1
            if not sample_path:
                # sample = next(iter(data))
                intput_ms = sample["lrms"] if input_type == "reduced" else sample["ms"]
                intput_pan = (
                    sample["lrpan"] if input_type == "reduced" else sample["pan"]
                )
            groud_truth = sample["ms"]
            plt.figure(figsize=(20, 4))
            # 调小间距
            # plt.subplots_adjust(wspace=0.01, hspace=0.01)
            # 设置字体和字号
            for i, j in enumerate(self.model_sets):
                with torch.no_grad():
                    fake = self.load_model(j).forward(
                        *[i.unsqueeze(0) for i in [intput_ms, intput_pan]]
                    )
                # fake = (fake*255).type(torch.uint8)
                fake = torch.clip(fake, 0, 1)
                PSNR, SSIM, SAM, ERGAS, SCC, Q = ref_evaluate(
                    *[
                        i.permute(1, 2, 0).numpy()
                        for i in [fake.squeeze(0), groud_truth]
                    ]
                )
                plt.subplot(2, 10, i + 1)
                plt.title(
                    f"{PSNR:.4f}", fontdict={"family": "Times New Roman", "size": 12}
                )
                plt.imshow(fake.squeeze(0).permute(1, 2, 0)[..., :3])
                plt.axis("off")
                plt.subplot(2, 10, i + 11)
                plt.title(
                    f"{ERGAS:.4f}", fontdict={"family": "Times New Roman", "size": 12}
                )
                diff = (fake.squeeze(0) - groud_truth).abs().permute(1, 2, 0)
                diff = torch.mean(diff, dim=-1, keepdim=True) / 255
                # diff = np.abs((fake.squeeze(0) - groud_truth).permute(1, 2, 0).numpy())
                # diff = np.mean(diff, axis=-1)/255
                # 2%线性拉伸diff
                diff = np.clip(diff, np.quantile(diff, 0.02), np.quantile(diff, 0.98))

                plt.imshow(diff, cmap="rainbow")
                plt.axis("off")
            plt.subplot(2, 10, 10)
            plt.title("GT")
            plt.imshow(groud_truth.permute(1, 2, 0)[..., :3])
            plt.axis("off")
            plt.subplot(2, 10, 20)
            plt.imshow(
                torch.zeros_like(groud_truth.permute(1, 2, 0)[..., 0]), cmap="rainbow"
            )
            plt.axis("off")
            plt.savefig(
                f"vis_pdf/{index}_{sample['sate']}_{sample['scene']}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )


if __name__ == "__main__":
    Visualize().visualize()
