import torch
import glob
import sys
from rich.progress import track
from src.utils.metrics import ref_evaluate
from src.utils.metrics import no_ref_evaluate
from src.data.panbench_datamodule import PanBenchDataModule

from src.models.components.pnn import PNN
from src.models.components.pannet import PanNet
from src.models.components.msdcnn import MSDCNN
from src.models.components.tfnet import TFNet
from src.models.components.fusionnet import FusionNet
from src.models.components.gppnn import GPPNN
from src.models.components.srppnn import SRPPNN
from src.models.components.pgcu import PGCU
from src.models.components.uedm_diffusion import UEDMDiffusion
from src.models.components.uedm import UEDM
from torchmetrics.image.psnr import PeakSignalNoiseRatio as _PNSR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as _SSIM


class Eval:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cpu"
        self.load_data()
        self.load_model()
        self.criterion = torch.nn.MSELoss()
        self.psnr = _PNSR(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        self.ssim = _SSIM(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        self.mse = torch.nn.MSELoss()

    def load_data(self) -> None:
        self.dm = PanBenchDataModule(batch_size=1, pin_memory=True)
        self.dm.setup()
        self.test_loader = self.dm.test_dataloader()
        self.data = self.test_loader

    def get_model(self) -> torch.nn.Module:
        if self.model_name == "pnn":  # ("net._orig_mod.", "")
            return PNN()
        elif self.model_name == "pannet":
            return PanNet()
        elif self.model_name == "msdcnn":
            return MSDCNN()
        elif self.model_name == "tfnet":
            return TFNet()
        elif self.model_name == "fusionnet":
            return FusionNet()
        elif self.model_name == "gppnn":
            return GPPNN()
        elif self.model_name == "srppnn":
            return SRPPNN()
        elif self.model_name == "pgcu":
            return PGCU()
        elif self.model_name == "uedm":  # ("net.", "")
            return UEDM(width=32)
        elif self.model_name == "uedmdf":  # ("net.", "")
            return UEDMDiffusion(width=32)
        else:
            raise ValueError("Model name is not valid")

    def load_model(self) -> None:
        ckpt_path = glob.glob(f"logs/train/runs/{self.model_name}color/checkpoints/*.ckpt")[
            0
        ]
        checkpoint = torch.load(ckpt_path)
        state_dict = {}
        for key in checkpoint["state_dict"]:
            if "net._orig_mod." in key:
                new_key = key.replace("net._orig_mod.", "")
            else:
                new_key = key.replace("net.", "")
            state_dict[new_key] = checkpoint["state_dict"][key]
        self.model = UEDM(width=32)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)

        self.model.eval()

    def evaluate(self) -> dict:
        no_ref_metrics = {"D_lambda": [], "D_s": [], "QNR": []}
        ref_metrics = {
            "PSNR": [],
            "SSIM": [],
            "SAM": [],
            "ERGAS": [],
            "SCC": [],
            "Q": [],
        }
        sates = {
            "water": {
                "PSNR": [],
                "SSIM": [],
                "MSE": [],
            },
            "urban": {
                "PSNR": [],
                "SSIM": [],
                "MSE": [],
            },
            "ice": {
                "PSNR": [],
                "SSIM": [],
                "MSE": [],
            },
            "crops": {
                "PSNR": [],
                "SSIM": [],
                "MSE": [],
            },
            "vegetation": {
                "PSNR": [],
                "SSIM": [],
                "MSE": [],
            },
            "barren": {
                "PSNR": [],
                "SSIM": [],
                "MSE": [],
            },
        }
        

        with torch.no_grad():
            for x in track(self.data):
                ms, lrpan, lrms, pan = [
                    i.to(self.device)
                    for i in [x["ms"], x["lrpan"], x["lrms"], x["pan"]]
                ]
                # sate = x["scene"][0].split(',')
                # print(sate)
                

                
                # fake_ms = self.model.forward(*[i.unsqueeze(0) for i in [ms, pan]])
                fake_lrms = self.model.forward(*[i for i in [lrms, lrpan]])
                # D_lambda, D_s, QNR = no_ref_evaluate(
                #     *[i.permute(1, 2, 0).numpy() for i in [fake_ms.squeeze(0), pan, ms]]
                # )
                PSNR, SSIM, SAM, ERGAS, SCC, Q = ref_evaluate(
                    *[i.permute(1, 2, 0).numpy() for i in [fake_lrms.squeeze(0), ms.squeeze(0)]]
                )
                
                # ref_metrics["PSNR"].append(PSNR)
                # ref_metrics["SSIM"].append(SSIM)
                ref_metrics["SAM"].append(SAM)
                ref_metrics["ERGAS"].append(ERGAS)
                ref_metrics["SCC"].append(SCC)

                

                # ref_metrics["Q"].append(Q)
                # no_ref_metrics["D_lambda"].append(D_lambda)
                # no_ref_metrics["D_s"].append(D_s)
                # no_ref_metrics["QNR"].append(QNR)
        result = {
            # "Full Resolution": {
            #     "D_lambda": f"{sum(no_ref_metrics['D_lambda'])/len(no_ref_metrics['D_lambda']):.4f}",
            #     "D_s": f"{sum(no_ref_metrics['D_s'])/len(no_ref_metrics['D_s']):.4f}",
            #     "QNR": f"{sum(no_ref_metrics['QNR'])/len(no_ref_metrics['QNR']):.4f}",
            # },
            "Reduced Resolution": {
                # "PSNR": f"{sum(ref_metrics['PSNR'])/len(ref_metrics['PSNR']):.4f}",
                # "SSIM": f"{sum(ref_metrics['SSIM'])/len(ref_metrics['SSIM']):.4f}",
                "SAM": f"{sum(ref_metrics['SAM'])/len(ref_metrics['SAM']):.4f}",
                "ERGAS": f"{sum(ref_metrics['ERGAS'])/len(ref_metrics['ERGAS']):.4f}",
                "SCC": f"{sum(ref_metrics['SCC'])/len(ref_metrics['SCC']):.4f}",
                # "Q": f"{sum(ref_metrics['Q'])/len(ref_metrics['Q']):.4f}",
            },
        }

        
        return {self.model_name: result}


if __name__ == "__main__":
    model_name = sys.argv[1]
    eval = Eval(model_name)
    result = eval.evaluate()
    print(result)
