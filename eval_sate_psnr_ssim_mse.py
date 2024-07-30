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
from src.models.components.cmfnet_diffusion import UEDMDiffusion
from src.models.components.cmfnet import UEDM
from torchmetrics.image.psnr import PeakSignalNoiseRatio as _PNSR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as _SSIM
import numpy as np


class Eval:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = self.get_device()
        self.load_data()
        self.load_model()
        self.criterion = torch.nn.MSELoss()
        self.psnr = _PNSR(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3]).to(self.device)
        self.ssim = _SSIM(data_range=1.0, reduction="elementwise_mean").to(self.device)
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

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> None:
        ckpt_path = glob.glob(f"logs/train/runs/{self.model_name}/checkpoints/*.ckpt")[
            0
        ]
        checkpoint = torch.load(ckpt_path,map_location=self.device)
        state_dict = {}
        for key in checkpoint["state_dict"]:
            if "net._orig_mod." in key:
                new_key = key.replace("net._orig_mod.", "")
            else:
                new_key = key.replace("net.", "")
            state_dict[new_key] = checkpoint["state_dict"][key]
        self.model = self.get_model()
        self.model.load_state_dict(state_dict)
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
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "urban": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "ice": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "crops": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "vegetation": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "barren": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "GF1": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "GF2": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "GF6": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "LC7": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "LC8": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "WV2": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "WV3": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "WV4": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "QB": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
            "IN": {
                "SAM": [],
                "ERGAS": [],
                "SCC": [],
                "MSE": [],
                "PSNR": [],
                "SSIM": [],
                "D_lambda": [],
                "D_s": [],
                "QNR": [],
            },
        }
        

        with torch.no_grad():
            for x in track(self.data):
                ms, lrpan, lrms, pan = [
                    i.to(self.device)
                    for i in [x["ms"], x["lrpan"], x["lrms"], x["pan"]]
                ]
                sate = x["scene"][0].split(",")
                # print(sate)
                

                
                # fake_ms = self.model.forward(*[i.unsqueeze(0) for i in [ms, pan]])
                fake_lrms = self.model.forward(*[i for i in [lrms, lrpan]])
                # D_lambda, D_s, QNR = no_ref_evaluate(
                #     *[i.permute(1, 2, 0).numpy() for i in [fake_ms.squeeze(0), pan, ms]]
                # )

                c_pred, c_pan, c_hs = fake_lrms.detach().cpu().numpy()[0], lrpan.detach().cpu().numpy()[0], \
                lrms.detach().cpu().numpy()[0]
                c_pred = np.transpose(c_pred, (1, 2, 0))
                c_pan = np.transpose(c_pan, (1, 2, 0))
                c_hs = np.transpose(c_hs, (1, 2, 0))
                # print(c_pred.shape,c_pan.shape,c_hs.shape)
                c_D_lambda, c_D_s, c_qnr = no_ref_evaluate(c_pred, c_pan, c_hs)

                PSNR, SSIM, SAM, ERGAS, SCC, Q = ref_evaluate(
                    *[i.permute(1, 2, 0).cpu().numpy() for i in [fake_lrms.squeeze(0), ms.squeeze(0)]]
                )
                
                # ref_metrics["PSNR"].append(PSNR)
                # ref_metrics["SSIM"].append(SSIM)
                # ref_metrics["SAM"].append(SAM)
                # ref_metrics["ERGAS"].append(ERGAS)
                # ref_metrics["SCC"].append(SCC)
                # if sate in ["GF1", "GF2", "GF6", "LC7", "LC8", "WV2", "WV3", "WV4", "QB", "IN"]:
                #     sates[sate]["SAM"].append(SAM)
                #     sates[sate]["ERGAS"].append(ERGAS)
                #     sates[sate]["SCC"].append(SCC)

                psnr = self.psnr(fake_lrms, ms).item()
                ssim = self.ssim(fake_lrms, ms).item()
                mse = self.mse(fake_lrms, ms).item()

                sates[x['sate'][0]]["SAM"].append(SAM)
                sates[x['sate'][0]]["ERGAS"].append(ERGAS)
                sates[x['sate'][0]]["SCC"].append(SCC)

                sates[x['sate'][0]]["MSE"].append(mse)
                sates[x['sate'][0]]["PSNR"].append(psnr)
                sates[x['sate'][0]]["SSIM"].append(ssim)

                sates[x['sate'][0]]["D_lambda"].append(c_D_lambda)
                sates[x['sate'][0]]["D_s"].append(c_D_s)
                sates[x['sate'][0]]["QNR"].append(c_qnr)

                for scene in sate:
                    sates[scene]["PSNR"].append(psnr)
                    sates[scene]["SSIM"].append(ssim)
                    sates[scene]["MSE"].append(mse)
                    sates[scene]["SAM"].append(SAM)
                    sates[scene]["ERGAS"].append(ERGAS)
                    sates[scene]["SCC"].append(SCC)

                    sates[scene]["D_lambda"].append(c_D_lambda)
                    sates[scene]["D_s"].append(c_D_s)
                    sates[scene]["QNR"].append(c_qnr)

                # sates[sate]["PSNR"].append(psnr)
                # sates[sate]["SSIM"].append(ssim)
                # sates[sate]["MSE"].append(mse)
                
                
                
                # sates[sate]["SSIM"].append(ssim)
                # sates[sate]["MSE"].append(mse)
                # ref_metrics["Q"].append(Q)
                # no_ref_metrics["D_lambda"].append(D_lambda)
                # no_ref_metrics["D_s"].append(D_s)
                # no_ref_metrics["QNR"].append(QNR)
        # result = {
            # "Full Resolution": {
            #     "D_lambda": f"{sum(no_ref_metrics['D_lambda'])/len(no_ref_metrics['D_lambda']):.4f}",
            #     "D_s": f"{sum(no_ref_metrics['D_s'])/len(no_ref_metrics['D_s']):.4f}",
            #     "QNR": f"{sum(no_ref_metrics['QNR'])/len(no_ref_metrics['QNR']):.4f}",
            # },
            # "Reduced Resolution": {
            # "PSNR": f"{sum(ref_metrics['PSNR'])/len(ref_metrics['PSNR']):.4f}",
            # "SSIM": f"{sum(ref_metrics['SSIM'])/len(ref_metrics['SSIM']):.4f}",
            # "SAM": f"{sum(ref_metrics['SAM'])/len(ref_metrics['SAM']):.4f}",
            # "ERGAS": f"{sum(ref_metrics['ERGAS'])/len(ref_metrics['ERGAS']):.4f}",
            # "SCC": f"{sum(ref_metrics['SCC'])/len(ref_metrics['SCC']):.4f}",
            # "Q": f"{sum(ref_metrics['Q'])/len(ref_metrics['Q']):.4f}",
            # },
            # "Sates": {
            #     "GF1": {
            #         "SAM": f"{sum(sates['GF1']['SAM'])/len(sates['GF1']['SAM']):.4f}",
            #         "ERGAS": f"{sum(sates['GF1']['ERGAS'])/len(sates['GF1']['ERGAS']):.4f}",
            #         "SCC": f"{sum(sates['GF1']['SCC'])/len(sates['GF1']['SCC']):.4f}",
            #     },
            #     "GF2": {
            #         "SAM": f"{sum(sates['GF2']['SAM'])/len(sates['GF2']['SAM']):.4f}",
            #         "ERGAS": f"{sum(sates['GF2']['ERGAS'])/len(sates['GF2']['ERGAS']):.4f}",
            #         "SCC": f"{sum(sates['GF2']['SCC'])/len(sates['GF2']['SCC']):.4f}",
            #     },
            #     "GF6": {
                
        # }
        result = sates
        for sate in sates:
            result[sate] = {
                "SAM": f"{sum(sates[sate]['SAM'])/len(sates[sate]['SAM']):.4f}",
                "ERGAS": f"{sum(sates[sate]['ERGAS'])/len(sates[sate]['ERGAS']):.4f}",
                "SCC": f"{sum(sates[sate]['SCC'])/len(sates[sate]['SCC']):.4f}",
                "MSE": f"{sum(sates[sate]['MSE']) / len(sates[sate]['MSE']):.9f}",
                "PSNR": f"{sum(sates[sate]['PSNR']) / len(sates[sate]['PSNR']):.4f}",
                "SSIM": f"{sum(sates[sate]['SSIM']) / len(sates[sate]['SSIM']):.4f}",

                "D_lambda": f"{sum(sates[sate]['D_lambda']) / len(sates[sate]['D_lambda']):.9f}",
                "D_s": f"{sum(sates[sate]['D_s']) / len(sates[sate]['D_s']):.4f}",
                "QNR": f"{sum(sates[sate]['QNR']) / len(sates[sate]['QNR']):.4f}",
            }
            # result[sate] = {
            #     "PSNR": f"{sum(sates[sate]['PSNR'])/len(sates[sate]['PSNR']):.4f}",
            #     "SSIM": f"{sum(sates[sate]['SSIM'])/len(sates[sate]['SSIM']):.4f}",
            #     "MSE": f"{sum(sates[sate]['MSE'])/len(sates[sate]['MSE'])*1e4:.4f}",
            # }
        # print(result)
        
        return {self.model_name: result}


if __name__ == "__main__":
    model_name = sys.argv[1]
    eval = Eval(model_name)
    result = eval.evaluate()
    print(result)
