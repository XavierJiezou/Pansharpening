from src.models.panbench_module import PanBenchLitModule
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl

from src.models.components.unet import UNet
from src.models.components.diffusion import GaussianDiffusion, make_beta_schedule


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def samples_fn(model, diffusion, shape):
    samples = diffusion.p_sample_loop(model=model, shape=shape, noise_fn=torch.randn)
    return {"samples": (samples + 1) / 2}


def progressive_samples_fn(model, diffusion, shape, device, include_x0_pred_freq=50):
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
    )
    return {
        "samples": (samples + 1) / 2,
        "progressive_samples": (progressive_samples + 1) / 2,
    }


def bpd_fn(model, diffusion, x):
    total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(
        model=model, x_0=x, clip_denoised=True
    )

    return {
        "total_bpd": total_bpd_b,
        "terms_bpd": terms_bpd_bt,
        "prior_bpd": prior_bpd_b,
        "mse": mse_bt,
    }


def validate(val_loader, model, diffusion):
    model.eval()
    bpd = []
    mse = []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(val_loader)):
            x = x
            metrics = bpd_fn(model, diffusion, x)

            bpd.append(metrics["total_bpd"].view(-1, 1))
            mse.append(metrics["mse"].view(-1, 1))

        bpd = torch.cat(bpd, dim=0).mean()
        mse = torch.cat(mse, dim=0).mean()

    return bpd, mse


class DiffusionLitModule(PanBenchLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__(net, optimizer, scheduler, compile)
        self.betas = make_beta_schedule()
        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type="xstart",  # 'xstart': x_0, 'eps': noise
            model_var_type="fixedlarge",
            loss_type="mse",
        )  # what is model_mean_type and model_var_type?

    def forward(self, ms, pan):
        return self.diffusion.p_sample_loop(self.net, ms, pan)

    def model_step(self, batch):
        lrms = batch["lrms"]
        lrpan = batch["lrpan"]
        ms = batch["ms"]
        time = (torch.rand(ms.shape[0]) * 1000).type(torch.int64).to(ms.device)
        preds = self.diffusion.training_losses(self.net, ms, time, lrms, lrpan)
        loss = self.criterion(preds, ms)
        return loss, preds, ms


    def validation_step(self, batch, batch_idx):
        lrms = batch["lrms"]
        lrpan = batch["lrpan"]
        targets = batch["ms"]
        preds = self.forward(lrms, lrpan)
        loss = self.criterion(preds, targets)
        
        self.val_loss(loss)
        self.val_psnr(preds, targets)
        self.val_ssim(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        lrms = batch["lrms"]
        lrpan = batch["lrpan"]
        targets = batch["ms"]
        preds = self.forward(lrms, lrpan)
        loss = self.criterion(preds, targets)
        
        self.test_loss(loss)
        self.test_psnr(preds, targets)
        self.test_ssim(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", self.test_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", self.test_ssim, on_step=False, on_epoch=True, prog_bar=True)
    


if __name__ == "__main__":
    pass
