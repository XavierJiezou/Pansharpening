from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PNSR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM

# from torchmetrics.image.sam import SpectralAngleMapper as SAM
# from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
# from torchmetrics.image.d_lambda import SpectralDistortionIndex as D_lambda


class PanBenchLitModule(LightningModule):
    """A `LightningModule` for PanBench pansharpening."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize.

        Args:
            net (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
            compile (bool): New in PyTorch 2.0.0. If True, the model will be compiled for training boosting.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging psnr across batches
        self.train_psnr = PNSR(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        self.val_psnr = PNSR(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        self.test_psnr = PNSR(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        # self.train_psnr = MeanMetric()
        # self.val_psnr = MeanMetric()
        # self.test_psnr = MeanMetric()

        # metric objects for calculating and averaging ssim across batches
        self.train_ssim = SSIM(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        self.val_ssim = SSIM(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        self.test_ssim = SSIM(data_range=1.0, reduction="elementwise_mean", dim=[1, 2, 3])
        # self.train_ssim = MeanMetric()
        # self.val_ssim = MeanMetric()
        # self.test_ssim = MeanMetric()

        # metric objects for calculating and averaging sam across batches
        # self.train_sam = SAM()
        # self.val_sam = SAM()
        # self.test_sam = SAM()

        # metric objects for calculating and averaging ergas across batches
        # self.train_ergas = ERGAS()
        # self.val_ergas = ERGAS()
        # self.test_ergas = ERGAS()

        # metric objects for calculating and averaging d_lambda across batches
        # self.train_d_lambda = D_lambda()
        # self.val_d_lambda = D_lambda()
        # self.test_d_lambda = D_lambda()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_ssim_best = MaxMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): A tensor of multispectral images.
            y (torch.Tensor): A tensor of panchromatic images.

        Returns:
            torch.Tensor: A tensor of pansharpened iamges.
        """
        return self.net(x, y)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()
        # self.val_sam.reset()
        # self.val_ergas.reset()
        # self.val_d_lambda.reset()
        self.val_ssim_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (in order):
                - A tensor of losses.
                - A tensor of predictions.
                - A tensor of target labels.
        """
        preds = self.forward(batch["lrms"], batch["lrpan"])
        loss = self.criterion(preds, batch["ms"])
        return loss, preds, batch["ms"]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_psnr(preds, targets)
        self.train_ssim(preds, targets)
        # self.train_sam(preds, targets)
        # self.train_ergas(preds, targets)
        # self.train_d_lambda(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/psnr", self.train_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ssim", self.train_ssim, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "train/sam", self.train_sam, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "train/ergas", self.train_ergas, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "train/d_lambda",
        #     self.train_d_lambda,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        # return loss or backpropagation will fail
        return loss

    # def on_train_epoch_end(self) -> None:
    #     "Lightning hook that is called when a training epoch ends."
    #     pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_psnr(preds, targets)
        self.val_ssim(preds, targets)
        # self.val_sam(preds, targets)
        # self.val_ergas(preds, targets)
        # self.val_d_lambda(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/sam", self.val_sam, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "val/ergas", self.val_ergas, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "val/d_lambda",
        #     self.val_d_lambda,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        ssim = self.val_ssim.compute()  # get current val acc
        self.val_ssim_best(ssim)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/ssim_best", self.val_ssim_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            batch_idx (int): The index of the current batch.
        """

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_psnr(preds, targets)
        self.test_ssim(preds, targets)
        # self.test_sam(preds, targets)
        # self.test_ergas(preds, targets)
        # self.test_d_lambda(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", self.test_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", self.test_ssim, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/sam", self.test_sam, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "test/ergas", self.test_ergas, on_step=False, on_epoch=True, prog_bar=True
        # )
        # self.log(
        #     "test/d_lambda",
        #     self.test_d_lambda,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    # def on_test_epoch_end(self) -> None:
    #     """Lightning hook that is called when a test epoch ends."""
    #     pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PanBenchLitModule(None, None, None, None)
