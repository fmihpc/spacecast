# Third-party
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

# Local
from . import utils


class SchedulerFM(pl.Callback):
    def __init__(self, args, dirpath):
        super().__init__()
        self.args = args
        self.scheduler_epochs = args.scheduler_epochs
        self.total_ar_steps = args.ar_steps_train
        self.dirpath = dirpath

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]["lr"]
        current_ar_steps = trainer.datamodule.ar_steps_train

        if not self.scheduler_epochs:
            return

        e1 = self.scheduler_epochs[0]
        e2 = (
            self.scheduler_epochs[1]
            if len(self.scheduler_epochs) > 1
            else self.args.epochs
        )

        # lower lr before increasing ar step
        if current_epoch == e1:
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr / 10.0
            trainer.save_checkpoint(f"{self.dirpath}/before_ar.ckpt")

        # ramping up ar steps
        if e1 <= current_epoch <= e2:
            progress = (current_epoch - e1) / max(1, e2 - e1)
            new_ar_steps = 1 + int(progress * (self.total_ar_steps - 1))
            if new_ar_steps != current_ar_steps:
                trainer.datamodule.update_ar_step(new_ar_steps)

        # divergence loss
        if len(self.scheduler_epochs) > 1 and current_epoch == e2:
            pl_module.div_weight = self.args.div_weight
            trainer.save_checkpoint(f"{self.dirpath}/before_div.ckpt")

        utils.rank_zero_print(
            f"\n\nlr: {optimizer.param_groups[0]['lr']}, "
            f"ar_steps: {trainer.datamodule.ar_steps_train}, "
            f"div_weight: {pl_module.div_weight}"
        )


class SchedulerEFM(pl.Callback):
    def __init__(self, args, dirpath):
        super().__init__()
        self.args = args
        self.scheduler_epochs = args.scheduler_epochs
        self.total_ar_steps = args.ar_steps_train
        self.dirpath = dirpath

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]["lr"]
        current_ar_steps = trainer.datamodule.ar_steps_train

        if not self.scheduler_epochs:
            return

        # kulbach-leibler
        if current_epoch == self.scheduler_epochs[0]:
            pl_module.kl_beta = self.args.kl_beta
            trainer.save_checkpoint(f"{self.dirpath}/before_kl.ckpt")

        # ramping up ar steps
        if len(self.scheduler_epochs) > 1:
            e1 = self.scheduler_epochs[1]
            e2 = (
                self.scheduler_epochs[2]
                if len(self.scheduler_epochs) > 2
                else self.args.epochs
            )

            # lower lr before increasing ar step
            if current_epoch == e1:
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr / 10.0
                trainer.save_checkpoint(f"{self.dirpath}/before_ar.ckpt")

            # increast ar steps
            if e1 <= current_epoch <= e2:
                progress = (current_epoch - e1) / max(1, (e2 - e1))
                new_ar_steps = 1 + int(progress * (self.total_ar_steps - 1))
                if new_ar_steps != current_ar_steps:
                    trainer.datamodule.update_ar_step(new_ar_steps)

        # crps loss
        if len(self.scheduler_epochs) > 2 and current_epoch == self.scheduler_epochs[2]:
            pl_module.crps_weight = self.args.crps_weight
            trainer.save_checkpoint(f"{self.dirpath}/before_crps.ckpt")

        # divergence loss
        if len(self.scheduler_epochs) > 3 and current_epoch == self.scheduler_epochs[3]:
            pl_module.div_weight = self.args.div_weight
            trainer.save_checkpoint(f"{self.dirpath}/before_div.ckpt")

        utils.rank_zero_print(
            f"\n\nlr: {optimizer.param_groups[0]['lr']}, "
            f"kl_beta: {pl_module.kl_beta}, "
            f"ar_steps: {trainer.datamodule.ar_steps_train}, "
            f"crps_weight: {pl_module.crps_weight}, "
            f"div_weight: {pl_module.div_weight}"
        )
