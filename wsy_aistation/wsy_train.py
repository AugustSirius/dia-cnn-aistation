from __future__ import annotations

import argparse
import datetime
import logging
import os
import yaml
import random
import numpy as np
import pandas as pd
import math
import pickle
import sys

import lightning.pytorch as ptl
from typing import Any
from copy import deepcopy

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import IterableDataset

import torch
from torch import nn
import torch.utils.data as pt_data
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from model import DIArtModel
from dataset import create_iterable_dataset, Dataset
from utils import set_seeds, mkdir_p

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 设置乘法的精度(fast matrix multiplication algorithms )
torch.set_float32_matmul_precision = 'high'

from torch.nn import functional as F


class focal_loss():
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def __call__(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor):
    
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        # logging.info(f"loss1: {loss}")
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'mean', 'sum'"
            )
        loss.requires_grad_(True)
        return loss

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_iter: int,
                 max_iter: int,
                 max_lr: int,
                 min_lr: int,
                 warmup_type: str,
                 base_iter: int = 0,
                 ) -> None:
        """
        Args:
        optimizer: 优化器类型
        warmup_iter: 进行warmup的iter数量
        max_iter: 最大的iter数量
        max_lr: 最大的学习率
        min_lr: 最小的学习率
        base_iter: 预训练的iter数量, 默认为0
        warmup_type: warmup类型（exp：指数；cos：余弦; contant: 常量）

        """
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_type = warmup_type
        
        self.base_iter = base_iter

        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        if self.warmup_type == 'exp':
            lr_factor = self.get_exponential_lr_factor(epoch=self.last_epoch + self.base_iter)
        elif self.warmup_type == 'cos':
            lr_factor = self.get_cosine_lr_factor(epoch=self.last_epoch + self.base_iter)
        else:
            lr_factor = 1.0

        if isinstance(lr_factor, float):
            lr_factor = min(1.0, lr_factor)
        else:
            # when lr_factor is complex, designate lr_factor equal 0 where lr equal min_lr
            lr_factor = 0.0

        return [float(base_lr * lr_factor) if float(base_lr * lr_factor) > self.min_lr else self.min_lr
                for base_lr in self.base_lrs]

    def get_exponential_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup_iter:
            lr_factor *= epoch / self.warmup_iter
        elif epoch <= self.max_iter:
            lr_factor = (1 - (epoch - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** 0.9
        else:
            lr_factor = 0.0
        return lr_factor

    def get_cosine_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup_iter:
            lr_factor *= epoch / self.warmup_iter
        elif epoch <= self.max_iter:
            lr = self.min_lr + \
                 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(epoch / (self.max_iter - self.warmup_iter) * np.pi))
            lr_factor = lr / self.max_lr
        else:
            lr_factor = 0.0
        return lr_factor


class PTModule(ptl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
            self,
            config: dict[str, Any],
            model: DIArtModel,
            sw: SummaryWriter,
            optim: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.sw = sw
        self.optim = optim
        self.scheduler = scheduler

        # 损失函数
        if config["selfAdaptiveTraining"] == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif config["selfAdaptiveTraining"] == 'focal_loss':
            self.loss_fn = focal_loss(alpha=float(config['alpha']),
                                      gamma=float(config['gamma']),
                                      reduction=config['reduction'])

        self.running_loss = None
        self.steps = 0
        self.train_step_scale = config["train_step_scale"]

    def forward(
            self,
            rsm: Tensor,
            frag_info: Tensor,
            feat: Tensor,
    ) -> tuple[Tensor]:
        """Model forward pass."""
        return self.model(rsm, frag_info, feat)

    def training_step(  # need to update this
            self,
            batch: tuple[Tensor, Tensor, Tensor, Tensor, list, list],
    ) -> torch.Tensor:
        """A single training step.

        Args:
            batch (tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.IntTensor, list, list]) :
                A batch of rsm, frag_info, feat, label as torch Tensors, file_name, precursor_id as list.

        Returns:
            torch.FloatTensor: training loss
        """
        try:
            # dataloader
            rsm, frag_info, feat, label, file_name, precursor_id = batch
        except:
            # iterable_dataset
            batch = next(iter(batch))
            rsm, frag_info, feat, label, file_name, precursor_id = batch
        
        rsm = rsm.to(self.device)
        frag_info = frag_info.to(self.device)
        feat = feat.to(self.device)
        label = label.to(self.device)
        
        # logger.info(f'rsm shape: {rsm.shape}')
        # logger.info(f'frag_info shape: {frag_info.shape}')
        # logger.info(f'feat shape: {feat.shape}')
        # logger.info(f'label shape: {label.shape}')

        # pred： (batch , score)
        # truth： (batch , 0/1)
        pred = self.forward(rsm, frag_info, feat)
        loss = self.loss_fn(pred, label)
        self.log('train_loss', loss)

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        # skip first iter
        if ((self.steps + 1) % int(self.train_step_scale)) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]

            self.sw.add_scalar("train/train_loss_raw", loss.item(), self.steps - 1)
            self.sw.add_scalar("train/train_loss_smooth", self.running_loss, self.steps - 1)
            self.sw.add_scalar("optim/lr", lr, self.steps - 1)
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps - 1)

            try:
                # fix gradient vanishing
                for name, layer in self.model.named_parameters():
                    self.sw.add_histogram('weight/' + name + '_data',
                                          layer.cpu().data.to(torch.float).numpy(),
                                          self.steps - 1)

                    if layer.grad is not None:
                        self.sw.add_histogram(
                            f"gradient/{name}",
                            layer.grad.cpu().data.to(torch.float).numpy(),
                            global_step=self.steps - 1)
            except ValueError:
                pass

        self.steps += 1
        return loss

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        self.sw.add_scalar(f"eval/train_loss", self.running_loss, epoch)
        self.running_loss = None
        
        # save model
        torch.save(self.model, os.path.join(self.config['model_save_folder_path'], f"model_epoch{epoch}.pth"))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save config with checkpoint."""
        checkpoint["config"] = self.config
        checkpoint["epoch"] = self.trainer.current_epoch

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        self.config = checkpoint["config"]
        self.optim = checkpoint["optim"]

    def configure_optimizers(
            self,
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        return [self.optim], {"scheduler": self.scheduler, "interval": "step"}

def Optimizers(model, config) -> torch.optim.Optimizer:
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # merge other
    other = param_dict.keys() - (decay | no_decay)
    decay = decay | other

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": float(config["weight_decay"])},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optim = torch.optim.Adam(optim_groups,
                             lr=float(config["learning_rate"]))
    return optim


# flake8: noqa: CR001
def train(
        data_path: str,
        config: dict,
        model_path: str | None = None,
) -> None:
    """Training function."""
    config["tb_summarywriter"] = os.path.join(config["tb_summarywriter"], datetime.datetime.now().strftime(
        "train_%y_%m_%d_%H_%M_%S"
    ))
    sw = SummaryWriter(config["tb_summarywriter"])
    
    # 校验目录（如果没有该目录，则新建）
    mkdir_p(config["tb_summarywriter"])
    mkdir_p(config["out_path"])
    mkdir_p(config["model_save_folder_path"])
    
    logging.info(f"Loading data, GPU nums: {torch.cuda.device_count()}")

    train_dl = create_iterable_dataset(data_path, logging, config)
    logging.info(f"Updates the iter of per epoch is: {len(train_dl):,}"
                 f", optim_weight_part_decay: {bool(config['optim_weight_part_decay']):,}")

    # init model
    if (model_path is not None) and (model_path != ''):
        model = DIArtModel.load(model_path)
    else:
        model = DIArtModel(dropout=float(config["dropout"]),
                           eps=float(config["eps"]))
    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    # Train on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if bool(config["optim_weight_part_decay"]):
        # optimer
        optim = Optimizers(model, config)
    else:
        # optimer
        optim = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )



    # 更新训练集的iter、以及监控参数
    one_epoch_iters = len(train_dl)
    max_iters = config["epochs"] * one_epoch_iters
    warmup_iters = int(float(config["warmup_ratio"]) * max_iters)
    config["train_step_scale"] = int(one_epoch_iters * float(config["train_step_ratio"]))
    config["ckpt_interval"] = one_epoch_iters

    logging.info(f"Updates max_iters of per epoch is : {max_iters:,},"
                 f" warmup_iters={warmup_iters}, "
                 f" ckpt interval={config['ckpt_interval']}")
    logging.info(
        f"Updates train_step_scale is : {config['train_step_scale']:,}")

    # 使用指数衰减方式，进行warmup
    scheduler = WarmupScheduler(optim, warmup_iters, max_iters, float(config['learning_rate']), float(config['min_lr']), config['warmup_strategy'])
    ptmodel = PTModule(config, model, sw, optim, scheduler)

    if config["save_model"]:
        callbacks = [
            ptl.callbacks.ModelCheckpoint(
                dirpath=config["model_save_folder_path"],
                save_top_k=-1,  # 保存所有的模型
                save_weights_only=config["save_weights_only"],
                every_n_train_steps=config["ckpt_interval"],
            ),
        ]
    else:
        callbacks = None

    logging.info("Initializing PL trainer., epoch: {}".format(config["epochs"]))

    if config["train_strategy"] == 'ddp':
        strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
    else:
        strategy = config["train_strategy"]

    trainer = ptl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed", # 混合精度
        callbacks=callbacks,
        max_epochs=config["epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        accumulate_grad_batches=config["grad_accumulation"],
        gradient_clip_val=config["gradient_clip_val"],
        strategy=strategy,
    )

    if config["train_strategy"] in ['deepspeed_stage_1', 'deepspeed_stage_2', 'deepspeed_stage_2_offload'] :
        trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

    try:
        # Train the model.
        trainer.fit(ptmodel, train_dl)
    except Exception as e:
        logging.info("error: {}".format(e))
    finally:
        logging.info("model save !!")
        torch.save(model, config["model_save_folder_path"] + '/msbert.pth')
        
    logging.info("Training complete.")
    sys.exit(0) # 退出程序，返回状态码0


def main() -> None:
    """Train the model."""
    logging.info("Initializing training.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./model.yaml")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)

    config['gpu_num'] = 8
    logging.info(f"config:  {args.config}")

    # 设置全局seed
    set_seeds(config['seed'])

    if config['model_path'] == '':
        model_path = None
    else:
        model_path = args.model_path
        logging.info(f"model_path: {model_path}")

    train(config['data_path'], config, model_path)


if __name__ == "__main__":
    main()
