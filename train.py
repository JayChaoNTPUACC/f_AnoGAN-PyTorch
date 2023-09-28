# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim.swa_utils import AveragedModel
from torch.utils import data

from dataset import BaseDataset, CPUPrefetcher, CUDAPrefetcher
from model import dcnet, discriminator, encodernet
from utils import load_pretrained_state_dict, load_resume_state_dict, make_directory, AverageMeter, Summary, ProgressMeter
import torchvision.transforms


class Trainer(object):
    def __init__(self, config: Any):
        # 运行环境相关参数
        self.project_name = config["PROJECT_NAME"]
        self.exp_name = config["EXP_NAME"] + time.strftime("-%Y%m%d-%H_%M_%S", time.localtime(int(round(time.time() * 1000)) / 1000))
        self.seed = config["SEED"]
        self.mixing_precision = config["MIXING_PRECISION"]
        self.scaler = None  # TODO: 未来支持混合精度训练
        self.device = config["DEVICE"]
        self.cudnn_benchmark = config["CUDNN_BENCHMARK"]

        self.wandb_config = config
        self.wandb_project_name = config["PROJECT_NAME"]

        self.samples_dir = f"./samples/{self.exp_name}"
        self.results_dir = f"./results/{self.exp_name}"
        self.visuals_dir = f"./results/visuals/{self.exp_name}"

        # 模型相关参数
        self.g_model = None
        self.ema_g_model = None
        self.g_model_name = config["MODEL"]["G"]["NAME"]
        self.g_model_latent_dim = config["MODEL"]["G"]["LATENT_DIM"]
        self.g_model_out_channels = config["MODEL"]["G"]["OUT_CHANNELS"]
        self.g_model_channels = config["MODEL"]["G"]["CHANNELS"]
        self.g_model_ema = config["MODEL"]["G"]["EMA"]
        self.g_model_compiled = config["MODEL"]["G"]["COMPILED"]

        self.d_model = None
        self.ema_d_model = None
        self.d_model_name = config["MODEL"]["D"]["NAME"]
        self.d_model_in_channels = config["MODEL"]["D"]["IN_CHANNELS"]
        self.d_model_out_channels = config["MODEL"]["D"]["OUT_CHANNELS"]
        self.d_model_channels = config["MODEL"]["D"]["CHANNELS"]
        self.d_model_ema = config["MODEL"]["D"]["EMA"]
        self.d_model_compiled = config["MODEL"]["D"]["COMPILED"]

        self.e_model = None
        self.ema_e_model = None
        self.e_model_name = config["MODEL"]["E"]["NAME"]
        self.e_model_latent_dim = config["MODEL"]["E"]["LATENT_DIM"]
        self.e_model_in_channels = config["MODEL"]["E"]["IN_CHANNELS"]
        self.e_model_channels = config["MODEL"]["E"]["CHANNELS"]
        self.e_model_ema = config["MODEL"]["E"]["EMA"]
        self.e_model_compiled = config["MODEL"]["E"]["COMPILED"]

        self.ema_avg_fn = None
        self.ema_decay = config["MODEL"]["EMA"]["DECAY"]
        self.ema_compiled = config["MODEL"]["EMA"]["COMPILED"]

        self.pretrained_g_model_weights_path = config["MODEL"]["CHECKPOINT"]["PRETRAINED_G_MODEL_WEIGHTS_PATH"]
        self.pretrained_d_model_weights_path = config["MODEL"]["CHECKPOINT"]["PRETRAINED_D_MODEL_WEIGHTS_PATH"]
        self.pretrained_e_model_weights_path = config["MODEL"]["CHECKPOINT"]["PRETRAINED_E_MODEL_WEIGHTS_PATH"]
        self.resumed_g_model_weights_path = config["MODEL"]["CHECKPOINT"]["RESUME_G_MODEL_WEIGHTS_PATH"]
        self.resumed_d_model_weights_path = config["MODEL"]["CHECKPOINT"]["RESUME_D_MODEL_WEIGHTS_PATH"]
        self.resumed_e_model_weights_path = config["MODEL"]["CHECKPOINT"]["RESUME_E_MODEL_WEIGHTS_PATH"]

        # 数据集相关参数
        self.train_root_dir = config["TRAIN"]["DATASET"]["ROOT_DIR"]
        self.train_batch_size = config["TRAIN"]["HYP"]["IMGS_PER_BATCH"]
        self.train_shuffle = config["TRAIN"]["HYP"]["SHUFFLE"]
        self.train_num_workers = config["TRAIN"]["HYP"]["NUM_WORKERS"]
        self.train_pin_memory = config["TRAIN"]["HYP"]["PIN_MEMORY"]
        self.train_drop_last = config["TRAIN"]["HYP"]["DROP_LAST"]
        self.train_persistent_workers = config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"]

        self.train_data_prefetcher = None

        # 损失函数参数
        self.gan_criterion = None
        self.e_criterion = None

        self.gan_criterion_name = config["TRAIN"]["LOSSES"]["GAN_CRITERION"]["NAME"]
        self.e_criterion_name = config["TRAIN"]["LOSSES"]["E_CRITERION"]["NAME"]

        # 优化器参数
        self.g_optimizer = None
        self.g_optimizer_name = config["TRAIN"]["OPTIMIZER"]["G"]["NAME"]
        self.g_optimizer_lr = config["TRAIN"]["OPTIMIZER"]["G"]["LR"]
        self.g_optimizer_betas = config["TRAIN"]["OPTIMIZER"]["G"]["BETAS"]
        self.g_optimizer_weight_decay = config["TRAIN"]["OPTIMIZER"]["G"]["WEIGHT_DECAY"]
        self.d_optimizer = None
        self.d_optimizer_name = config["TRAIN"]["OPTIMIZER"]["D"]["NAME"]
        self.d_optimizer_lr = config["TRAIN"]["OPTIMIZER"]["D"]["LR"]
        self.d_optimizer_betas = config["TRAIN"]["OPTIMIZER"]["D"]["BETAS"]
        self.d_optimizer_weight_decay = config["TRAIN"]["OPTIMIZER"]["D"]["WEIGHT_DECAY"]
        self.e_optimizer = None
        self.e_optimizer_name = config["TRAIN"]["OPTIMIZER"]["E"]["NAME"]
        self.e_optimizer_lr = config["TRAIN"]["OPTIMIZER"]["E"]["LR"]
        self.e_optimizer_betas = config["TRAIN"]["OPTIMIZER"]["E"]["BETAS"]

        # 学习率调度器参数
        self.g_scheduler = None
        self.d_scheduler = None
        self.e_scheduler = None

        # 训练参数
        self.start_epoch = 0
        self.epochs = config["TRAIN"]["HYP"]["EPOCHS"]
        self.n_critic = config["TRAIN"]["N_CRITIC"]
        self.print_freq = config["TRAIN"]["PRINT_FREQ"]
        self.g_loss = torch.Tensor([0.0])
        self.d_real_loss = torch.Tensor([0.0])
        self.d_fake_loss = torch.Tensor([0.0])
        self.d_loss = torch.Tensor([0.0])
        self.e_image_loss = torch.Tensor([0.0])
        self.e_feature_loss = torch.Tensor([0.0])
        self.e_loss = torch.Tensor([0.0])

        # 训练环境
        make_directory(self.samples_dir)
        make_directory(self.results_dir)
        make_directory(self.visuals_dir)
        self.setup_seed()
        self.setup_mixing_precision()
        self.setup_device()
        self.setup_wandb()
        # 模型
        self.build_models()
        # 数据集
        self.load_datasets()
        # 损失函数
        self.define_loss()
        # 优化器
        self.define_optimizer()
        # 学习率调度器
        self.define_scheduler()
        # 加载模型权重
        self.load_model_weights()

    def setup_seed(self):
        # 固定随机数种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def setup_mixing_precision(self):
        # 初始化混合精度训练方法
        if self.mixing_precision:
            self.scaler = amp.GradScaler()
        else:
            print("Mixing precision training is not enabled.")

    def setup_device(self):
        # 初始化训练的设备名称
        device = "cpu"
        if self.device != "cpu" and self.device != "":
            if not torch.cuda.is_available():
                warnings.warn("No GPU detected, defaulting to `cpu`.")
            else:
                device = self.device
        if self.device == "":
            warnings.warn("No device specified, defaulting to `cpu`.")
        self.device = torch.device(device)

        # 如果输入图像尺寸是固定的，固定卷积算法可以提升训练速度
        if self.cudnn_benchmark:
            cudnn.benchmark = True
        else:
            cudnn.benchmark = False

    def setup_wandb(self):
        # 初始化wandb
        wandb.init(config=self.wandb_config, project=self.wandb_project_name, name=self.exp_name)

    def build_models(self):
        if self.g_model_name == "dcnet":
            self.g_model = dcnet(latent_dim=self.g_model_latent_dim,
                                 out_channels=self.g_model_out_channels,
                                 channels=self.g_model_channels)
        else:
            raise ValueError(f"The `{self.g_model_name}` is not supported.")

        if self.d_model_name == "discriminator":
            self.d_model = discriminator(in_channels=self.d_model_in_channels,
                                         out_channels=self.d_model_out_channels,
                                         channels=self.d_model_channels)
        else:
            raise ValueError(f"The `{self.d_model_name}` is not supported.")

        if self.e_model_name == "encodernet":
            self.e_model = encodernet(latent_dim=self.e_model_latent_dim,
                                      in_channels=self.e_model_in_channels,
                                      channels=self.e_model_channels)
        else:
            raise ValueError(f"The `{self.e_model_name}` is not supported.")

        # 送至指定设备上运行
        self.g_model = self.g_model.to(self.device)
        self.d_model = self.d_model.to(self.device)
        self.e_model = self.e_model.to(self.device)

        if self.ema_decay != 0:
            self.ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                (1 - self.ema_decay) * averaged_model_parameter + self.ema_decay * model_parameter
        if self.ema_g_model:
            self.ema_g_model = AveragedModel(self.g_model, device=self.device, avg_fn=self.ema_avg_fn)
        if self.ema_d_model:
            self.ema_d_model = AveragedModel(self.d_model, device=self.device, avg_fn=self.ema_avg_fn)
        if self.ema_e_model:
            self.ema_e_model = AveragedModel(self.e_model, device=self.device, avg_fn=self.ema_avg_fn)

            # 编译模型
        if config["MODEL"]["G"]["COMPILED"]:
            self.g_model = torch.compile(self.g_model)
        if config["MODEL"]["D"]["COMPILED"]:
            self.d_model = torch.compile(self.d_model)
        if config["MODEL"]["E"]["COMPILED"]:
            self.e_model = torch.compile(self.e_model)
        if config["MODEL"]["EMA"]["COMPILED"]:
            if self.ema_g_model is not None:
                self.ema_g_model = torch.compile(self.ema_g_model)
            if self.ema_d_model is not None:
                self.ema_d_model = torch.compile(self.ema_d_model)
                warnings.warn("Dynamic compilation of discriminator is not recommended, "
                              "and the support on PyTorch2.0.1 version is not good enough.")
            if self.ema_e_model is not None:
                self.ema_e_model = torch.compile(self.ema_e_model)

    def load_datasets(self):
        image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        defect_dataset = BaseDataset(self.train_root_dir, image_transforms)
        defect_dataloader = data.DataLoader(
            defect_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.train_num_workers,
            pin_memory=self.train_pin_memory,
            drop_last=self.train_drop_last,
            persistent_workers=self.train_persistent_workers,
        )

        if self.device.type == "cuda":
            # 将数据加载器替换为CUDA以加速
            self.train_data_prefetcher = CUDAPrefetcher(defect_dataloader, self.device)
        if self.device.type == "cpu":
            # 将数据加载器替换为CPU以加速
            self.train_data_prefetcher = CPUPrefetcher(defect_dataloader)

    def define_loss(self):
        if self.gan_criterion_name == "bce":
            self.gan_criterion = nn.BCELoss()
        else:
            raise NotImplementedError(f"Loss {self.gan_criterion_name} is not supported.")
        if self.e_criterion_name == "mse":
            self.e_criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {self.e_criterion_name} is not supported.")

        self.gan_criterion = self.gan_criterion.to(self.device)
        self.e_criterion = self.e_criterion.to(self.device)

    def define_optimizer(self):
        if self.g_optimizer_name == "adam":
            self.g_optimizer = optim.Adam(self.g_model.parameters(),
                                          self.g_optimizer_lr,
                                          self.g_optimizer_betas,
                                          weight_decay=self.g_optimizer_weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.g_optimizer_name} is not supported.")
        if self.d_optimizer_name == "adam":
            self.d_optimizer = optim.Adam(self.d_model.parameters(),
                                          self.d_optimizer_lr,
                                          self.d_optimizer_betas,
                                          weight_decay=self.d_optimizer_weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.d_optimizer_name} is not supported.")

        if self.e_optimizer_name == "adam":
            self.e_optimizer = optim.Adam(self.e_model.parameters(),
                                          self.e_optimizer_lr,
                                          self.e_optimizer_betas)
        else:
            raise NotImplementedError(f"Optimizer {self.e_optimizer_name} is not supported.")

    def define_scheduler(self):
        pass

    def load_model_weights(self):
        if self.pretrained_g_model_weights_path != "":
            self.g_model = load_pretrained_state_dict(self.g_model, self.pretrained_g_model_weights_path, self.g_model_compiled)
            self.g_model = torch.load(self.pretrained_g_model_weights_path)
            print(f"Loaded `{self.pretrained_g_model_weights_path}` pretrained model weights successfully.")
        if self.pretrained_d_model_weights_path != "":
            self.d_model = load_pretrained_state_dict(self.d_model, self.pretrained_d_model_weights_path, self.d_model_compiled)
            print(f"Loaded `{self.pretrained_d_model_weights_path}` pretrained model weights successfully.")
        if self.pretrained_e_model_weights_path != "":
            self.e_model = load_pretrained_state_dict(self.e_model, self.pretrained_e_model_weights_path, self.e_model_compiled)
            print(f"Loaded `{self.pretrained_e_model_weights_path}` pretrained model weights successfully.")

        if self.resumed_g_model_weights_path != "":
            self.g_model, self.ema_g_model, self.start_epoch, self.g_optimizer, self.g_scheduler = load_resume_state_dict(
                self.g_model,
                self.ema_g_model,
                self.g_optimizer,
                self.g_scheduler,
                self.resumed_g_model_weights_path,
                self.g_model_compiled,
            )
            print(f"Loaded `{self.resumed_g_model_weights_path}` resume model weights successfully.")

        if self.resumed_d_model_weights_path != "":
            self.d_model, self.ema_d_model, self.start_epoch, self.d_optimizer, self.d_scheduler = load_resume_state_dict(
                self.d_model,
                self.ema_d_model,
                self.d_optimizer,
                self.d_scheduler,
                self.resumed_d_model_weights_path,
                self.d_model_compiled,
            )
            print(f"Loaded `{self.resumed_d_model_weights_path}` resume model weights successfully.")

        if self.resumed_e_model_weights_path != "":
            self.e_model, self.ema_e_model, self.start_epoch, self.e_optimizer, self.e_scheduler = load_resume_state_dict(
                self.e_model,
                self.ema_e_model,
                self.e_optimizer,
                self.e_scheduler,
                self.resumed_e_model_weights_path,
                self.e_model_compiled,
            )
            print(f"Loaded `{self.resumed_e_model_weights_path}` resume model weights successfully.")

    def train(self):
        # 将模型调整为训练模式
        self.g_model.train()
        self.d_model.train()
        self.e_model.train()

        for epoch in range(self.start_epoch, self.epochs):
            batch_index = 0
            self.train_data_prefetcher.reset()
            end = time.time()
            batch_data = self.train_data_prefetcher.next()

            # 计算一个epoch的批次数量
            batches = len(self.train_data_prefetcher)
            # 进度条信息
            batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
            data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
            g_losses = AverageMeter("G loss", ":6.6f", Summary.NONE)
            d_losses = AverageMeter("D loss", ":6.6f", Summary.NONE)

            progress = ProgressMeter(batches,
                                     [batch_time, data_time, g_losses, d_losses],
                                     f"Epoch: [{epoch + 1}]")

            while batch_data is not None:
                # 计算加载一个批次数据时间
                data_time.update(time.time() - end)

                tensor = batch_data["tensor"]

                self.train_gan(tensor, batch_index)

                # 统计需要打印的损失
                g_losses.update(self.g_loss.item(), self.train_batch_size)
                d_losses.update(self.d_loss.item(), self.train_batch_size)

                # 计算训练完一个批次时间
                batch_time.update(time.time() - end)
                end = time.time()

                # 保存训练日志
                wandb.log({
                    "iter": batch_index + epoch * batches + 1,
                    "g_loss": self.g_loss,
                    "d_real_loss": self.d_real_loss,
                    "d_fake_loss": self.d_fake_loss,
                    "d_loss": self.d_loss,
                })
                # 打印训练进度
                if self.print_freq <= 0:
                    raise ValueError(f"Invalid value of print_freq: {self.print_freq}, must be greater than 0.")
                if batch_index == 0 or (batch_index + 1) % self.print_freq == 0:
                    progress.display(batch_index + 1)

                # 加载下一个batch_data
                batch_index += 1
                batch_data = self.train_data_prefetcher.next()

            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.g_model.state_dict(),
                    "ema_state_dict": self.ema_g_model.state_dict() if self.ema_g_model is not None else None,
                    "optimizer": self.g_optimizer.state_dict(),
                    "scheduler": self.g_scheduler.state_dict() if self.g_scheduler is not None else None,
                }, f"{self.samples_dir}/g_epoch_{epoch + 1}.pth.tar")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.d_model.state_dict(),
                    "ema_state_dict": self.ema_d_model.state_dict() if self.ema_d_model is not None else None,
                    "optimizer": self.d_optimizer.state_dict(),
                    "scheduler": self.d_scheduler.state_dict() if self.d_scheduler is not None else None,
                }, f"{self.samples_dir}/d_epoch_{epoch + 1}.pth.tar")

        for epoch in range(self.start_epoch, self.epochs):
            batch_index = 0
            self.train_data_prefetcher.reset()
            end = time.time()
            batch_data = self.train_data_prefetcher.next()

            # 计算一个epoch的批次数量
            batches = len(self.train_data_prefetcher)
            # 进度条信息
            batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
            data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
            e_losses = AverageMeter("E loss", ":6.6f", Summary.NONE)

            progress = ProgressMeter(batches,
                                     [batch_time, data_time, e_losses],
                                     f"Epoch: [{epoch + 1}]")

            while batch_data is not None:
                # 计算加载一个批次数据时间
                data_time.update(time.time() - end)

                tensor = batch_data["tensor"]

                self.train_encoder(tensor, batch_index)

                # 统计需要打印的损失
                e_losses.update(self.e_loss.item(), self.train_batch_size)

                # 计算训练完一个批次时间
                batch_time.update(time.time() - end)
                end = time.time()

                # 保存训练日志
                wandb.log({
                    "iter": batch_index + epoch * batches + 1,
                    "e_image_loss": self.e_image_loss,
                    "e_feature_loss": self.e_feature_loss,
                    "e_loss": self.e_loss,
                })
                # 打印训练进度
                if self.print_freq <= 0:
                    raise ValueError(f"Invalid value of print_freq: {self.print_freq}, must be greater than 0.")
                if batch_index == 0 or (batch_index + 1) % self.print_freq == 0:
                    progress.display(batch_index + 1)

                # 加载下一个batch_data
                batch_index += 1
                batch_data = self.train_data_prefetcher.next()

            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.e_model.state_dict(),
                    "ema_state_dict": self.ema_e_model.state_dict() if self.ema_e_model is not None else None,
                    "optimizer": self.e_optimizer.state_dict(),
                    "scheduler": self.e_scheduler.state_dict() if self.e_scheduler is not None else None,
                }, f"{self.samples_dir}/e_epoch_{epoch + 1}.pth.tar")

    def train_gan(self, real_samples, batch_index: int):
        real = real_samples.to(self.device, non_blocking=True)

        # 随机生成噪声数据
        noise = torch.randn(self.train_batch_size, self.g_model_latent_dim, 1, 1).to(self.device, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        # 生成一张虚假样本
        fake = self.g_model(noise)
        # 判别器计算真实样本和虚假样本的值
        real_output = self.d_model(real, only_features=False)
        fake_output = self.d_model(fake.detach(), only_features=False)

        # 计算判别器损失
        d_real_loss = self.gan_criterion(real_output, torch.ones_like(real_output).to(self.device))
        d_fake_loss = self.gan_criterion(fake_output, torch.zeros_like(fake_output).to(self.device))
        d_loss = d_real_loss + d_fake_loss

        # 更新判别器参数
        d_loss.backward()
        self.d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        if batch_index % self.n_critic == 0:
            self.g_optimizer.zero_grad()

            # 生成一张虚假样本
            fake = self.g_model(noise)
            # 判别器计算虚假样本的值
            fake_output = self.d_model(fake.detach())

            # 计算生成器损失
            g_loss = self.gan_criterion(fake_output, torch.ones_like(fake_output).to(self.device))

            # 更新生成器参数
            g_loss.backward()
            self.g_optimizer.step()

            self.g_loss = g_loss
        self.d_real_loss = d_real_loss
        self.d_fake_loss = d_fake_loss
        self.d_loss = d_loss

    def train_encoder(self, real_samples, batch_index: int):
        self.g_model.eval()
        self.d_model.eval()

        real = real_samples.to(self.device, non_blocking=True)
        # ---------------------
        #  Train Encoder
        # ---------------------
        self.e_optimizer.zero_grad()

        # 由Encoder生成噪声输入
        noise = self.e_model(real)

        # 生成一张虚假样本
        fake = self.g_model(noise)
        # 判别器计算真实样本和虚假样本的值
        real_features = self.d_model(real, only_features=True)
        fake_features = self.d_model(fake, only_features=True)

        # 计算编码器损失
        e_image_loss = self.e_criterion(fake, real)
        e_feature_loss = self.e_criterion(fake_features, real_features)
        e_loss = e_image_loss + e_feature_loss

        # 更新编码器参数
        e_loss.backward()
        self.e_optimizer.step()

        self.e_image_loss = e_image_loss
        self.e_feature_loss = e_feature_loss
        self.e_loss = e_loss


if __name__ == "__main__":
    # 通过命令行参数读取配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/f_anogan-yb.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    trainer = Trainer(config)
    trainer.train()

    # 结束wandb
    wandb.finish()
