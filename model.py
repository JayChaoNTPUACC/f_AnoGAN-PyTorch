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
import torch
from torch import Tensor, nn

from utils import weights_init

__all__ = [
    "DCNet", "Discriminator", "EncoderNet", "GradientPenaltyLoss",
    "dcnet", "discriminator", "encodernet", "gradient_penalty_loss"
]


class DCNet(nn.Module):
    def __init__(self, latent_dim: int = 100, out_channels: int = 3, channels: int = 64) -> None:
        super(DCNet, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, channels * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

        # 初始化模型权重
        weights_init(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels=1, channels: int = 64) -> None:
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(channels, int(channels * 2), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(channels * 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 2), int(channels * 4), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(channels * 4)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 4), int(channels * 8), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(channels * 8)),
            nn.LeakyReLU(0.2, True),

        )
        self.classifier = nn.Sequential(
            nn.Conv2d(int(channels * 8), out_channels, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

        # 初始化模型权重
        weights_init(self.modules())

    def forward(self, x: Tensor, only_features: bool = False) -> Tensor:
        """判别器推理模式

        Args:
            x (Tensor): 输入数据
            only_features (bool, optional): 是否只输出特征. 默认: ``False``
        """

        if only_features:
            x = self.features(x)
        else:
            x = self.features(x)
            x = self.classifier(x)

        return x


class EncoderNet(nn.Module):
    def __init__(self, latent_dim: int = 100, in_channels: int = 3, channels: int = 64) -> None:
        super(EncoderNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(channels, int(channels * 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(channels * 2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int(channels * 2), int(channels * 4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(channels * 4)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int(channels * 4), int(channels * 8), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(channels * 8)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int(channels * 8), latent_dim, 4, 1, 0, bias=True),
            nn.Tanh(),
        )

        # 初始化模型权重
        weights_init(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)

        return x


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        """PyTorch实现GradientPenalty损失，以避免训练GAN过程中出现“模型崩塌”问题"""
        super(GradientPenaltyLoss, self).__init__()

    @staticmethod
    def forward(model: nn.Module, source: Tensor, target: Tensor) -> Tensor:
        """计算梯度惩罚损失

        Args:
            model (nn.Module): 鉴别器模型
            source (Tensor): 真实样本
            target (Tensor): 虚假样本
        """

        # 真伪样本间内插的随机权重项
        alpha = torch.rand(*source.shape[:2], 1, 1).to(target.device)

        # 获取真实和虚假样本之间的随机内插
        interpolates = (alpha * source + (1 - alpha) * target)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = model(interpolates)
        fake = torch.ones(*d_interpolates.shape).to(source.device)

        # 获取梯度
        gradients = torch.autograd.grad(outputs=d_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=fake,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


def dcnet(**kwargs) -> DCNet:
    """DCGAN的生成器

    Args:
        **kwargs: 参考``DCNet``

    Returns:
        DCNet: DCGAN的生成器
    """
    model = DCNet(**kwargs)

    return model


def discriminator(**kwargs) -> Discriminator:
    """DCGAN的鉴别器

    Args:
        **kwargs: 参考``Discriminator``

    Returns:
        Discriminator: DCGAN的鉴别器
    """

    model = Discriminator(**kwargs)
    return model


def encodernet(**kwargs) -> EncoderNet:
    """DCGAN的编码器

    Args:
        **kwargs: 参考``EncoderNet``

    Returns:
        EncoderNet: DCGAN的编码器
    """

    model = EncoderNet(**kwargs)
    return model


def gradient_penalty_loss() -> GradientPenaltyLoss:
    """PyTorch实现GradientPenalty损失，以避免训练GAN过程中出现“模型崩塌”问题

    Returns:
        GradientPenaltyLoss: PyTorch实现GradientPenalty损失
    """
    return GradientPenaltyLoss()
