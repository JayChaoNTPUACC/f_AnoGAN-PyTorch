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
import os
import queue
import threading

import cv2
import numpy as np
import torch
import torchvision
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, root: str, transform: torchvision.transforms = torchvision.transforms.ToTensor) -> None:
        """基础数据集类

        Args:
            root (str): 数据集根目录
            transform (torchvision.transforms): 数据集转换方法，基于torchvision.transforms

        Attributes:
            image_file_paths (List[str]): 图像文件路径列表

        Examples:
            >>> dataset = BaseDataset(root="./dataset", transforms=torchvision.transforms.ToTensor())
            >>> dataset[0]
            {'tensor': tensor([[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],]]}
        """
        super(BaseDataset, self).__init__()
        self.transform = transform

        # 检查文件夹下是否有文件
        if os.listdir(root) == 0:
            raise RuntimeError("Dataset folder is empty.")

        # 获取文件夹下所有文件路径
        image_file_names = natsorted(os.listdir(root))
        self.image_file_paths = [os.path.join(root, image_file_name) for image_file_name in image_file_names]

    def __getitem__(
            self,
            batch_index: int
    ) -> [Tensor, Tensor]:
        # 读取批次图像
        image = cv2.imread(self.image_file_paths[batch_index]).astype(np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)

        return {"tensor": tensor}

    def __len__(self) -> int:
        return len(self.image_file_paths)


class PrefetchGenerator(threading.Thread):
    """借助PyTorch队列功能生成数据生成器

    Args:
        generator (Generator): 生成器
        num_data_prefetch_queue (int): 预加载数据队列长度
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """借助PyTorch队列功能生成DataLoader预加载器

    Args:
        num_data_prefetch_queue (int): 预加载数据队列长度
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """CPU版本的数据预加载器

    Args:
        dataloader (DataLoader): PrefetchDataLoader预加载器
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """CUDA版本的数据预加载器

    Args:
        dataloader (DataLoader): PrefetchDataLoader预加载器
        device (torch.device): 设备类型
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
