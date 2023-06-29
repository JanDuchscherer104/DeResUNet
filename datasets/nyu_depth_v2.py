import os
import random
from copy import deepcopy

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from torch.utils.data import Dataset

from .transform_2d import RGBDTransform


class NYUDepthV2Dataset(Dataset):
    """
    Link to the dataset: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
    TODO
        - merge classes
        - fix mappping id to class_name
        - add augmentation for train_set
        - add steps in transform pipeline
    """

    __file_dir = os.path.dirname(os.path.abspath(__file__))
    __data_dir = os.path.join(__file_dir, os.pardir, ".data")
    __set_name = "NYUDepthDatasetV2"
    __set_file = "nyu_depth_v2_masked.mat"

    def __init__(
        self,
        data_path=None,
        split_type="full",
        transform=None,
        resize=False,
        verbose=True,
        random_seed=42,
        num_samples=10,
    ):
        super().__init__()
        self.data_path = os.path.abspath(
            data_path
            if data_path
            else os.path.join(self.__data_dir, self.__set_name, self.__set_file)
        )
        self.split_type = split_type
        self.transform = transform
        self.verbose = verbose
        self.random_seed = random_seed
        # random.seed(self.random_seed)
        self.num_samples = num_samples  # TODO

        # Load the .mat f
        with h5py.File(self.data_path, "r") as f:
            # HxWx3XN -> NxHxWx3
            self.rgb_images = np.transpose(
                f["images"][: self.num_samples].astype(np.uint8),
                (0, 3, 2, 1),
            )
            self.depth_maps = np.transpose(f["depths"][: self.num_samples], (0, 2, 1))
            self.masks = np.transpose(
                f["labels"][: self.num_samples].astype(np.int64), (0, 2, 1)
            )  # int16 would be enough; int64 is for compatibility with torch
            chr_arr = [list(f[ref][()].flatten()) for ref in f["names"][0]]
            self.class_names = ["".join(chr(c) for c in name) for name in chr_arr]

        if self.transform:
            size = np.array(self.rgb_images.shape[1:3])
            if resize:
                # both dimensions must be divisible by 32 since UNet decreases the spatial dimensions by 2^5
                size = tuple((np.round(size * resize / 32) * 32).astype(np.int32))

            std = np.divide(
                np.std(self.rgb_images.reshape(-1, 3), axis=0), 255
            ).tolist()
            mean = np.divide(
                np.mean(self.rgb_images.reshape(-1, 3), axis=0), 255
            ).tolist()
            std_depth = np.std(self.depth_maps.reshape(-1)).item()
            mean_depth = np.mean(self.depth_maps.reshape(-1)).item()
            std.append(std_depth)
            mean.append(mean_depth)

            self.transform = RGBDTransform(
                num_classes=self.num_classes, resize=size, mean=mean, std=std
            )

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        depth_map = self.depth_maps[idx]
        mask = self.masks[idx]

        sample = self.transform(rgb_image, depth_map, mask, self.split_type)

        return sample

    @property
    def num_classes(self):
        return len(self.class_names) + 1

    @property
    def rgbd_shape(self):
        return self[-1][0].shape

    @property
    def mask_shape(self):
        return self[-1][1].shape

    def __repr__(self) -> str:
        repr = f"""
        NYUDepthV2Dataset(split={self.split_type}, data_path={self.data_path})
        Number of samples: {len(self)}
        Number of classes: {self.num_classes}
        RGB image shape: {self.rgb_images.shape}, dtype: {self.rgb_images.dtype}
        Depth map shape: {self.depth_maps.shape}, dtype: {self.depth_maps.dtype}
        Mask shape: {self.masks.shape}, dtype: {self.masks.dtype}
        RGBD sample shape: {self.rgbd_shape}, dtype: {self[-1][0].dtype}
        Mask shape: {self.mask_shape}, dtype: {self[-1][1].dtype}"""
        return repr

    def visualize_random_sample(self, show_norm=False):
        random_idx = random.randint(0, len(self) - 1)
        if show_norm:
            rgbd, mask = self.__getitem__(random_idx)
            rgb_image = rgbd[:3:, :, :].cpu().numpy().transpose(1, 2, 0)
            depth_map = rgbd[3, :, :].cpu().numpy()
            print(mask.shape)
            mask = torch.argmax(mask, dim=0).cpu().numpy()
            print(mask.shape)
        else:
            rgb_image = self.rgb_images[random_idx]
            depth_map = self.depth_maps[random_idx]
            mask = self.masks[random_idx]

            rgb_image = rgb_image.astype(np.float32) / 255.0
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )

        # uique_masks = np.unique(mask.flatten())
        # print(
        #     "Found the following masks:\n",
        #     [self.mask_id2name(e) for e in uique_masks],

        fig, (ax0, ax1, ax2) = plt.subplots(
            1, 3, sharex=False, sharey=False, figsize=(18, 6)
        )
        fig.suptitle(f"Sample {random_idx} - normalized: {show_norm}")
        ax0.imshow(rgb_image)
        ax0.set_title("RGB image")
        ax0.set_axis_off()
        ax1.imshow(depth_map, cmap="hot", interpolation="nearest")
        ax1.set_title("Depth map")
        ax1.set_axis_off()
        # im = ax2.imshow(mask, cmap=cmap, norm=norm)
        ax2.imshow(color.label2rgb(mask))
        ax2.set_title("Ground truth")
        ax2.set_axis_off()

        plt.show()

    def split_dataset(self, split=0.8):
        """Split the dataset into two parts with the specified ratio."""
        if split > 1 or split < 0:
            raise ValueError("Split ratio must be between 0 and 1.")
        num_samples = len(self)
        num_samples_train = int(num_samples * split)
        num_samples_val = num_samples - num_samples_train
        train_set, val_set = torch.utils.data.random_split(
            self,
            [num_samples_train, num_samples_val],
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        # Set split_type attributes for train_set and val_set
        train_set.dataset = deepcopy(train_set.dataset)
        train_set.dataset.split_type = "train"
        val_set.dataset = deepcopy(val_set.dataset)
        val_set.dataset.split_type = "val"
        return train_set, val_set

    def mask_name2id(self, mask_name):
        return self.class_names.index(mask_name)

    def mask_id2name(self, mask_id):
        return self.class_names[mask_id]
