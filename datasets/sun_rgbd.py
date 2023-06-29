import random
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io
import torch
import yaml

# from matplotlib import patches as mpatches
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from skimage import color
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .transform_2d import RGBDTransform

"""TODO
- init transforms
- move setup (metadata) methods from dataset to datamodule?
- move visualize method
- adapt dataset so that it accepts split_type
-
"""


class SunRGBDDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split_type="full",
        rgbd_size=False,
        transform=None,
        verbose=False,
    ):
        super().__init__()
        self.metadata_path = Path(
            str(metadata_path).replace("full", split_type)
        ).resolve()
        self.split_type = split_type
        self.rgbd_size = rgbd_size
        self.verbose = verbose
        self.transform = transform

        self.sample_paths = []
        self.dataset_len = 0

        self._load_metadata()

    def __len__(self):
        if self.dataset_len == 0:
            self.dataset_len = sum(1 for _ in open(self.metadata_path)) - 1
        return self.dataset_len

    def __getitem__(self, idx):
        rgb, depth, mask, sample_dir = self.sample_paths[idx]
        rgb_image = cv2.cvtColor(cv2.imread(str(sample_dir / rgb)), cv2.COLOR_BGR2RGB)
        depth_map = cv2.imread(str(sample_dir / depth), cv2.IMREAD_ANYDEPTH).astype(
            np.float32
        )
        mask = np.load(sample_dir / mask)

        if self.transform:
            return self.transform(rgb_image, depth_map, mask, self.split_type)

        return self.init_transform(rgb_image, depth_map), mask

    def init_transform(self, rgb, depth):
        rgb = cv2.resize(rgb, self.rgbd_size, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, self.rgbd_size, interpolation=cv2.INTER_LINEAR)
        rgbd = np.concatenate(
            (rgb.astype(np.float32) / 255.0, np.expand_dims(depth, axis=-1)), axis=-1
        )
        rgbd = torch.from_numpy(rgbd.transpose((2, 0, 1)))

        return rgbd

    def find_smallest_size(self):
        smallest_size = np.array([np.inf, np.inf])
        for i in range(len(self)):
            rgb, depth, _, sample_dir = self.sample_paths[i]
            rgb_image = cv2.cvtColor(
                cv2.imread(str(sample_dir / rgb)), cv2.COLOR_BGR2RGB
            )
            depth_map = cv2.imread(str(sample_dir / depth), cv2.IMREAD_ANYDEPTH)
            smallest = np.minimum(
                np.array(rgb_image.shape[:2]), np.array(depth_map.shape[:2])
            )
            smallest_size = np.minimum(smallest_size, smallest)
        return smallest_size.astype(int).tolist()

    def _load_metadata(self):
        meta_df = pd.read_csv(self.metadata_path)
        self.sample_paths = list(
            zip(
                meta_df["rgb"].values,
                meta_df["depth"].values,
                meta_df["mask"].values,
                [Path(dir).resolve() for dir in meta_df["sample_dir"].values],
            )
        )
        self.dataset_len = len(self.sample_paths)

    def visualize_random_sample(
        self, cmap: Dict[str, List[float]], show_norm=True, idx=None
    ):
        if idx is None:
            random_idx = random.randint(0, len(self) - 1)
        else:
            random_idx = idx
        rgb, _, _, sample_dir = self.sample_paths[random_idx]
        rgb = cv2.imread(str(sample_dir / rgb)).astype(np.float32) / 255.0
        if show_norm:
            rgbd, mask = self.__getitem__(random_idx)
            rgb_image = rgbd[:3:, :, :].cpu().numpy().transpose(1, 2, 0)
            depth_map = rgbd[3, :, :].cpu().numpy()
            mask = mask.cpu().numpy()
        else:
            raise NotImplementedError()
            rgb_image = self.rgb_images[random_idx]
            depth_map = self.depth_maps[random_idx]
            mask = self.masks[random_idx]

            rgb_image = rgb_image.astype(np.float32) / 255.0
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(
            1, 4, sharex=False, sharey=False, figsize=(18, 6)
        )
        fig.suptitle(
            f"Sample {random_idx} - normalized: {show_norm} - split_type: {self.split_type}"
        )
        ax0.imshow(rgb)
        ax0.set_title("RGB image - Original")
        ax0.set_axis_off()
        ax1.imshow(rgb_image)
        ax1.set_title("RGB image - Augmented")
        ax1.set_axis_off()
        ax2.imshow(depth_map, cmap="hot", interpolation="nearest")
        ax2.set_title("Depth map - Augmented")
        ax2.set_axis_off()

        custom_cmap = mcolors.ListedColormap(list(cmap.values()))

        ax3.imshow(mask, cmap=custom_cmap, interpolation="nearest")
        ax3.set_title("Ground truth - Augmented")
        ax3.set_axis_off()

        plt.tight_layout()
        plt.show()


class SunRGBDDataModule(pl.LightningDataModule):
    DATA_ROOT = Path(".data/")

    # relative to DATA_ROOT
    DATASET_DIR = "SUNRGBD"
    OWN_META_DIR = "SUNRGBD/meta"
    META_PD_FILE = "meta_full.csv"  # in own meta dir
    SUNRGBD_STATS_FILE = "sunrgbd_stats.yml"  # in own meta dir
    ORIGINAL_META_DIR = "SUNRGBDtoolbox/Metadata"  # original meta data

    def __init__(
        self,
        data_root: str = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        resize: float = False,
        transform: transforms = None,
        verbose: bool = False,
        random_seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.resize = resize
        self.transform = transform
        self.verbose = verbose
        self.random_seed = random_seed

        self.num_classes: int = None
        self.class_cmap: Dict[str, List] = None

        self.train_dataset: SunRGBDDataset = None
        self.test_dataset: SunRGBDDataset = None
        self.val_dataset: SunRGBDDataset = None

        self.set_paths(data_root)

        # torch.manual_seed(self.random_seed)
        # random.seed(self.random_seed)

    def set_paths(self, data_root):
        project_root = Path(__file__).resolve().parents[2]
        self.data_root = project_root / (
            Path(data_root) if data_root else self.DATA_ROOT
        )
        assert self.data_root.exists(), "Data root does not exist."
        self.meta_pd_file = self.data_root / self.OWN_META_DIR / self.META_PD_FILE
        self.stats_file = self.data_root / self.OWN_META_DIR / "sunrgbd_stats.yaml"
        self.sunrgbd_meta_dir = self.data_root / self.ORIGINAL_META_DIR
        if self.verbose:
            print(f"Data root directory: {self.data_root}")
            print(f"Metadata directory: {self.sunrgbd_meta_dir}")
            print(f"Metadata file: {self.meta_pd_file}")
            print(f"Stats file: {self.stats_file}")

    def prepare_data(self, get_samples=False, val_size=0.2, test_size=0.05):
        # This method should only be called once or to change the split! TODO
        if self.verbose:
            print("Preparing data...")
            print(f"Parsing metadata from {self.sunrgbd_meta_dir}...")
        cmap, num_classes = self.setup_metadata(get_samples=get_samples)

        dataset_stats = {"class_cmap": cmap, "num_classes": num_classes}

        full_dataset = SunRGBDDataset(
            metadata_path=self.meta_pd_file,
            split_type="full",
            verbose=self.verbose,
        )

        if self.verbose:
            print("Finding the smallest shape...")
        smallest_size = full_dataset.find_smallest_size()
        dataset_stats.update({"smallest_shape": smallest_size})

        # split dataset and save the split to three different csv files
        if self.verbose:
            print("Splitting dataset...")
        self._init_dataset_split(full_dataset, val_size=val_size, test_size=test_size)
        del full_dataset

        if self.verbose:
            print("Computing mean and std...")
        self.train_dataset = SunRGBDDataset(
            metadata_path=self.meta_pd_file,
            split_type="train",
            transform=None,
            rgbd_size=smallest_size,
            verbose=self.verbose,
        )
        dataset_stats.update(
            self.compute_mean_std(train_loader=self.train_dataloader(batch_size=1))
        )
        self.train_dataset = None
        with open(self.stats_file, "w") as f:
            yaml.dump(dataset_stats, f, default_flow_style=False)

    def setup(self, stage="init"):
        if stage == "init":
            with open(self.stats_file, "r") as f:
                stats = yaml.load(f, Loader=yaml.FullLoader)
            self.num_classes = stats["num_classes"]
            self.class_cmap = stats["class_cmap"]
            rgbd_mean = stats["rgbd_mean"]
            rgbd_std = stats["rgbd_std"]
            self.resize = (
                (
                    np.round(
                        np.array(stats["smallest_shape"])
                        * float(self.resize or 1.0)
                        / 32
                    )
                    * 32
                )
                .astype(int)
                .tolist()
            )

            self.transform = RGBDTransform(
                resize=self.resize, mean=rgbd_mean, std=rgbd_std
            )
        elif stage == "fit":
            self.train_dataset = SunRGBDDataset(
                metadata_path=self.meta_pd_file,
                split_type="train",
                rgbd_size=self.resize,
                transform=self.transform,
                verbose=self.verbose,
            )
        elif stage == "validate":
            self.val_dataset = SunRGBDDataset(
                metadata_path=self.meta_pd_file,
                split_type="val",
                rgbd_size=self.resize,
                transform=self.transform,
                verbose=self.verbose,
            )
        if stage == "test":
            self.test_dataset = SunRGBDDataset(
                metadata_path=self.meta_pd_file,
                split_type="test",
                rgbd_size=self.resize,
                transform=self.transform,
                verbose=self.verbose,
            )

    def train_dataloader(self, batch_size=None):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _init_dataset_split(self, dataset, val_size=0.2, test_size=0.05):
        dataset_size = len(dataset)
        test_size = round(test_size * dataset_size)
        val_size = round(val_size * dataset_size)
        train_size = dataset_size - val_size - test_size

        # Split train, validation and test metadata and save them to their own CSV files
        train_data, val_data, test_data = random_split(
            dataset, [train_size, val_size, test_size]
        )

        meta_df = pd.read_csv(self.meta_pd_file)
        train_meta_df = meta_df.loc[train_data.indices]
        val_meta_df = meta_df.loc[val_data.indices]
        test_meta_df = meta_df.loc[test_data.indices]

        if self.verbose:
            print(
                f"Split sizes: {dataset_size}|{len(train_data.indices)}:{len(val_data.indices)}:{len(test_data.indices)}"
            )
            print("full|train:val:test")

        train_meta_df.to_csv(
            Path(str(self.meta_pd_file).replace("full", "train")), index=False
        )
        val_meta_df.to_csv(
            Path(str(self.meta_pd_file).replace("full", "val")), index=False
        )
        test_meta_df.to_csv(
            Path(str(self.meta_pd_file).replace("full", "test")), index=False
        )

    def compute_mean_std(self, train_loader):
        """Compute the mean and std deviation of the training samples."""
        # TODO check if this is correct
        if train_loader.dataset.transform is not None:
            print(
                "Warning: transform is not None! Setting it to None to compute stats..."
            )
            train_loader.dataset.transform = None

        mean = 0.0
        std = 0.0
        n_samples = 0.0
        for rgbd, _ in train_loader:
            batch_samples = rgbd.size(0)
            rgbd = rgbd.view(batch_samples, rgbd.size(1), -1)
            mean += rgbd.mean(2).sum(0)
            std += rgbd.std(2).sum(0)
            n_samples += batch_samples

        mean /= n_samples
        std /= n_samples

        return {"rgbd_mean": mean.tolist(), "rgbd_std": std.tolist()}

    def setup_metadata(self, get_samples=False):
        self.meta_pd_file.parent.mkdir(parents=True, exist_ok=True)
        if get_samples:
            self.save_sample_meta()
        class_names = self.get_class_names()
        num_classes = len(class_names)
        colors = self.get_color_map(num_classes)
        class_mappings = dict(zip(class_names, colors))
        return class_mappings, num_classes

    def save_sample_meta(self):
        sunrgbd_meta = scipy.io.loadmat(
            self.sunrgbd_meta_dir / "SUNRGBDMeta.mat",
            squeeze_me=True,
            struct_as_record=False,
        ).get("SUNRGBDMeta")
        sunrgbd_seg = h5py.File(self.sunrgbd_meta_dir / "SUNRGBD2Dseg.mat", "r")
        masks = sunrgbd_seg.get("SUNRGBD2Dseg").get("seglabel")

        rgb_files = []
        depth_files = []
        mask_files = []
        sample_dirs = []

        for i, sample_meta in enumerate(sunrgbd_meta):
            sample_dir = Path(self.data_root) / sample_meta.sequenceName
            assert sample_dir.exists(), f"{sample_dir} does not exist"
            sample_dirs.append(str(sample_dir))

            rgb_files.append("image/" + sample_meta.rgbname)
            depth_files.append("depth_bfx/" + sample_meta.depthname)

            mask_file = "mask/mask.npy"
            mask_path = sample_dir / Path(mask_file)
            if not mask_path.exists():
                mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.array(sunrgbd_seg[masks[i][0]]).transpose(1, 0)  # (H, W)
            np.save(mask_path, mask)
            mask_files.append(mask_file)

        # Save the DataFrame to a CSV file
        meta_df = pd.DataFrame(
            {
                "sample_dir": sample_dirs,
                "rgb": rgb_files,
                "depth": depth_files,
                "mask": mask_files,
            }
        )
        meta_df.to_csv(self.meta_pd_file, index=False)

    def get_class_names(self):
        classes = list(
            scipy.io.loadmat(
                self.sunrgbd_meta_dir / "seg37list.mat",
                squeeze_me=True,
                struct_as_record=False,
            ).get("seg37list")
        )
        classes.insert(0, "background")
        return classes

    def get_color_map(self, num_classes):
        num_classes = num_classes - 1  # 0 is background
        # Create an array with evenly spaced values in the range [0, 1)
        hues = np.linspace(0, 1, num_classes, endpoint=False)
        hsv_colors = np.zeros((num_classes, 3))
        hsv_colors[:, 0] = hues
        hsv_colors[:, 1] = 1
        hsv_colors[:, 2] = 1
        background = np.array([130] * 3).astype(np.float32) / 255.0

        rgb_colors = color.hsv2rgb(hsv_colors.reshape((num_classes, 1, 3))).reshape(
            (num_classes, 3)
        )
        rgb_colors = np.vstack((background, rgb_colors))

        return rgb_colors.tolist()

    def visualize_random_sample(self, idx=None):
        if self.train_dataset is None:
            self.setup("fit")
        self.train_dataset.visualize_random_sample(
            cmap=self.class_cmap, show_norm=True, idx=idx
        )

    def get_shapes(self):
        self.setup("test")
        rgbd, mask = self.test_dataset[-1]
        self.test_dataset = None
        return rgbd.shape, mask.shape


if __name__ == "__main__":
    datamodule = SunRGBDDataModule(verbose=True, resize=0.5)
    datamodule.prepare_data(get_samples=True)
    datamodule.setup()
    datamodule.visualize_random_sample()
