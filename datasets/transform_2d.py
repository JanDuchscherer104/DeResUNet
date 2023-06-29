from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2


class RGBDTransform:
    def __init__(
        self,
        resize: Tuple[int, int],
        mean: List[float],
        std: List[float],
    ):
        # default: 480, 640 for NYUv2
        self.height, self.width = resize
        self.mean = mean
        self.std = std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device != "cuda":
            print("Warning: Using CPU for data augmentation.")

        self.tf_resize_mask = A.Resize(
            width=self.width, height=self.height, interpolation=cv2.INTER_NEAREST
        )

        self.tf_augment_rgb = A.Compose(
            [
                A.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.95
                ),
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=0.5),
                        A.ColorJitter(p=0.8),
                    ],
                    p=0.8,
                ),
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.6),
                A.GaussianBlur(p=0.4),
                A.RandomGamma(p=0.6),
            ]
        )
        self.tf_augment = A.Compose(
            [
                A.RandomCrop(
                    width=int(self.width * 0.5), height=int(self.height * 0.5), p=0.45
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomScale(scale_limit=0.2, p=1),
                A.ElasticTransform(alpha=32, sigma=200, alpha_affine=120 * 0.03, p=1),
                A.OpticalDistortion(p=1, shift_limit=0.2),
                A.GridDistortion(num_steps=2, distort_limit=0.2, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1
                ),
                A.RandomRotate90(p=0.25),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            ],
        )
        self.tf_norm_and_resize = A.Compose(
            [
                A.Resize(
                    width=self.width, height=self.height, interpolation=cv2.INTER_LINEAR
                ),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

    def __call__(self, rgb, depth, mask, split_type):
        # augment
        if split_type == "train":
            rgb = self.tf_augment_rgb(image=rgb)["image"]
            # Concatenate the depth channel as the fourth channel 3xHxW, HxW -> 4xHxW
            rgb = rgb.astype(np.float32) / 255.0
            rgbd = np.concatenate((rgb, np.expand_dims(depth, axis=-1)), axis=-1)
            augmented = self.tf_augment(image=rgbd, mask=mask)
            rgbd, mask = augmented["image"], augmented["mask"]
        else:
            rgb = rgb.astype(np.float32) / 255.0
            rgbd = np.concatenate((rgb, np.expand_dims(depth, axis=-1)), axis=-1)

        rgbd = self.tf_norm_and_resize(image=rgbd)["image"]
        mask = self.tf_resize_mask(image=mask)["image"]  # use INTER_NEAREST

        mask = torch.from_numpy(mask).long()

        return rgbd.to(self.device), mask.to(self.device)
