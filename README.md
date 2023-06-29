# DeResUNet

![DeResUNet](https://dopbs.twimg.com/media/Fv1EjNuakAE44z_.jpg:large)

DeResUNet is a deep learning model for semantic segmentation tasks. The project is implemented in PyTorch and uses the SunRGBD dataset for training and evaluation.

## Overview

The project includes the following components:

- DeResUNet: The main model architecture, which is a variant of the U-Net model with residual connections.
- BatchNorm2D & PixelShuffle: Key operations used in the model.
- Albumentations: A library used for data augmentation.
- PyTorch Lightning: A lightweight PyTorch wrapper for high-performance AI research.

## Dataset

The model is trained on the SunRGBD dataset, which includes over 10,000 RGB-D images. The dataset provides 2D polygons and 3D bounding box annotations.

## Model Architecture

The DeResUNet model is composed of two parallel ResNet50 encoders, a bottleneck layer, and a decoder that uses PixelShuffle for upsampling.

## Training

The model is trained with a batch size of 6, an initial learning rate of 1e-4, and uses the ReduceLROnPlateau learning rate scheduler. Training is stopped early if the validation loss does not improve for a certain number of epochs.

## Performance

The model's performance is evaluated using the Intersection over Union (IoU) metric. The IoU values and loss values during training are logged for analysis.

## Usage

To train the model, run the `train.ipynb` notebook. Make sure to install the dependencies listed in `requirements-linux.txt`.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Assessing out-of-domain generalization for robust building damage detection](https://www.researchgate.net/publication/346089961_Assessing_out-of-domain_generalization_for_robust_building_damage_detection)
- [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158v2.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://www.linkedin.com/pulse/ways-improve-your-deep-learning-model-batch-adam-albuquerque-lima/)
- [Getting started with Albumentation](https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8)

## License

This project is open source and available under the [MIT License](LICENSE).
