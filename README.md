# DeResUNet

DeResUNet (DualEncodeResidualUNet) is a deep learning model for semantic segmentation tasks. The project is implemented in PyTorch and uses the SunRGBD dataset for training and evaluation.

## Dataset

The model is trained on the SunRGBD dataset, which includes over 10,000 RGB-D images. The dataset provides 2D polygons and 3D bounding box annotations.

## Model Architecture

The DeResUNet model is composed of two parallel ResNet50 encoders, a bottleneck layer, and a decoder that uses PixelShuffle for upsampling.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Assessing out-of-domain generalization for robust building damage detection](https://www.researchgate.net/publication/346089961_Assessing_out-of-domain_generalization_for_robust_building_damage_detection)
- [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158v2.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://www.linkedin.com/pulse/ways-improve-your-deep-learning-model-batch-adam-albuquerque-lima/)
- [Getting started with Albumentation](https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8)

## License

This project is open source and available under the [MIT License](LICENSE).
