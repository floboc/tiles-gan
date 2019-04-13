# Generating 2D map tiles with Generative Adversarial Network (GAN)

## Introduction

This repository contains all the source code used to generate small 2D tiles commonly used in video games for building maps.
Generating tiles was achieved using a **Deep Convolutional Generative Adversarial Network** (DCGAN).
This code uses PyTorch as backend.

This is an example of what it can generate:\
![](https://github.com/floboc/tiles-gan/blob/master/samples/gan_generated_tiles_2.jpg)

And when interpolating in its latent space:\
![](https://github.com/floboc/tiles-gan/blob/master/samples/gan_latent_space_interpolation.jpg)

**You can find more information and details about this source code in this article:** <https://playerone-studio.com/gan-2d-tiles-generative-adversarial-network>


## Dependencies

There are few dependencies:
- PyTorch (<https://pytorch.org/>)
- Numpy
- Python Imaging Library (PIL)


## How to use it

1. Create your own dataset:
- You can use the simple Processing (<https://processing.org/>) script provided in this repository to convert downloaded tilesets into individual tiles
- All tilesets must be in a folder with no other file, and tiles should be of the same size in all tilesets (here 32x32)
- Tiles will be saved as individual PNG files. Empty tiles will be omitted.

Here are some samples of tiles I used for my dataset:\
![](https://github.com/floboc/tiles-gan/blob/master/samples/training_images.jpg)

2. Train your model:
- Edit train.py so that the paths match that of your dataset (images have to be power of 2)
- Also adjust any settings as you want. The settings are detailed in the file gan.py
- run using "python train.py"

You should be patient as etting the first results can take some time. Here are some results during training:\
![](https://github.com/floboc/tiles-gan/blob/master/samples/gan_evolution_training_epoch.jpg)

3. Test your model:
- Edit test.py to match the path where you saved your model
- Also adjust latent space dimension if required
- run using "python test.py" to generate some test images in your output folder
- images will be saved to the same path as your model


**Enjoy!**
