#repository files
from utils import RandomNoise
from gan import GANOptions, GAN
from torchsummary import summary

#other dependencies
import random
import numpy
import torch
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image

#Set random seem for reproducibility
rnd_seed = 123456789 #datetime.now()
random.seed(rnd_seed)
numpy.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
print("Random Seed: ", rnd_seed)

#Change the settings to your own dataset paths
opts = GANOptions(data_path = "D:/GAN/Databases/tiles/", file_extension = "png", 
                  output_path = "D:/GAN/Results/tiles/",
                  image_size = 32)

#The settings I used for my article (https://playerone-studio.com/gan-2d-tiles-generative-adversarial-network)
opts.generator_batchnorm = True
opts.discriminator_batchnorm = True
opts.discriminator_dropout = 0.3
opts.generator_frequency = 1
opts.generator_dropout = 0.3
opts.label_softness = 0.2
opts.batch_size = 128
opts.epoch_number = 300

#Note: if you are using jupyter notebook you might need to disable workers: 
opts.workers_nbr = 0;

#List of tranformations applied to each input image
opts.transforms = [
    transforms.Resize(int(opts.image_size), Image.BICUBIC),
    transforms.ToTensor(), #do not forget to transform image into a tensor
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    #RandomNoise(0.01) #adding noise might prevent the discriminator from over-fitting
]

#Build GAN
model = GAN(opts)

#Display GAN architecture (note: only work if cuda is enabled)
summary(model.generator.cuda(), input_size=(opts.latent_dim, 1, 1))
summary(model.discriminator.cuda(), input_size=(opts.channels_nbr, opts.image_size, opts.image_size))

#Start training
model.train()

#Save model
model.save(opts.output_path)
