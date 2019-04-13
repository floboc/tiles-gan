from gan import GANOptions, GAN
from interpolation import lerp, slerp
from torchsummary import summary

#other dependencies
import random
import numpy
import torch
import torchvision.utils
import torchvision.transforms as transforms
from datetime import datetime


#This is ugly but for now it does the work...
#Change this so that it matches your model options
opts = GANOptions()
opts.latent_dim = 50
opts.image_size = 32
opts.channels_nbr = 3
opts.output_path = "D:/GAN/Results/tiles/models"


#Set random seem for reproducibility (if required)
rnd_seed = 123456789
random.seed(rnd_seed)
numpy.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
print("Random Seed: ", rnd_seed)


#CUDA
if torch.cuda.is_available():
	Tensor = torch.cuda.FloatTensor
	print("CUDA is available")
else:
	Tensor = torch.Tensor     
	print("CUDA is NOT available")

#load model
gan = GAN(opts)
gan.load(opts.output_path)

#display model info
summary(gan.generator.cuda(), input_size=(opts.latent_dim, 1, 1))
summary(gan.discriminator.cuda(), input_size=(opts.channels_nbr, opts.image_size, opts.image_size))


#IMAGE GENERATION

#test model
nrows = 16
ncols = 16
latent_vectors = Tensor(numpy.random.normal(0.0, 1.0, (nrows * ncols, opts.latent_dim, 1, 1)))
gen = gan.generator(latent_vectors).detach()

#save image
torchvision.utils.save_image(gen, opts.output_path + '/generated.png', nrow=ncols, normalize=True, padding=0)



#LATENT SPACE INTERPOLATION

#source and target latent representations
source = numpy.random.normal(0.0, 1.0, opts.latent_dim)
target = numpy.random.normal(0.0, 1.0, opts.latent_dim)
steps = 16
#interpolation in latent space
latents = []
for i in range(steps):
    factor = i / (steps - 1)
    val = slerp(factor, source, target)
    z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(Tensor(val), 0),2),3)
    latents.append(z)
latents = torch.cat(latents)

#apply model
gen = gan.generator(latents).detach()

#save image
torchvision.utils.save_image(gen, opts.output_path + '/interpolation.png', nrow=steps, normalize=True, padding=0)
