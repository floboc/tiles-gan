#repositiory
from data import ImageDataset, ReplayBuffer

#other dependencies
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import math
import numpy
import random
import torch
import torch.nn as nn
import torchvision.utils
from torchvision.utils import save_image

class GANOptions:
    def __init__(self, image_size = 32, n_epochs = 10000, latent_dim = 50, batch_size = 128, data_path = "", file_extension = "", output_path = "", workers_nbr = 8, transform_list = []):
        self.epoch_number = n_epochs #number of epochs
        self.latent_dim = latent_dim #size of latent space
        self.batch_size = batch_size #size of mini-batch
        self.label_softness = 0.1 #amplitude of the noise added to label when training discriminator
        
        self.data_path = data_path #path of the data
        self.file_extension = file_extension #file extension
        self.workers_nbr = workers_nbr #number of workers for data loading
        self.image_size = image_size #size of the image
        self.channels_nbr = 3 #number of channels in the image
        self.transforms = transform_list
        
        self.output_path = output_path #path for output
        self.save_interval = 100 #number of batches between saves
        
        self.generator_frequency = 1 #number of training steps for the generator (not recommended, keep it to 1)
        self.generator_dropout = 0.2 #drop-out probability for generator (0 = none)
        self.generator_batchnorm = True #use batch-normalization in generator
        self.generator_lr = 2e-4 #generator learning rate
        self.generator_beta1 = 0.5 #generator beta1 parameter (adam)
        
        self.discriminator_dropout = 0.2 #drop-out probability for discriminator (0 = none)
        self.discriminator_batchnorm = True #use batch-normalization in discriminator
        self.discriminator_lr = 2e-4 #discriminator learning rate
        self.discriminator_beta1 = 0.5 #discriminator beta1 parameter (adam)
        self.discriminator_kernel = 5 #kernel size for discriminator


#custom weight initialization
def gan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#GAN generator
class Generator(torch.nn.Module):

	#latent_dim: size of latent vectors
	#image_size: size of generated images (must be a power of 2)
	#channels_nbr: number of channels in the image (3 for RGB)
	#features_nbr: number of filters at the last stage
	#dropout: unit drop-out probability (0 for no drop-out)
	#batch_norm: if true, batch normalization will be used, otherise, instance normalization will be applied
    def __init__(self, latent_dim, image_size, channels_nbr = 3, features_nbr = 64, dropout = 0.2, batch_norm = True):
        super(Generator, self).__init__()
        
        layers = []

        #Compute model size
        in_features = latent_dim
        num_blocks = round(math.log2(image_size)) - 3
        out_features = features_nbr * (2 ** num_blocks)
        
        #First expansion layer (to 4x4 size)
        layers += [
            torch.nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if batch_norm else None,
            torch.nn.LeakyReLU(0.1, inplace=True)
        ]
        in_features = out_features
        out_features = out_features // 2
        
        #Transpose convolution blocks
        for i in range(num_blocks):
            layers += [
                torch.nn.Upsample(scale_factor=2),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0),
                torch.nn.Dropout2d(p=0.3, inplace=True) if dropout > 0 else None,
                torch.nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if batch_norm else None,
                torch.nn.InstanceNorm2d(out_features) if (not batch_norm) else None,
                torch.nn.LeakyReLU(0.1, inplace=True)
            ]
            in_features = out_features
            out_features = out_features // 2
        
        #Last conversion layer
        layers += [
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=in_features, out_channels=channels_nbr, kernel_size=3, stride=1, padding=0), 
            torch.nn.Tanh()
        ]
        
        #create sequential nn from layers list
        self.net = torch.nn.Sequential(*filter(lambda x: x is not None, layers))


    def forward(self, x):
        return self.net(x)

#GAN discriminator
class Discriminator(torch.nn.Module):

	#image_size: size of generated images (must be a power of 2)
	#channels_nbr: number of channels in the image (3 for RGB)
	#features_nbr: number of filters in the first stage
	#kernel_size: size of the convolution kernels
	#dropout: unit drop-out probability (0 for no drop-out)
	#batch_norm: if true, batch normalization will be used, otherwise, instance normalization will be applied
    def __init__(self, image_size, channels_nbr = 3, features_nbr = 64, kernel_size = 5, dropout = 0.2, batch_norm = True):
        super(Discriminator, self).__init__()
        
        layers = []
        
        #Compute model size
        in_features = channels_nbr
        out_features = features_nbr
        num_blocks = round(math.log2(image_size)) - 3
        
        #First layer (input is C x S x S, output is F x (S/2) x (S/2))
        layers += [
            torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True)
        ]
        in_features = out_features
        out_features *= 2
        
        #Transpose convolution blocks
        for i in range(num_blocks):
            layers += [
                torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, bias=False),
                torch.nn.Dropout2d(p=0.3, inplace=True) if dropout > 0 else None,
                torch.nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if batch_norm else None,
                torch.nn.InstanceNorm2d(out_features) if (not batch_norm) else None,
                torch.nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features
            out_features *= 2
        
        #Last layer (input is N x 4 x 4, output is 1 x 1 x 1)
        layers += [
            torch.nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        ]
        
        #create sequential nn from layers list
        self.net = torch.nn.Sequential(*filter(lambda x: x is not None, layers))

    def forward(self, x):
        return self.net(x)

#GAN class used for training
class GAN:
	#options a GANOption obect
    def __init__(self, options):
        self.discriminator = Discriminator(image_size=options.image_size, channels_nbr=options.channels_nbr, 
                                           features_nbr = 64, kernel_size = options.discriminator_kernel, 
                                           dropout=options.discriminator_dropout, batch_norm = options.discriminator_batchnorm)
        self.generator = Generator(latent_dim=options.latent_dim, image_size=options.image_size, 
                                   channels_nbr=options.channels_nbr, features_nbr = 64,
                                  dropout=options.generator_dropout, batch_norm = options.generator_batchnorm)
        self.options = options
        
	#saves the generator and discriminator to the path given
	#note: for now it doesn't save any option (like latent space dimension)
    def save(self, path):
        torch.save(self.generator.state_dict(), '%s/generator.model' % path)
        torch.save(self.discriminator.state_dict(), '%s/discriminator.model' % path)
        
	#loads a previously saved GAN model FOR EVALUATION
    def load(self, path):
        self.generator.load_state_dict(torch.load('%s/generator.model' % path))
        self.discriminator.load_state_dict(torch.load('%s/discriminator.model' % path))
        
        #set dropout and batch normalization layers to evaluation mode
        self.generator.eval()
        self.discriminator.eval()
        
	#starts training using the options given at initialization
    def train(self):
        #create folders
        os.makedirs('%s/images' % self.options.output_path, exist_ok=True)
        os.makedirs('%s/models' % self.options.output_path, exist_ok=True)

        #define dataset
        dataset = ImageDataset(path=self.options.data_path, extension=self.options.file_extension, transforms_list=self.options.transforms)
        data_loader = DataLoader(dataset, batch_size=self.options.batch_size, shuffle=True, num_workers=self.options.workers_nbr)
        print('Dataset has %s samples (%s batches)' % (len(dataset), len(data_loader)))
        
        #define losses
        discriminator_loss = torch.nn.BCELoss()
        
        #CUDA conversion
        #Note: this must be done before defining optimizers
        if torch.cuda.is_available():
            Tensor = torch.cuda.FloatTensor
            discriminator_loss = discriminator_loss.cuda()
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            print("CUDA is available")
        else:
            Tensor = torch.Tensor     
            print("CUDA is NOT available")
            
        #Optimizers
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.options.generator_lr, betas=(self.options.generator_beta1, 0.999))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.options.discriminator_lr, betas=(self.options.discriminator_beta1, 0.999))
        
        #Initialize weights
        self.generator.apply(gan_weights_init)
        self.discriminator.apply(gan_weights_init)
        
        #Buffer for Experience Replay
        fake_buffer = ReplayBuffer(max_size=self.options.batch_size * 2, replace_probability=0.5)
        
        #Note: softening the labels by a random amount for the DISCRIMINATOR training avoid approaching 0 loss too rapidly
        label_softness = 0.1

        for epoch in range(self.options.epoch_number):
            for (i, batch) in enumerate(data_loader):
                
                # --- DISCRIMINATOR ---
                
                #reset discriminator's gradient
                discriminator_optimizer.zero_grad()
                
                #TODO: occasionally flip labels
                
                #backward on real data
                real = Variable(batch.type(Tensor))
                batch_size = real.shape[0] #get batch-size
                real_labels = Variable(Tensor(numpy.random.uniform(0.0, self.options.label_softness, (batch_size, 1, 1, 1))))
                real_loss = discriminator_loss(self.discriminator(real), real_labels)
                real_loss.backward()
                
                #backward on fake data
                #z = Variable(Tensor(numpy.random.uniform(0.0, 1.0, (batch_size, self.options.latent_dim, 1, 1))))
                z = Variable(Tensor(numpy.random.normal(0.0, 1.0, (batch_size, self.options.latent_dim, 1, 1))))
                fake_labels = Variable(Tensor(numpy.random.uniform(1.0 - self.options.label_softness, 1.0, (batch_size, 1, 1, 1))))
                fake = self.generator(z)
                fake_buffered = fake_buffer.push_and_pop(fake)
                fake_loss = discriminator_loss(self.discriminator(fake_buffered.detach()), fake_labels)
                fake_loss.backward()
                
                #train discriminator
                discriminator_optimizer.step()

                
                # --- GENERATOR ---
                
                #train the generator multiple times to force it to get better
				#Note: this is not recommended
                for substep in range(self.options.generator_frequency):
                    #reset generator's gradient
                    generator_optimizer.zero_grad()

                    #backward on generator (we want fake ones to be considered real so we use real label)
                    #Discriminator loss converges rapidly to zero thus preventing the Generator from learning
                    # => Use a different training noise sample for the Generator
                    z2 = Variable(Tensor(numpy.random.normal(0.0, 1.0, (batch_size, self.options.latent_dim, 1, 1))))
                    fake2 = self.generator(z2)
                    real_labels2 = Variable(Tensor(0.0 * numpy.ones((batch_size, 1, 1, 1))))
                    g_loss = discriminator_loss(self.discriminator(fake2), real_labels2)
                    g_loss.backward()

                    #train generator
                    generator_optimizer.step()
                
                #log info
                print("epoch %i - batch %i: D=%f G=%f" % (epoch, i, (0.5 * (real_loss + fake_loss)), g_loss))
            
                if i % self.options.save_interval == 0:
                    #save some results
                    z = Tensor(numpy.random.normal(0.0, 1.0, (64, self.options.latent_dim, 1, 1)))
                    gen = self.generator(z)
                    save_image(gen, '%s/images/%s-%s.png' % (self.options.output_path, epoch, i), nrow=8, normalize=True)

