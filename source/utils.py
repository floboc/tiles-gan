import random
import torch

#can be used as a transform to add random noise
class RandomNoise(object):
    def __init__(self, std):
         self.std = std
            
    def __call__(self, tensor):
        return tensor + self.std * torch.randn(tensor.size())

