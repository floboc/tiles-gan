import torch
import os
import glob
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, extension, transforms_list, recursive=True):
        self.path = path
        self.transform = transforms.Compose(transforms_list)
        if recursive:
            self.files = sorted(glob.glob(path + '/**/*.' + extension, recursive=True))
        else:
            self.files = sorted(glob.glob(path + '/*.' + extension))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)];
        image = self.transform(Image.open(filepath).convert('RGB'))
        return image

class ReplayBuffer():
    def __init__(self, max_size=50, replace_probability = 0.5):
        assert (max_size > 0), 'Buffer size must be superior to 0'
        self.max_size = max_size
        self.replace_probability = replace_probability
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) < self.replace_probability:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

