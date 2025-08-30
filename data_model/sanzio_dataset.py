import os
import PIL.Image
import torch.nn as nn
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import PIL

class SanzioDataset(Dataset):
    def __init__(self, path):
        self.images = []
        
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, ...]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.images = self.load_images(path)
        
    def load_images(self, path):
        images = []
        for f_path in glob.glob(f'{path}/*.jpg'):
            images.append(f_path)
        return images
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        title = os.path.basename(im_path).split('.')[0]
        im = PIL.Image.open(im_path)
        im.show()
        im = self.transform(im)
        return im, title

if __name__ == "__main__":
    s = SanzioDataset("data/images")
    