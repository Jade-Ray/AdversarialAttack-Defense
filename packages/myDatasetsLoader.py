#%%
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io, transform
import os

def create_myLoader(func_datasets, root, download=False, transform=None, batch_size=1):
    train_datasets = func_datasets(root=root, train=True, download=download, transform=transform)
    lengthes = [int(0.8*len(train_datasets)), int(0.2*len(train_datasets))]
    train_datasets, val_datasets = random_split(train_datasets, lengthes)
    
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        func_datasets(root=root, train=False, download=download, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=0)
    
    size_train = len(train_loader.dataset)
    size_val = len(val_loader.dataset)
    size_test = len(test_loader.dataset)
    class_name = test_loader.dataset.classes

    return {'train': train_loader,
            'val': val_loader, 
            'test': test_loader, 
            'sizes': {'train': size_train, 'val': size_val, 'test':size_test},
            'class_names': class_name}

loader_MNIST = create_myLoader(datasets.MNIST, root='./Data/MNIST', download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]), batch_size=64)
loader_CIFAR10 = create_myLoader(datasets.CIFAR10, root='./Data/CIFAR10', download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]), batch_size=64)

class NIPS2017AdversaryCompetitionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.categories_frame = pd.read_csv(os.path.join(self.root_dir, 'categories.csv'))
        self.images_frame = pd.read_csv(os.path.join(self.root_dir, 'images.csv'))
        self.transform = transform
        self.class_names = []
        for i in range(len(self.categories_frame)):
            self.class_names.append(self.categories_frame.CategoryName[i])

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, index):
        img_name = os.path.join(os.path.join(self.root_dir, 'images'), self.images_frame.ImageId[index] + '.png')
        image = io.imread(img_name)
        target = self.images_frame.TrueLabel[index] - 1

        if self.transform:
            image = self.transform(image)
        
        return image, target


# %%
