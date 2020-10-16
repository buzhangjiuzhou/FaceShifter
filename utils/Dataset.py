from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import numpy as np
import os
import cv2

# 利用pytorch的TensorDataset类
class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.N)

class Faces(TensorDataset):
    def __init__(self, data_path):
        self.datasets = image_list = glob.glob(f'{data_path}/*.*g')
        self.transformers = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        image_path = self.datasets[item]
        image = cv2.imread(image_path)
        image = Image.fromarray(image)
        
        return self.transformers(image)

    def __len__(self):
        return len(self.datasets)

class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders, batch_size):
        self.dataloaders = dataloaders
        self.batch_size = batch_size
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for i, data_loader in enumerate(self.dataloaders):
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        b = ''
        for data_iter in self.loader_iter:
            a = next(data_iter, None)
            if a is None:
                temp_dataloader = DataLoader(self.dataloaders[0].dataset, batch_size=self.batch_size, shuffle=True)
                a = next(iter(temp_dataloader))
                if b is None:
                    return None
                b = None
            out.append(a) # may raise StopIteration
        return tuple(out)