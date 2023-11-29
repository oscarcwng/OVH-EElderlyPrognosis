from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

class MyData_train(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.classes = np.unique(y)
        self.y = np.array(y)
        self.tumorid = y.index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        transform = transforms.Compose([
                                        transforms.RandomAffine(degrees=(0,360),translate=(0.1,0.3)),
#                                         transforms.CenterCrop((512,512)),
                                        transforms.RandomRotation(360),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
#         print(Image.open(self.X[index]))
        
        image = transform(torch.tensor(np.moveaxis(np.array(Image.open(self.X[index])), -1, 0).astype(float))).float()
        label = self.y[index].astype(float)

        return image, label
    
class MyData_test(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.classes = np.unique(y)
        self.y = np.array(y)
        self.tumorid = y.index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        transform = transforms.Compose([
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
        image = transform(torch.tensor(np.moveaxis(np.array(Image.open(self.X[index])), -1, 0).astype(float))).float()
        label = self.y[index].astype(float)

        return image, label