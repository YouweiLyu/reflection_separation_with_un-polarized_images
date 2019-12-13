import os
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision
import cv2
import torch

class Real_Dataloader(data.Dataset):
    def __init__(self, path, transform=None):
        super(Real_Dataloader, self).__init__()
        self.path = path
        self.files = []
        if transform is not None:
            self.transform = torchvision.transforms.Compose([transform, torchvision.transforms.ToTensor()])
        else:
            self.transform = torchvision.transforms.ToTensor()
        pol00_path = os.path.join(path, "pol_1/")
        pol90_path = os.path.join(path, "pol_3/")
        print(pol00_path, pol90_path)
        
        for _, _, filenames in os.walk(pol00_path):
            for name in filenames:
                pol00_file = os.path.join(pol00_path, name)
                pol90_file = os.path.join(pol90_path, name)
                self.files.append({
                    "pol00": pol00_file,
                    "pol90": pol90_file,            
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        #### load the datas ####
        pol00 = cv2.imread(datafiles["pol00"], -1).astype(np.double)/65535.
        pol90 = cv2.imread(datafiles["pol90"], -1).astype(np.double)/65535.
        unpol = pol00 + pol90
        h, w = pol90.shape
        h, w = (h // 32) * 32, (w // 32) * 32
        unpol, pol00= np.dstack([unpol[:h, :w]]*3), np.dstack([pol00[:h, :w]]*3)
        unpol = self.transform(unpol)
        pol00 = self.transform(pol00)
        return (unpol, pol00)
