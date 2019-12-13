import os
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision
import cv2
import torch

class Ref_Dataloader(data.Dataset):
    def __init__(self, path, transform=None):
        super(Ref_Dataloader, self).__init__()
        self.path = path
        self.files = []
        if transform is not None:
            self.transform = torchvision.transforms.Compose([transform, torchvision.transforms.ToTensor()])
        else:
            self.transform = torchvision.transforms.ToTensor()
        unpol_path = os.path.join(path, "unpol/")
        pol_path = os.path.join(path, "pol_1/")
        r_path = os.path.join(path, "reflection/")
        b_path = os.path.join(path, "background/")
        a0_path = os.path.join(path, "alpha0/")
        a1_path = os.path.join(path, "alpha1/")

        for _, _, filenames in os.walk(unpol_path):
            for name in filenames:
                unpol_file = os.path.join(unpol_path, name)
                pol_file = os.path.join(pol_path, name)
                r_file = os.path.join(r_path, name)
                b_file = os.path.join(b_path, name)
                name = name[:-4] + '.npy'
                a0_file = os.path.join(a0_path, name)
                a1_file = os.path.join(a1_path, name)
                self.files.append({
                    "unpol": unpol_file,
                    "pol": pol_file,
                    "r": r_file,
                    "b": b_file,
                    "a0": a0_file,
                    "a1": a1_file,
                    "name": name[:-4]
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        #### load the datas ####
        unpol = Image.open(datafiles["unpol"]).convert('RGB')
        pol = Image.open(datafiles["pol"]).convert('RGB')
        r = Image.open(datafiles["r"]).convert('RGB')
        b = Image.open(datafiles["b"]).convert('RGB')
        a0 = np.load(datafiles["a0"])[:, :, None]
        a1 = np.load(datafiles["a1"])[:, :, None]
        unpol = self.transform(unpol)
        pol = self.transform(pol)
        r = self.transform(r)
        b = self.transform(b)
        a0 = self.transform(a0)
        a1 = self.transform(a1)
        return (unpol, pol, r, b, a0, a1)
