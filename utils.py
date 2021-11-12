import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as  np

def create_advfile(dataset, epsilon, original, adv):
    firstpath = "./Adversarial/"+dataset
    if not os.path.exists(firstpath):
        os.mkdir(firstpath)
    firstpath += "/ori{}_adv{}".format(original, adv)
    if not os.path.exists(firstpath):
        os.mkdir(firstpath)
    firstpath += "/epsilon{}".format(epsilon)
    if not os.path.exists(firstpath):
        os.mkdir(firstpath)
    return firstpath

class  Mydataset(Dataset):
    def __init__(self, img_path, label_path=None,transform=None,target_transform=None):
        if label_path is None :
            file = open(img_path,'r',encoding='utf-8')
            imgs = []
            for line in file :
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0],int(words[1])))
            self.imgs = imgs
            self.labels = None
            self.transform = transform
            self.target_transfrom=target_transform
        else:
            self.imgs = np.load(img_path).reshape((-1,28,28,1))
            self.labels = np.load(label_path).reshape((-1))
            self.transform = transform
            self.target_transfrom=target_transform

    def __getitem__(self, item):
        if self.labels is None:
            fn, label = self.imgs[item]
            img = Image.open(fn).convert('RGB')
        else:
            img = self.imgs[item]
            label = self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)