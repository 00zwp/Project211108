from tqdm import tqdm
from torch_model import *
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import foolbox as fb
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#利用foolbox包生成对抗样本 —— 1.读取模型生成foolbox包装模型框架  2.attack的执行

Mymodel = Model(784,10)
checkpoint = torch.load('./models/mnist_Linear_model.pkl', map_location=torch.device('cpu'))
Mymodel.load_state_dict(checkpoint)
Mymodel.eval()

MNIST_path = './'
batch_size = 1
test_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                          train=False,
                                          transform=transforms.ToTensor()
                                          )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel = fb.PyTorchModel(Mymodel, bounds=bounds, preprocessing=None)

attack = fb.attacks.FGSM()

for i, (test_data,test_label) in enumerate(tqdm(test_loader)):
    if test_label == torch.max(Mymodel(Variable(test_data)),1)[1].data:
        raw, clipped, is_adv = attack(fmodel, test_data, test_label, epsilons=0.1)
        x = raw.data.numpy()
        x1 = clipped.data.numpy()

        if is_adv:
            print(test_label)
            print(torch.max(Mymodel(Variable(test_data)),1)[1].data)
            print(torch.max(Mymodel(Variable(raw)),1)[1].data)
            print(torch.max(Mymodel(Variable(clipped)),1)[1].data)
            plt.plot(x.reshape((28,28)),cmap='gray')
            plt.show()
            plt.imshow(x1.reshape((28,28)),cmap='gray')
            plt.show()
            quit(0)